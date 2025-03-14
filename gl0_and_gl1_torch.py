from imports import *
device = "cuda" if torch.cuda.is_available() else "cpu"

class GL0Element:
    def __init__(self, m, n, p, q, M_V=None, M_U=None):
        self.n = n
        self.p = p
        self.q = q
        self.m = m  # batch size


        if M_V is None:
            M_V = torch.eye(n + p, dtype=torch.int32).repeat(m, 1, 1)
        if M_U is None:
            M_U = torch.eye(n + q, dtype=torch.int32).repeat(m, 1, 1)

        self.M_V = M_V.to(device)
        self.M_U = M_U.to(device)
        self.tuple = (self.M_V, self.M_U)

    def almost_equal(self, other):
        # Check if all elements in the batch are almost equal
        return torch.allclose(self.M_V, other.M_V) and torch.allclose(self.M_U, other.M_U)

    def inv(self):
        # Perform batch inversion
        M_V_new = torch.linalg.inv(self.M_V)
        M_U_new = torch.linalg.inv(self.M_U)
        return GL0Element(self.m, self.n, self.p, self.q, M_V_new, M_U_new)

    @staticmethod
    def random_element(m, n, p, q):
        # Generate m random elements with batch tensors
        P = torch.eye(n).repeat(m, 1, 1) + torch.randint(1, 10, (m, n, n))
        S = torch.eye(p).repeat(m, 1, 1) + torch.randint(1, 10, (m, p, p))
        D = torch.eye(q).repeat(m, 1, 1) + torch.randint(1, 10, (m, q, q))

        R = torch.randint(0, 10, (m, p, n))
        B = torch.randint(0, 10, (m, n, q))

        M_V = torch.cat([
            torch.cat([P, torch.zeros((m, n, p))], dim=2),
            torch.cat([R, S], dim=2)
        ], dim=1)

        M_U = torch.cat([
            torch.cat([P, B], dim=2),
            torch.cat([torch.zeros((m, q, n)), D], dim=2)
        ], dim=1)

        return GL0Element(m, n, p, q, M_V, M_U)

    def __mul__(self, other):
        if not isinstance(other, GL0Element):
            raise ValueError("Multiplication is only defined between GL0Element objects.")
        
        # Batch matrix multiplication
        M_V_new = self.M_V @ other.M_V
        M_U_new = self.M_U @ other.M_U

        return GL0Element(self.m, self.n, self.p, self.q, M_V_new, M_U_new)

    def act_on(self, h):
        if h.matrix.shape != (self.m, self.n + self.p, self.n + self.q):
            raise ValueError(f"Matrix h must have dimensions ({self.m}, {self.n + self.p}, {self.n + self.q}).")

        # Batch operation for action
        if torch.any(torch.linalg.det(self.M_U) == 0):
            raise ValueError("At least one of the M_U matrices is not invertible.")

        # Perform batch matrix multiplication
        M_U_inv = torch.linalg.inv(self.M_U.to(device))  
        Action = torch.bmm(torch.bmm(self.M_V.to(device), h.matrix.to(device)), M_U_inv)

        return GL1Element(self.m, self.n, self.p, self.q, Action)

    @staticmethod
    def reverse_feedback(gl0_element, N):
        if not isinstance(gl0_element, GL0Element):
            raise ValueError("Input must be a GL0Element.")

        n, p, q = gl0_element.n, gl0_element.p, gl0_element.q
        m = gl0_element.m

        M_V = gl0_element.M_V
        M_U = gl0_element.M_U

        if M_V.shape != (m, n + p, n + p) or M_U.shape != (m, n + q, n + q):
            raise ValueError("M_V or M_U have invalid dimensions for GL0.")


        # Extract top and bottom rows for feedback
        P = M_V[:, :n, :n] - torch.eye(n).repeat(m, 1, 1).to(device)
        B = M_U[:, :n, n:].to(device)
        R = M_V[:, n:, :n].to(device)

        top_row = torch.cat((P.to(device), B), dim=2).to(device)
        bottom_row = torch.cat((R, N.to(device)), dim=2).to(device)

        result_matrix = torch.cat((top_row, bottom_row), dim=1)

        return GL1Element(m, n, p, q, result_matrix)

    @staticmethod
    def from_vector(m, Xt, Xs):
        n, p, q = 2, 1, 1
        fV = torch.eye(n + p).repeat(m, 1, 1)
        fU = torch.eye(n + q).repeat(m, 1, 1)
        dX = Xs - Xt

        fV[:, 0, 0] = fU[:, 0, 0] = torch.exp(dX)
        fV[:, 1, 1] = fU[:, 1, 1] = torch.exp(dX ** 2)
        fV[:, 2, 0] = torch.sin(dX)
        fV[:, 2, 1] = dX ** 5
        fU[:, 0, 2] = dX ** 3
        fU[:, 1, 2] = 7 * dX
 
        return GL0Element(m, n, p, q, fV, fU)

    @staticmethod
    def kernel_gl1(p1, p2, p3, p4):
        return (p1+p3-p2-p4).unsqueeze(-1).unsqueeze(-1)

class GL1Element:
    def __init__(self, m, n, p, q, matrix=None):
        """
        Initialize a GL-1 element for a batch of matrices.
        """
        self.m = m
        self.n = n
        self.p = p
        self.q = q

        if matrix is None:
            self.matrix = torch.eye(n + p, n + q, dtype=torch.int32).unsqueeze(0).repeat(m, 1, 1).to(device)
        else:
            self.matrix = matrix.to(device)

        # Validate dimensions
        self.validate()

    def validate(self):
        """
        Ensure the matrix has the correct dimensions (n+p) x (n+q).
        """
        expected_shape = (self.n + self.p, self.n + self.q)
        if self.matrix.shape[1:] != expected_shape:
            raise ValueError(f"Matrix must have dimensions {expected_shape}, got {self.matrix.shape[1:]}.")

    def inv(self):
        m, n, p, q = self.m, self.n, self.p, self.q
        matrix1 = self.matrix

        # Extract submatrices and compute inverses
        P = matrix1[:, :n, :n].to(device) + torch.eye(n).unsqueeze(0).expand(m, -1, -1).to(device)
        B = matrix1[:, :n, n:].to(device)
        R = matrix1[:, n:, :n].to(device)
        N = matrix1[:, n:, n:].to(device)

        P_inv = torch.linalg.inv(P)

        P_new = -(P - torch.eye(n).unsqueeze(0).expand(m, -1, -1).to(device)) @ P_inv
        B_new = (P - torch.eye(n).unsqueeze(0).expand(m, -1, -1).to(device)) @ P_inv @ B - B
        R_new = -R @ P_inv
        N_new = R @ P_inv @ B - N

        top_row = torch.cat((P_new, B_new), dim=2)
        bottom_row = torch.cat((R_new, N_new), dim=2)
        result_matrix = torch.cat((top_row, bottom_row), dim=1)

        return GL1Element(self.m, self.n, self.p, self.q, result_matrix)

    def __mul__(self, other):
        """
        Multiplication for GL-1 elements across a batch.
        """
        if not isinstance(other, GL1Element):
            raise ValueError("Multiplication is only defined between GL1Element objects.")

        if self.m != other.m or self.n != other.n or self.p != other.p or self.q != other.q:
            raise ValueError("GL1Element objects must have matching dimensions for multiplication.")

        matrix1 = other.matrix
        matrix2 = self.matrix

        # Extract submatrices for batch operations
        P1 = matrix1[:, :self.n, :self.n] + torch.eye(self.n, dtype=matrix1.dtype).unsqueeze(0).expand(self.m, -1, -1).to(device)
        B1 = matrix1[:, :self.n, self.n:].to(device)
        R1 = matrix1[:, self.n:, :self.n].to(device)
        N1 = matrix1[:, self.n:, self.n:].to(device)

        P2 = matrix2[:, :self.n, :self.n] + torch.eye(self.n, dtype=matrix2.dtype).unsqueeze(0).expand(self.m, -1, -1).to(device)
        B2 = matrix2[:, :self.n, self.n:].to(device)
        R2 = matrix2[:, self.n:, :self.n].to(device)
        N2 = matrix2[:, self.n:, self.n:].to(device)

        # Perform bach-wise multiplication
        new_P = P2 @ P1 - torch.eye(self.n).unsqueeze(0).expand(self.m, -1, -1).to(device)  # P'P - I_n
        new_B = P2 @ B1 + B2  # P'B + B'
        new_R = R2 @ P1 + R1  # R'P + R
        new_N = R2 @ B1 + N1 + N2  # R'B + N + N'

        # Combine blocks into the resulting matrix
        top_row = torch.cat((new_P, new_B), dim=2)
        bottom_row = torch.cat((new_R, new_N), dim=2)
        result_matrix = torch.cat((top_row, bottom_row), dim=1)

        return GL1Element(self.m, self.n, self.p, self.q, result_matrix)

    def feedback(self):
        """
        Compute the feedback matrix for the GL-1 element across a batch.
        """
        M_V = torch.zeros((self.m, self.n + self.p, self.n + self.p))
        M_U = torch.zeros((self.m, self.n + self.q, self.n + self.q))

        # Top-left block of M_V
        M_V[:, :self.n, :self.n] = self.matrix[:, :self.n, :self.n] + torch.eye(self.n).unsqueeze(0).expand(self.m, -1, -1).to(device)

        # Bottom-left block of M_V
        M_V[:, self.n:, :self.n] = self.matrix[:, self.n:, :self.n].to(device)
        M_V[:, self.n:, self.n:] = torch.eye(self.p).unsqueeze(0).expand(self.m, -1, -1).to(device)

        # Top-left block of M_U
        M_U[:, :self.n, :self.n] = self.matrix[:, :self.n, :self.n] + torch.eye(self.n).unsqueeze(0).expand(self.m, -1, -1).to(device)
        M_U[:, :self.n, self.n:] = self.matrix[:, :self.n, self.n:].to(device)

        # Bottom-left block of M_U
        M_U[:, self.n:, self.n:] = torch.eye(self.q).unsqueeze(0).expand(self.m, -1, -1).to(device)

        return GL0Element(self.m, self.n, self.p, self.q, M_V, M_U)

    @staticmethod
    def random_element(m, n, p, q):
        """
        Generate a random GL1Element for a batch of m elements.
        """
        # Generate random invertible n x n block
        top_left = torch.eye(n).unsqueeze(0).repeat(m, 1, 1).to(device) + torch.randint(1, 10, (m, n, n)).to(device)

        # Generate random blocks for the rest of the matrix
        top_right = torch.randint(0, 10, (m, n, q)).to(device)  # Top-right block
        bottom_left = torch.randint(0, 10, (m, p, n)).to(device)  # Bottom-left block
        bottom_right = torch.randint(0, 10, (m, p, q)).to(device)  # Bottom-right block

        # Construct the full matrix
        top = torch.cat((top_left, top_right), dim=2)  # Concatenate along columns
        bottom = torch.cat((bottom_left, bottom_right), dim=2)  
        matrix = torch.cat((top, bottom), dim=1)

        return GL1Element(m, n, p, q, matrix)

def test_equivariance():
    m, n, p, q = 1, 2, 1, 1

    # Generate random elements using PyTorch
    gl0 = GL0Element.random_element(m, n, p, q)  # Assuming random_element is adapted for PyTorch
    gl1 = GL1Element.random_element(m, n, p, q)

    # Compute feedback
    feedback_result = gl1.feedback()

    # Left-hand side (LHS)
    LHS = gl0 * gl1.feedback() * gl0.inv()

    # Right-hand side (RHS)
    RHS = gl0.act_on(gl1).feedback()

    # Assert equivalence using torch.allclose
    # print(f"{LHS.tuple[0]=}", f"{RHS.tuple[0]=}")
    assert torch.allclose(LHS.tuple[0], RHS.tuple[0], atol=0.001), "First tuple elements are not close!"
    # print(f"{LHS.tuple[1]=}", f"{RHS.tuple[1]=}")
    assert torch.allclose(LHS.tuple[1], RHS.tuple[1], atol=0.001), "Second tuple elements are not close!"

def test_tau_morphism():
    m, n, p, q = 1, 2, 1, 1
    gl1_a = GL1Element.random_element(m, n, p, q)
    gl1_b = GL1Element.random_element(m, n, p, q)

    product = gl1_a * gl1_b
    feedback_product = product.feedback()

    feedback_a = gl1_a.feedback()
    feedback_b = gl1_b.feedback()

    composed_feedback = feedback_a * feedback_b 

    assert torch.allclose(feedback_product.tuple[0], composed_feedback.tuple[0]), "Feedback morphism fails for M_V"
    assert torch.allclose(feedback_product.tuple[1], composed_feedback.tuple[1]), "Feedback morphism fails for M_U"

def test_peiffer_identity():
    m, n, p, q = 1, 2, 1, 1
    gl1_a = GL1Element.random_element(m, n, p, q)
    gl1_b = GL1Element.random_element(m, n, p, q)

    LHS = (gl1_a.feedback()).act_on(gl1_b)
    RHS = gl1_a * gl1_b * gl1_a.inv()

    Zeros = torch.zeros((n+p, n+q)).to(device)
    Identity_gl1 = gl1_a * (gl1_a.inv())
    # print(Identity_gl1.matrix)

    assert torch.allclose(Zeros, Identity_gl1.matrix, atol=0.001)
    # print(f"{LHS.matrix=}", f"{RHS.matrix=}")
    assert torch.allclose(LHS.matrix, RHS.matrix, atol=0.001)

if __name__ == "__main__":
    test_tau_morphism()
    test_equivariance()
    test_peiffer_identity()
