import torch

class GL0Element:
    def __init__(self, n, p, q, M_V=None, M_U=None):
        self.n = n
        self.p = p
        self.q = q

        if M_V is None:
            M_V = torch.eye(n + p, dtype=torch.int32)
        if M_U is None:
            M_U = torch.eye(n + q, dtype=torch.int32)

        self.tuple = (M_V, M_U)

    def almost_equal(self, other):
        return torch.allclose(self.tuple[0], other.tuple[0]) and torch.allclose(self.tuple[1], other.tuple[1])

    def inv(self):
        M_V, M_U = self.tuple
        return GL0Element(self.n, self.p, self.q, torch.linalg.inv(M_V), torch.linalg.inv(M_U))

    @staticmethod
    def random_element(n, p, q):
        P = torch.eye(n) + torch.randint(1, 10, (n, n))
        S = torch.eye(p) + torch.randint(1, 10, (p, p))
        D = torch.eye(q) + torch.randint(1, 10, (q, q))

        R = torch.randint(0, 10, (p, n))
        B = torch.randint(0, 10, (n, q))

        M_V = torch.cat([
            torch.cat([P, torch.zeros((n, p))], dim=1),
            torch.cat([R, S], dim=1)
        ], dim=0)

        M_U = torch.cat([
            torch.cat([P, B], dim=1),
            torch.cat([torch.zeros((q, n)), D], dim=1)
        ], dim=0)


        return GL0Element(n, p, q, M_V, M_U)

    def __mul__(self, other):
        if not isinstance(other, GL0Element):
            raise ValueError("Multiplication is only defined between GL0Element objects.")

        M_V_self, M_U_self = self.tuple
        M_V_other, M_U_other = other.tuple

        M_V_new = M_V_self @ M_V_other
        M_U_new = M_U_self @ M_U_other

        return GL0Element(self.n, self.p, self.q, M_V_new, M_U_new)

    def act_on(self, h):
        M_V, M_U = self.tuple

        if h.matrix.shape != (self.n + self.p, self.n + self.q):
            raise ValueError(f"Matrix h must have dimensions ({self.n + self.p}, {self.n + self.q}).")

        if torch.linalg.det(M_U) == 0:
            raise ValueError("Matrix M_U is not invertible.")

        Action = M_V @ h.matrix @ torch.linalg.inv(M_U)

        return GL1Element(self.n, self.p, self.q, Action)

    @staticmethod
    def from_vector(Xt, Xs):
        n, p, q = 2, 1, 1
        fV = torch.eye(n + p)
        fU = torch.eye(n + q)
        dX = Xs - Xt

        fV[0, 0] = fU[0, 0] = torch.exp(dX)
        fV[1, 1] = fU[1, 1] = torch.exp(dX ** 2)
        fV[2, 0] = torch.sin(dX)
        fV[2, 1] = dX ** 5
        fU[0, 2] = dX ** 3
        fU[1, 2] = 7 * dX

        return GL0Element(2, 1, 1, fV, fU)

    @staticmethod
    def reverse_feedback(gl0_element, N):
        if not isinstance(gl0_element, GL0Element):
            raise ValueError("Input must be a GL0Element.")

        n, p, q = gl0_element.n, gl0_element.p, gl0_element.q
        M_V, M_U = gl0_element.tuple

        if M_V.shape != (n + p, n + p) or M_U.shape != (n + q, n + q):
            raise ValueError("M_V or M_U have invalid dimensions for GL0.")

        P = M_V[:n, :n] - torch.eye(n)
        B = M_U[:n, n:]
        R = M_V[n:, :n]

        top_row = torch.cat((P, B), dim=1)
        bottom_row = torch.cat((R, N), dim=1)
        result_matrix = torch.cat((top_row, bottom_row), dim=0)

        return GL1Element(n, p, q, result_matrix)

class GL1Element:
    def __init__(self, n, p, q, matrix=None):
        """
        Initialize a GL-1 element.
        """
        self.n = n
        self.p = p
        self.q = q

        if matrix is None:
            self.matrix = torch.eye(n + p, n + q, dtype=torch.int32)
        else:
            self.matrix = matrix

        # Validate dimensions
        self.validate()

    def validate(self):
        """
        Ensure the matrix has the correct dimensions (n+p) x (n+q).
        """
        expected_shape = (self.n + self.p, self.n + self.q)
        if self.matrix.shape != expected_shape:
            raise ValueError(f"Matrix must have dimensions {expected_shape}, got {self.matrix.shape}.")

    def inv(self):
        n, p, q = self.n, self.p, self.q
        matrix1 = self.matrix

        # Extract submatrices
        P = matrix1[:n, :n] + torch.eye(n)
        B = matrix1[:n, n:]
        R = matrix1[n:, :n]
        N = matrix1[n:, n:]

        P_inv = torch.linalg.inv(P)

        P_new = -(P - torch.eye(n)) @ P_inv
        B_new = (P - torch.eye(n)) @ P_inv @ B - B
        R_new = -R @ P_inv
        N_new = R @ P_inv @ B - N

        top_row = torch.hstack((P_new, B_new))
        bottom_row = torch.hstack((R_new, N_new))
        result_matrix = torch.vstack((top_row, bottom_row))

        return GL1Element(n, p, q, result_matrix)

    def __mul__(self, other):
        """
        Multiplication for GL-1 elements.
        """
        if not isinstance(other, GL1Element):
            raise ValueError("Multiplication is only defined between GL1Element objects.")

        if self.n != other.n or self.p != other.p or self.q != other.q:
            raise ValueError("GL1Element objects must have matching dimensions for multiplication.")

        n, p, q = self.n, self.p, self.q
        matrix1 = other.matrix
        matrix2 = self.matrix

        # Extract submatrices
        P1 = matrix1[:n, :n] + torch.eye(n, dtype=matrix1.dtype)
        B1 = matrix1[:n, n:]
        R1 = matrix1[n:, :n]
        N1 = matrix1[n:, n:]

        P2 = matrix2[:n, :n] + torch.eye(n, dtype=matrix2.dtype)
        B2 = matrix2[:n, n:]
        R2 = matrix2[n:, :n]
        N2 = matrix2[n:, n:]

        # Perform block-wise multiplication
        new_P = P2 @ P1 - torch.eye(n)  # P'P - I_n
        new_B = P2 @ B1 + B2                    # P'B + B'
        new_R = R2 @ P1 + R1                    # R'P + R
        new_N = R2 @ B1 + N1 + N2               # R'B + N + N'

        # Combine blocks into the resulting matrix
        top_row = torch.hstack((new_P, new_B))
        bottom_row = torch.hstack((new_R, new_N))
        result_matrix = torch.vstack((top_row, bottom_row))

        return GL1Element(n, p, q, result_matrix)

    def feedback(self):
        """
        Compute the feedback matrix for the GL-1 element.
        """
        M_V = torch.zeros((self.n + self.p, self.n + self.p))
        M_U = torch.zeros((self.n + self.q, self.n + self.q))

        # Top-left block of M_V
        M_V[:self.n, :self.n] = self.matrix[:self.n, :self.n] + torch.eye(self.n)  # P
        
        # Bottom-left block of M_V
        M_V[self.n:, :self.n] = self.matrix[self.n:, :self.n]  # R
        M_V[self.n:, self.n:] = torch.eye(self.p) 

        # Top-left block of M_U
        M_U[:self.n, :self.n] = self.matrix[:self.n, :self.n] + torch.eye(self.n)  # P
        M_U[:self.n, self.n:] = self.matrix[:self.n, self.n:]  # B
        
        # Bottom-left block of M_U
        M_U[self.n:, self.n:] = torch.eye(self.q)  # identity matrix of size q

        return GL0Element(self.n, self.p, self.q, M_V, M_U)

    @staticmethod
    def random_element(n, p, q):
        """
        Generate a random GL1Element.

        Conditions:
        - Shape: (n + p) x (n + q)
        - All integer values.
        - The first n x n block (top-left) is invertible.
        - Other elements are random integers.
        """
        # Generate a random invertible n x n block
        top_left = torch.eye(n) + torch.randint(1, 10, (n, n))

        # Generate random blocks for the rest of the matrix
        top_right = torch.randint(0, 10, (n, q))  # Top-right block
        bottom_left = torch.randint(0, 10, (p, n))  # Bottom-left block
        bottom_right = torch.randint(0, 10, (p, q))  # Bottom-right block

        # Construct the full matrix
        top = torch.cat((top_left, top_right), dim=1)  # Concatenate along columns
        bottom = torch.cat((bottom_left, bottom_right), dim=1)  
        matrix = torch.cat((top, bottom), dim=0)

        return GL1Element(n, p, q, matrix)

def equivariance():
    n, p, q = 2, 1, 1

    # Generate random elements using PyTorch
    gl0 = GL0Element.random_element(n, p, q)  # Assuming random_element is adapted for PyTorch
    gl1 = GL1Element.random_element(n, p, q)

    # Compute feedback
    feedback_result = gl1.feedback()

    # Left-hand side (LHS)
    LHS = gl0 * gl1.feedback() * gl0.inv()

    # Right-hand side (RHS)
    RHS = gl0.act_on(gl1).feedback()

    # Assert equivalence using torch.allclose
    assert torch.allclose(LHS.tuple[0], RHS.tuple[0]), "First tuple elements are not close!"
    # print(f"{LHS.tuple[1]=}", f"{RHS.tuple[1]=}")
    assert torch.allclose(LHS.tuple[1], RHS.tuple[1], atol=0.001), "Second tuple elements are not close!"

def tau_morphism():
    n, p, q = 2, 1, 1
    gl1_a = GL1Element.random_element(n, p, q)
    gl1_b = GL1Element.random_element(n, p, q)

    product = gl1_a * gl1_b
    feedback_product = product.feedback()

    feedback_a = gl1_a.feedback()
    feedback_b = gl1_b.feedback()

    composed_feedback = feedback_a * feedback_b 

    assert torch.allclose(feedback_product.tuple[0], composed_feedback.tuple[0]), "Feedback morphism fails for M_V"
    assert torch.allclose(feedback_product.tuple[1], composed_feedback.tuple[1]), "Feedback morphism fails for M_U"

def peiffer_identity():
    n, p, q = 2, 1, 1
    gl1_a = GL1Element.random_element(n, p, q)
    gl1_b = GL1Element.random_element(n, p, q)

    LHS = (gl1_a.feedback()).act_on(gl1_b)
    RHS = gl1_a * gl1_b * gl1_a.inv()

    Zeros = torch.zeros((n+p, n+q))
    Identity_gl1 = gl1_a * (gl1_a.inv())
    # print(Identity_gl1.matrix)

    assert torch.allclose(Zeros, Identity_gl1.matrix, atol=0.001)
    # print(f"{LHS.matrix=}", f"{RHS.matrix=}")
    assert torch.allclose(LHS.matrix, RHS.matrix, atol=0.001)

if __name__ == "__main__":
    tau_morphism()
    equivariance()
    peiffer_identity()
