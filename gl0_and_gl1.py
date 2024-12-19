# import os
# import sys

# # Get the directory of the current script
# current_dir = os.path.dirname(__file__)

# # Append the parent directory to the system path
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(parent_dir)

# Now you can import your modules
from imports import *

class GL0Element:
    def __init__(self,n,p,q, M_V=None, M_U=None):
        self.n = n
        self.p = p
        self.q = q

        if M_V is None:
            M_V = np.eye( n+p, dtype=int )
        if M_U is None:
            M_U = np.eye( n+q, dtype=int )

        self.tuple = (M_V, M_U)

    def inv(self):
        M_V, M_U = self.tuple
        return GL0Element(self.n, self.p, self.q, np.linalg.inv(M_V), np.linalg.inv(M_U))

    @staticmethod
    def random_element(n,p,q):
        # TODO generate integer matrices
        # Invertible blocks
        P = np.eye(n, dtype=int) + np.random.randint(1, 10, size=(n, n))  # Random integers in [1, 10)
        S = np.eye(p, dtype=int) + np.random.randint(1, 10, size=(p, p))
        D = np.eye(q, dtype=int) + np.random.randint(1, 10, size=(q, q))

        # Taking random blocks
        R = np.random.randint(0, 10, size=(p, n))  # Random integers in [0, 10)
        B = np.random.randint(0, 10, size=(n, q))  # Random integers in [0, 10)

        # Constructing f_V and f_U
        M_V = np.block([
            [P, np.zeros((n, p), dtype=int)],  # Ensure zero block is integer
            [R, S]
        ])

        M_U = np.block([
            [P, B],
            [np.zeros((q, n), dtype=int), D]  # Ensure zero block is integer
        ])

        return GL0Element(n, p, q, M_V, M_U)

    def __mul__(self, other):
        """
        Define multiplication for GL0Element.
        """
        if not isinstance(other, GL0Element):
            raise ValueError("Multiplication is only defined between GL0Element objects.")

        # Unpack the matrices
        M_V_self, M_U_self = self.tuple
        M_V_other, M_U_other = other.tuple

        # Perform matrix multiplication
        M_V_new =  M_V_self @ M_V_other
        M_U_new =  M_U_self @ M_U_other

        return GL0Element(self.n, self.p, self.q, M_V_new, M_U_new)

    def act_on(self, h):
        """
        Perform the action M_V @ h @ M_U^-1.
        h is expected to be a compatible matrix.
        """
        M_V, M_U = self.tuple

        # Check if h has compatible dimensions
        if h.matrix.shape != (self.n + self.p, self.n + self.q):
            raise ValueError(f"Matrix h must have dimensions ({self.n + self.p}, {self.n + self.q}).")

        # Ensure B is invertible
        if np.linalg.det(M_U) == 0:
            raise ValueError("Matrix M_U is not invertible.")

        Action = M_V @ h.matrix @ np.linalg.inv(M_U)

        # Perform the action
        return GL1Element(self.n, self.p, self.q, Action)

    @staticmethod
    def from_vector(Xt, Xs):
        """
        Path steps X_t & X_s
        considering n = 2, p = 1, q = 1
        """
        n = 2
        p = q = 1
        fV = np.eye(n+p)
        fU = np.eye( n+q )
        dX = Xt - Xs
        fV[0, 0] = fU[0, 0] = np.exp(dX)
        fV[0, 1] = fV[0, 2] = fV[1, 0] = fV[1, 2] = fU[0, 1] = fU[1, 0] = 0
        fV[1, 1] = fU[1, 1] = np.exp((dX)**2)
        fV[2, 0] = np.sin(dX)
        fV[2, 1] = (dX)**5
        fV[2, 2] = fU[2, 2] = 1
        fU[0, 2] = (dX)**3
        fU[1, 2] = 7*(dX)
        fU[2, 0] = fU[2, 1] = 0
        return GL0Element(2, 1, 1, fV, fU)

    # TODO make from_vector parameterisable

    @staticmethod
    def reverse_feedback(gl0_element, N):
        """
        Reverse the feedback mapping from GL0 to GL1.

        Parameters:
            gl0_element (GL0Element): An element of GL0 represented as (M_V, M_U).

        Returns:
            GL1Element: The corresponding GL1Element reconstructed from (M_V, M_U).
        """
        if not isinstance(gl0_element, GL0Element):
            raise ValueError("Input must be a GL0Element.")

        n, p, q = gl0_element.n, gl0_element.p, gl0_element.q
        M_V, M_U = gl0_element.tuple

        # Validate that M_V and M_U dimensions are correct
        if M_V.shape != (n + p, n + p) or M_U.shape != (n + q, n + q):
            raise ValueError("M_V or M_U have invalid dimensions for GL0.")

        # Extract components from M_V and M_U
        P = M_V[:n, :n] - np.eye(n)  # P = top-left block of M_V minus Id_n
        B = M_V[:n, n:]              # B = top-right block of M_V
        R = M_U[n:, :n]              # R = bottom-left block of M_U

        # Combine blocks into GL1 matrix
        top_row = np.hstack((P, B))
        bottom_row = np.hstack((R, N))
        result_matrix = np.vstack((top_row, bottom_row))

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
            self.matrix = np.eye(n + p, n + q, dtype=int)
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
        return GL1Element(self.n, self.p, self.q, np.linalg.inv(self.matrix))

    def __mul__(self, other):
        """
        Multiplication for GL-1 elements.
        """
        if not isinstance(other, GL1Element):
            raise ValueError("Multiplication is only defined between GL1Element objects.")

        if self.n != other.n or self.p != other.p or self.q != other.q:
            raise ValueError("GL1Element objects must have matching dimensions for multiplication.")

        n, p, q = self.n, self.p, self.q
        matrix1 = self.matrix
        matrix2 = other.matrix

        # Extract submatrices
        P1 = matrix1[:n, :n] + np.eye(n)
        B1 = matrix1[:n, n:]
        R1 = matrix1[n:, :n]
        N1 = matrix1[n:, n:]

        P2 = matrix2[:n, :n] + np.eye(n)
        B2 = matrix2[:n, n:]
        R2 = matrix2[n:, :n]
        N2 = matrix2[n:, n:]

        # Perform block-wise multiplication
        new_P = P2 @ P1 - np.eye(n)  # P'P - I_n
        new_B = P2 @ B1 + B2                    # P'B + B'
        new_R = R2 @ P1 + R1                    # R'P + R
        new_N = R2 @ B1 + N1 + N2               # R'B + N + N'

        # Combine blocks into the resulting matrix
        top_row = np.hstack((new_P, new_B))
        bottom_row = np.hstack((new_R, new_N))
        result_matrix = np.vstack((top_row, bottom_row))

        return GL1Element(n, p, q, result_matrix)

    def feedback(self):
        """
        Compute the feedback matrix for the GL-1 element.
        """
        # Create zero matrices for the output components
        M_V = np.zeros((self.n + self.p, self.n + self.p))
        M_U = np.zeros((self.n + self.q, self.n + self.q))

        # Top-left block of M_V
        M_V[:self.n, :self.n] = self.matrix[:self.n, :self.n]  + np.eye(self.n)# P
        M_V[:self.n, self.n:] = 0  # zeroes for the top-right corner of M_V

        # Bottom-left block of M_V
        M_V[self.n:, :self.n] = self.matrix[self.n:, :self.n]  # R
        M_V[self.n:, self.n:] = np.eye(self.p)  # identity matrix of size p for the bottom-right block

        # Top-left block of M_U
        M_U[:self.n, :self.n] = self.matrix[:self.n, :self.n]  + np.eye(self.n)# P
        M_U[:self.n, self.n:] = self.matrix[:self.n, self.n:]  # B

        # Bottom-left block of M_U
        M_U[self.n:, :self.n] = 0  # zeroes for the bottom-left block of M_U
        M_U[self.n:, self.n:] = np.eye(self.q)  # identity matrix of size q for the bottom-right block

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
        top_left = np.eye(n,n, dtype = int) + np.random.randint(1, 10, size=(n, n))  # Random integers in [1, 10)

        # Generate random blocks for the rest of the matrix
        top_right = np.random.randint(0, 10, size=(n, q))  # Top-right block
        bottom_left = np.random.randint(0, 10, size=(p, n))  # Bottom-left block
        bottom_right = np.random.randint(0, 10, size=(p, q))  # Bottom-right block

        # Construct the full matrix
        matrix = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])

        return GL1Element(n, p, q, matrix)

np.random.seed(42)
def equivariance():

    n, p, q = 2, 1, 1
    gl0 = GL0Element.random_element(n, p, q)
    gl1 = GL1Element.random_element(n, p, q)

    feedback_result = gl1.feedback()

    LHS = Ad_g0_feedback_g1 = gl0 * gl1.feedback() * gl0.inv()
    # = gl0.act_on(gl1.feedback())
    RHS = feedback_action_result = gl0.act_on(gl1).feedback()
    # ad_g0 = (A .. A^-1, B .. B^-1)

    assert np.allclose(LHS.tuple[0], RHS.tuple[0])
    assert np.allclose(LHS.tuple[1], RHS.tuple[1])


def tau_morphism():
    n, p, q = 2, 1, 1
    gl1_a = GL1Element.random_element(n, p, q)
    gl1_b = GL1Element.random_element(n, p, q)

    product = gl1_a.__mul__(gl1_b)
    feedback_product = product.feedback()

    feedback_a = gl1_a.feedback()
    feedback_b = gl1_b.feedback()

    composed_feedback = feedback_b * feedback_a

    assert np.array_equal(feedback_product.tuple[0], composed_feedback.tuple[0]), "Feedback morphism fails for M_V"
    assert np.array_equal(feedback_product.tuple[1], composed_feedback.tuple[1]), "Feedback morphism fails for M_U"

def peiffer_identity():
    n, p, q = 2, 1, 1
    gl1 = GL1Element.random_element(n, p, q)
    gl2 = GL1Element.random_element(n, p, q)
    print("gl1 = ", gl1.matrix)
    print("gl2 = ", gl2.matrix)
    print("gl1 feedback = ", gl1.feedback().tuple)

    LHS = (gl1.feedback()).act_on(gl2)
    print("LHS = ", LHS.matrix)
    RHS = gl1 * gl2 * gl1.inv()
    print("gl1 inv = ", gl1.inv().matrix)
    print("RHS = ", RHS.matrix)

    assert np.allclose(LHS.matrix, RHS.matrix)

if __name__ == "__main__":
    tau_morphism()
    equivariance()
    peiffer_identity()
