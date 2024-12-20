from imports import *
from gl0_and_gl1 import GL0Element, GL1Element

"""## Mapping to a new data structure"""

class MatrixElement:
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None
        self.up = None
        self.down = None

    def __repr__(self):
        return f"MatrixElement(value={self.value}, l={self.left}, r={self.right}, u={self.up}, d={self.down})"

class CustomMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = [[MatrixElement() for _ in range(cols)] for _ in range(rows)]

    def __getitem__(self, indices):
        row, col = indices
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.matrix[row][col]
        else:
            raise IndexError("Matrix indices out of range")

    def __setitem__(self, indices, value):
        row, col = indices
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.matrix[row][col].value = value
        else:
            raise IndexError("Matrix indices out of range")

def mapping(image):
    """
    maps elements from image to a new data structure
    """
    m = image.shape[0]
    gl0 = GL0Element(2, 1, 1)
    Map = CustomMatrix(m-1, m-1)
    for i in range(m-1):
        for j in range(m-1):
            pu = gl0.from_vector(image[i, j], image[i, j+1])
            pd = gl0.from_vector(image[i+1, j+1], image[i+1, j])
            pl = gl0.from_vector(image[i+1, j], image[i, j])
            pr = gl0.from_vector(image[i, j+1], image[i+1, j+1])

            Map[i, j].left = pl
            Map[i, j].right = pr
            Map[i, j].up = pu
            Map[i, j].down = pd

            Edges_mul = pd * pr * pu * pl
            N = [[image[i, j] + image[i+1, j+1] - image[i+1, j] - image[i, j+1]]]

            GL1 = gl0.reverse_feedback(Edges_mul, N)
            Map[i, j].value = GL1
    return Map