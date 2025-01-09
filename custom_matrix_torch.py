import torch
from gl0_and_gl1 import GL0Element, GL1Element

class MatrixElement:
    def __init__(self, value=None, left= None, right = None, up = None, down = None):
        self.value = value  # The value will be a GL1Element 
        self.left = None # Following are GL0Element
        self.right = None
        self.up = None
        self.down = None

    def __repr__(self):
        return f"MatrixElement(value={self.value}, l={self.left}, r={self.right}, u={self.up}, d={self.down})"

class CustomMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        # Initialize with MatrixElement instances
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
    Maps elements from image to the custom matrix using PyTorch tensors and GL0/GL1 elements.
    """
    m = image.shape[0]
    gl0 = GL0Element(2, 1, 1)
    Map = CustomMatrix(m - 1, m - 1)
    
    for i in range(m - 1):
        for j in range(m - 1):
            # Mapping vectors for each direction (up, down, left, right)
            pu = gl0.from_vector(image[i, j], image[i, j + 1])
            pd = gl0.from_vector(image[i + 1, j + 1], image[i + 1, j])
            pl = gl0.from_vector(image[i + 1, j], image[i, j])
            pr = gl0.from_vector(image[i, j + 1], image[i + 1, j + 1])

            # Set the directional relationships for the matrix elements
            Map[i, j].left = pl
            Map[i, j].right = pr
            Map[i, j].up = pu
            Map[i, j].down = pd

            # Multiply the edges (as GL0 elements)
            Edges_mul = pd * pr * pu * pl

            # Construct the N matrix (tensor)
            N = torch.tensor([[image[i, j] + image[i + 1, j + 1] - image[i + 1, j] - image[i, j + 1]]], dtype=torch.float32)

            # Perform reverse feedback mapping to GL1
            GL1 = gl0.reverse_feedback(Edges_mul, N)
            
            # Assign the computed GL1 element to the matrix element
            Map[i, j].value = GL1

    return Map
