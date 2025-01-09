import torch
from gl0_and_gl1_torch import GL0Element, GL1Element

import torch

class TwoCell:
    def __init__(self, value=None, left=None, right=None, up=None, down=None):
        self.value = value
        self.left = left
        self.right = right
        self.up = up
        self.down = down

    def validate(self):
        assert torch.allclose((self.down * self.right * self.up.inv() * self.left.inv()).tuple[0], self.value.feedback().tuple[0])
        # print(f"{(self.down * self.right * self.up.inv() * self.left.inv()).tuple[1]=}", f"{self.value.feedback().tuple[1]=}")
        assert torch.allclose((self.down * self.right * self.up.inv() * self.left.inv()).tuple[1], self.value.feedback().tuple[1], atol = 0.001)

    def clone(self):
        # Cloning the TwoCell instance
        new_elem = TwoCell(
            value = self.value,
            left = self.left,
            right = self.right,
            up = self.up,
            down = self.down
        )
        return new_elem

    def horizontal_compose_with(self, other):
        assert self.right.almost_equal(other.left)

        value = self.down.act_on(other.value) * self.value
        left = self.left
        right = other.right
        up = self.up * other.up
        down = self.down * other.down

        return TwoCell(value, left, right, up, down)

    def vertical_compose_with(self, other):
        assert self.up.almost_equal(other.down)

        value = self.value * (self.left.act_on(other.value))
        left = self.left * other.left
        right = self.right * other.right
        up = other.up
        down = self.down

        return TwoCell(value, left, right, up, down)

    def __repr__(self):
        return f"TwoCell(value={self.value}, l={self.left}, r={self.right}, u={self.up}, d={self.down})"


class GridOf2Cells:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = [[TwoCell() for _ in range(cols)] for _ in range(rows)]

    def __getitem__(self, indices):
        row, col = indices
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.matrix[row][col]
        else:
            raise IndexError("Matrix indices out of range")

    def __setitem__(self, indices, value):
        row, col = indices
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.matrix[row][col] = value
        else:
            raise IndexError("Matrix indices out of range")

    def __repr__(self):
        as_string = [",".join([repr(v) for v in row]) for row in self.matrix]
        return f"GridOf2Cells(rows={self.rows}, cols={self.cols}, matrix={as_string})"


def mapping(image):
    """
    Maps elements from an image (2D PyTorch tensor) to the custom matrix.
    """
    m, n = image.shape
    gl0 = GL0Element(2, 1, 1)
    Map = GridOf2Cells(m - 1, n - 1)

    for i in range(m - 1):
        for j in range(n - 1):
            pu = gl0.from_vector(image[i, j], image[i, j + 1])
            pd = gl0.from_vector(image[i + 1, j], image[i + 1, j + 1])
            pl = gl0.from_vector(image[i + 1, j], image[i, j])
            pr = gl0.from_vector(image[i + 1, j + 1], image[i, j + 1])

            Map[i, j].left = pl
            Map[i, j].right = pr
            Map[i, j].up = pu
            Map[i, j].down = pd

            Edges_mul = pd * pr * pu.inv() * pl.inv()
            N = torch.tensor([[image[i, j] + image[i + 1, j + 1] - image[i + 1, j] - image[i, j + 1]]])

            GL1 = gl0.reverse_feedback(Edges_mul, N)
            Map[i, j].value = GL1

    return Map


if __name__ == "__main__":
    m = 3
    n = 4
    torch.manual_seed(42)
    image = torch.rand(m, n)

    Image = mapping(image)
    for i in range(m-1):
        for j in range(n-1):
            Image[i,j].validate()

    print("A = ",Image[0,0].down.tuple,"B = ", Image[1,0].up.tuple)