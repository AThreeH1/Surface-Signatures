from imports import *
from gl0_and_gl1_torch import GL0Element, GL1Element

device = "cuda" if torch.cuda.is_available() else "cpu"

class TwoCell:
    def __init__(self, value=None, left=None, right=None, up=None, down=None):
        """
        A TwoCell represents a batch of 2D grid cells, where each cell has GL1 or GL0 elements in its
        four directions (left, right, up, down) and a central value.
        """
        self.value = value  # GL1Element (batched)
        self.left = left    # GL0Element (batched)
        self.right = right  # GL0Element (batched)
        self.up = up        # GL0Element (batched)
        self.down = down    # GL0Element (batched)

    def validate(self):
        """
        Validates the consistency condition for each TwoCell in the batch.
        """
        print(f"{(self.value.feedback().tuple[0]).device=}")
        assert torch.allclose(
            (self.down * self.right * self.up.inv() * self.left.inv()).tuple[0],
            self.value.feedback().tuple[0],
            atol=1e-6
        ), "Validation failed for the TwoCell batch."

        assert torch.allclose(
            (self.down * self.right * self.up.inv() * self.left.inv()).tuple[1],
            self.value.feedback().tuple[1],
            atol=1e-6
        ), "Validation failed for the TwoCell batch."

    def clone(self):
        """
        Creates a deep copy of the TwoCell, preserving batched structure.
        """
        return TwoCell(
            value=self.value,
            left=self.left,
            right=self.right,
            up=self.up,
            down=self.down
        )

    def horizontal_compose_with(self, other):
        """
        Horizontally composes the current TwoCell batch with another, assuming batched operations.
        """
        assert self.right.almost_equal(other.left), "Horizontal composition failed: `right` and `left` mismatch."

        value = (self.down.act_on(other.value)) * self.value
        left = self.left
        right = other.right
        up = self.up * other.up
        down = self.down * other.down

        return TwoCell(value, left, right, up, down)

    def vertical_compose_with(self, other):
        """
        Vertically composes the current TwoCell batch with another, assuming batched operations.
        """
        assert self.up.almost_equal(other.down), "Vertical composition failed: `up` and `down` mismatch."

        value = self.value * (self.left.act_on(other.value))
        left = self.left * other.left
        right = self.right * other.right
        up = other.up
        down = self.down

        return TwoCell(value, left, right, up, down)

    def __repr__(self):
        return (
            f"TwoCell(value={self.value}, "
            f"left={self.left}, right={self.right}, "
            f"up={self.up}, down={self.down})"
        )

class GridOf2Cells:
    def __init__(self, batch_size, rows, cols):
        self.rows = rows
        self.cols = cols
        self.batch_size = batch_size
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
        return f"GridOf2Cells(rows={self.rows}, cols={self.cols}, batch_size={self.batch_size}, matrix={as_string})"

@torch.compile
def to_custom_matrix(image, from_vector, kernel_gl1):
    """
    Maps elements from a batched image to the custom batched matrix structure.
    Args:
        image (torch.Tensor): Batched tensor of shape (batch_size, m, n)
    Returns:
        GridOf2Cells: Mapped batched structure
    """

    batch_size, m, n = image.shape
    gl0 = GL0Element(batch_size, 2, 1, 1)
    Map = GridOf2Cells(batch_size, m - 1, n - 1)
    
    for i in range(m - 1):
        for j in range(n - 1):

            if i == 0: 
                pu = from_vector(batch_size, image[:, i, j], image[:, i, j + 1])
            else:
                pu = Map[i-1, j].down

            if j == 0:
                pl = from_vector(batch_size, image[:, i + 1, j], image[:, i, j])
            else:
                pl = Map[i, j-1].right

            # Create batched GL0 elements
            # pu = from_vector(batch_size, image[:, i, j], image[:, i, j + 1])
            pd = from_vector(batch_size, image[:, i + 1, j], image[:, i + 1, j + 1])
            # pl = from_vector(batch_size, image[:, i + 1, j], image[:, i, j])
            pr = from_vector(batch_size, image[:, i + 1, j + 1], image[:, i, j + 1])

            # Assign to grid
            Map[i, j].left = pl
            Map[i, j].right = pr
            Map[i, j].up = pu
            Map[i, j].down = pd

            # Compute Edges_mul in batch
            Edges_mul = pd * pr * pu.inv() * pl.inv()

            # Compute N (batched tensor)
            p1 = image[:, i, j]
            p2 = image[:, i + 1, j]
            p3 = image[:, i + 1, j + 1] 
            p4 = image[:, i, j + 1]
            N = kernel_gl1(p1, p2, p3, p4) # Shape (batch_size, 1, 1)

            # Create GL1 elements in batch
            GL1 = gl0.reverse_feedback(Edges_mul, N)
            Map[i, j].value = GL1

    return Map

if __name__ == "__main__":
    batch_size = 1  # Number of batches
    m = 3           # Rows in each image
    n = 4           # Columns in each image

    def from_vector(m, Xt, Xs):
        n, p, q = 2, 1, 1
        fV = torch.eye(n + p).repeat(m, 1, 1).to(device)
        fU = torch.eye(n + q).repeat(m, 1, 1).to(device)
        dX = (Xs - Xt).to(device)

        fV[:, 0, 0] = fU[:, 0, 0] = torch.exp(dX)
        fV[:, 1, 1] = fU[:, 1, 1] = torch.exp(dX ** 2)
        fV[:, 2, 0] = torch.sin(dX)
        fV[:, 2, 1] = dX ** 5
        fU[:, 0, 2] = dX ** 3
        fU[:, 1, 2] = 7 * dX
 
        return GL0Element(m, n, p, q, fV, fU)

    def kernel_gl1(p1, p2, p3, p4):
        return (p1+p3-p2-p4).unsqueeze(-1).unsqueeze(-1).to(device)

    torch.manual_seed(42)
    image = torch.rand(batch_size, m, n)  # Generate batched random images

    # Map the batched image
    ImageBatch = to_custom_matrix(image, from_vector, kernel_gl1)

    # Validate each TwoCellBatch in the grid for all batches
    for i in range(m - 1):
        for j in range(n - 1):
            ImageBatch[i, j].validate()  # Validate each batch separately

    # Example of accessing specific elements and printing their tuples for the first batch
    print("A = ", ImageBatch[0, 0].down.tuple[0], "B = ", ImageBatch[1, 0].up.tuple[0])