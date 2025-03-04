from imports import *
from torch._higher_order_ops.associative_scan import associative_scan
from lifting import from_vector, kernel_gl1
device = "cuda" if torch.cuda.is_available() else "cpu"

# def from_vector(n, p, q, m, Xt, Xs):

#     fV = torch.eye(n + p).repeat(m, 1, 1)
#     fU = torch.eye(n + q).repeat(m, 1, 1)
#     dX = (Xs - Xt).to(device)

#     fV[:, 0, 0] = fU[:, 0, 0] = torch.exp(dX)
#     fV[:, 1, 1] = fU[:, 1, 1] = torch.exp(dX ** 2)
#     fV[:, 2, 0] = torch.sin(dX)
#     fV[:, 2, 1] = dX ** 5
#     fU[:, 0, 2] = dX ** 3
#     fU[:, 1, 2] = 7 * dX

#     return (fV.to(device), fU.to(device))

# def kernel_gl1(p1, p2, p3, p4):
#     return (p1+p3-p2-p4).to(device).unsqueeze(-1).unsqueeze(-1)

def reverse_feedback(n, tuple, N):

    M_V, M_U = tuple
    # Extract top and bottom rows for feedback
    rows, columns, m, _, _ = M_V.size()
    P = M_V[:, :, :, :n, :n].to(device) - torch.eye(n).repeat(rows, columns, m, 1, 1).to(device)
    B = M_U[:, :, :, :n, n:].to(device)
    R = M_V[:, :, :, n:, :n].to(device)

    top_row = torch.cat((P.to(device), B), dim=-1).to(device)
    bottom_row = torch.cat((R, N.to(device)), dim=-1).to(device)

    result_matrix = torch.cat((top_row, bottom_row), dim=-2)

    return result_matrix

@torch.compile
def to_tuple(n, p, q, image, from_vector, kernel_gl1):
    """
    Lifting Procedure - Maps elements from a batched image to the dictionary structure.
    Args:
        n, p, q: parameters
        image (torch.Tensor): Batched tensor of shape (batch_size, m, n)
    Returns:
        pytree structure containing all data, specifically in tensor form at leaves
    """

    batch_size, rows, columns = image.shape

    up = [[] for _ in range(rows-1)]
    down = [[] for _ in range(rows-1)]
    left = [[] for _ in range(rows-1)]
    right = [[] for _ in range(rows-1)]

    N = [[] for _ in range(rows-1)]

    for i in range(rows - 1):
        for j in range(columns - 1):

            if i == 0: 
                up[i].append(from_vector(n, p, q, batch_size, image[:, i, j], image[:, i, j + 1]))
            else:
                up[i].append(down[i-1][j])

            if j == 0:
                left[i].append(from_vector(n, p, q,batch_size, image[:, i + 1, j], image[:, i, j]))
            else:
                left[i].append(right[i][j-1])

            down[i].append(from_vector(n, p, q,batch_size, image[:, i + 1, j], image[:, i + 1, j + 1]))
            right[i].append(from_vector(n, p, q,batch_size, image[:, i + 1, j + 1], image[:, i, j + 1]))
            
            p1 = image[:, i, j]
            p2 = image[:, i + 1, j]
            p3 = image[:, i + 1, j + 1] 
            p4 = image[:, i, j + 1]
            N[i].append(kernel_gl1(p1, p2, p3, p4, p, q))

    def stack_grid(grid):
        a_rows = []
        b_rows = []
        for row in grid:  # Each row corresponds to fixed i
            a_items = []
            b_items = []
            for elem in row:  # Elements are tuples (a, b) for each j
                a, b = elem
                a_items.append(a)
                b_items.append(b)
            a_rows.append(torch.stack(a_items))  # Stack along j: (columns-1, ...)
            b_rows.append(torch.stack(b_items))
        return torch.stack(a_rows), torch.stack(b_rows)  # Stack along i: (rows-1, columns-1, ...)

    # Stack up, down, left, right into tensors preserving (i, j) structure
    up1, up2 = stack_grid(up)
    down1, down2 = stack_grid(down)
    left1, left2 = stack_grid(left)
    right1, right2 = stack_grid(right)

    # Stack N into a tensor preserving (i, j) structure
    N_tensor = torch.stack([torch.stack(Ni) for Ni in N])

    # Compute Edges_mul (order matches to_custom_matrix)
    Edges_mul = (
        down1 @ right1 @ torch.linalg.inv(up1) @ torch.linalg.inv(left1),
        down2 @ right2 @ torch.linalg.inv(up2) @ torch.linalg.inv(left2)
    )

    face_tensor = reverse_feedback(n, Edges_mul, N_tensor)

    # Return flat tuple of tensors
    final_tuple = (up1, up2, down1, down2, left1, left2, right1, right2, face_tensor)
    return final_tuple

def horizontal_compose_with(elem1, elem2):
    
    new_value = elem1[2] @ elem2[8] @ torch.linalg.inv(elem1[3]) @ elem1[8]
    
    new_up_1 = elem1[0] @ elem2[0]
    new_up_2 = elem1[1] @ elem2[1]

    new_down_1 = elem1[2] @ elem2[2]
    new_down_2 = elem1[3] @ elem2[3]

    tuple = (new_up_1, new_up_2, new_down_1, new_down_2, elem1[4], elem1[5], elem2[6], elem2[7], new_value)
    
    return tuple

def vertical_compose_with(elem1, elem2):

    new_value = elem1[8] @ elem1[4] @ elem2[8] @ torch.linalg.inv(elem1[5]) 

    new_left_1 = elem1[4] @ elem2[4]
    new_left_2 = elem1[5] @ elem2[5]

    new_right_1 = elem1[6] @ elem2[6]
    new_right_2 = elem1[7] @ elem2[7]

    tuple = (elem2[0], elem2[1], elem1[2], elem1[3], new_left_1, new_left_2, new_right_1, new_right_2, new_value)
    return tuple

@torch.compile
def cal_aggregate(horizontal_compose_with, vertical_compose_with, elements):

    aggregate_horizontal = associative_scan(horizontal_compose_with, elements, dim=1, combine_mode='generic')
    # ( tensor(rows * columns * batch_size * n+p * n+p), tensor(rows * columns * ...), ..., tensor(rows * columns * batch_size * n+p * n+q) )

    flipped = tuple(t.flip(0) for t in aggregate_horizontal)
    flipped_scanned = associative_scan(vertical_compose_with, flipped, dim=0, combine_mode='generic')
    aggregate = tuple(t.flip(0) for t in flipped_scanned)

    return aggregate

# Dims of face_tensor = rows * columns * batch_size * n+p * n+q
# ([[1, 2, 3],
# [2, 3, 4]], ....)

n = 2
p = 1
q = 1
batch_size = 2
torch.manual_seed(42)
images = torch.rand(batch_size, 5, 5).to(device)
print('initial = ',images[0])

elements = to_tuple(n, p, q, images, from_vector, kernel_gl1)

# associativity check
elem1 = tuple(x[0][0] for x in elements)
elem2 = tuple(x[0][1] for x in elements)
elem3 = tuple(x[0][2] for x in elements)
print(elem3[-1])
# print((horizontal_compose_with(horizontal_compose_with(elem1, elem2), elem3))[-1])
# print((horizontal_compose_with(elem1, horizontal_compose_with(elem2, elem3)))[-1])
assert torch.allclose((horizontal_compose_with(horizontal_compose_with(elem1, elem2), elem3))[-1], (horizontal_compose_with(elem1, horizontal_compose_with(elem2, elem3)))[-1])

for i in range(100):
    start_time = time.time()
    aggregate = cal_aggregate(horizontal_compose_with, vertical_compose_with, elements)
    end_time = time.time()
    # print(end_time - start_time)

print("aggregate = ", aggregate[-1][0][-1][0])

# Game plan
# Data into dictionary 
# Define operation
# Associate Scan

# Problem
# 1. Parallization for lifting procedure - batch size and entire image or just batch size.
# 2. Vertical comp down to up