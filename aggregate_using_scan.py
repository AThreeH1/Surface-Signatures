from imports import *
from torch._higher_order_ops.associative_scan import associative_scan
device = "cuda" if torch.cuda.is_available() else "cpu"

def from_vector(n, p, q, m, Xt, Xs):

    fV = torch.eye(n + p).repeat(m, 1, 1)
    fU = torch.eye(n + q).repeat(m, 1, 1)
    dX = Xs - Xt

    fV[:, 0, 0] = fU[:, 0, 0] = torch.exp(dX)
    fV[:, 1, 1] = fU[:, 1, 1] = torch.exp(dX ** 2)
    fV[:, 2, 0] = torch.sin(dX)
    fV[:, 2, 1] = dX ** 5
    fU[:, 0, 2] = dX ** 3
    fU[:, 1, 2] = 7 * dX

    return (fV, fU)


def kernel_gl1(p1, p2, p3, p4):
    return (p1+p3-p2-p4).unsqueeze(-1).unsqueeze(-1)

def reverse_feedback(n, tuple, N):

    M_V, M_U = tuple
    # Extract top and bottom rows for feedback
    rows, columns, m, _, _ = M_V.size()
    P = M_V[:, :, :, :n, :n] - torch.eye(n).repeat(rows, columns, m, 1, 1).to(device)
    B = M_U[:, :, :, :n, n:].to(device)
    R = M_V[:, :, :, n:, :n].to(device)

    top_row = torch.cat((P.to(device), B), dim=-1).to(device)
    bottom_row = torch.cat((R, N.to(device)), dim=-1).to(device)

    result_matrix = torch.cat((top_row, bottom_row), dim=-2)

    return result_matrix

def to_dictionary(n, p, q, image, from_vector, kernel_gl1):
    """
    Maps elements from a batched image to the dictionary structure.
    Args:
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

    for i in range(columns - 1):
        for j in range(rows - 1):

            if i == 0: 
                up[j].append(from_vector(n, p, q,batch_size, image[:, i, j], image[:, i, j + 1]))
            else:
                up[j].append(down[i-1][j])

            if j == 0:
                left[j].append(from_vector(n, p, q,batch_size, image[:, i + 1, j], image[:, i, j]))
            else:
                print(i, j)
                left[j].append(right[i][j-1])

            down[j].append(from_vector(n, p, q,batch_size, image[:, i + 1, j], image[:, i + 1, j + 1]))
            right[j].append(from_vector(n, p, q,batch_size, image[:, i + 1, j + 1], image[:, i, j + 1]))
            
            p1 = image[:, i, j]
            p2 = image[:, i + 1, j]
            p3 = image[:, i + 1, j + 1] 
            p4 = image[:, i, j + 1]
            N[j].append(kernel_gl1(p1, p2, p3, p4))

    up_stacked = [tuple(torch.stack(tensors) for tensors in zip(*sublist)) for sublist in zip(*up)]
    up_tuple = tuple(torch.stack(tensors) for tensors in zip(*up_stacked))

    down_stacked = [tuple(torch.stack(tensors) for tensors in zip(*sublist)) for sublist in zip(*down)]
    down_tuple = tuple(torch.stack(tensors) for tensors in zip(*down_stacked))

    left_stacked = [tuple(torch.stack(tensors) for tensors in zip(*sublist)) for sublist in zip(*left)]
    left_tuple = tuple(torch.stack(tensors) for tensors in zip(*left_stacked))

    right_stacked = [tuple(torch.stack(tensors) for tensors in zip(*sublist)) for sublist in zip(*right)]
    right_tuple = tuple(torch.stack(tensors) for tensors in zip(*right_stacked))

    stacked_tensors = [torch.stack(tensors) for tensors in N]
    N_tensor = torch.stack(stacked_tensors)

    Edges_mul = (down_tuple[0] @ right_tuple[0] @ torch.linalg.inv(up_tuple[0]) @ torch.linalg.inv(left_tuple[0]),
                    down_tuple[0] @ right_tuple[0] @ torch.linalg.inv(up_tuple[0]) @ torch.linalg.inv(left_tuple[0]))

    face_tensor = reverse_feedback(n, Edges_mul, N)

    dictionary = {
    'up': up_tuple,
    'down': down_tuple,
    'left': left_tuple,
    'right': right_tuple,
    'value': face_tensor}

    return dictionary

def horizontal_compose_with(elem1, elem2):
    
    new_value = (
        elem1['down'][0] @ elem2['value'] @ torch.linalg.inv(elem1['down'][1]) @ elem1['value']
    )
    new_up = tuple(u1 * u2 for u1, u2 in zip(elem1['up'], elem2['up']))
    new_down = tuple(d1 * d2 for d1, d2 in zip(elem1['down'], elem2['down']))

    return {
        'up': new_up,
        'down': new_down,
        'left': elem1['left'],   # left = left_1
        'right': elem2['right'], # right = right_2
        'value': new_value
    }

def vertical_compose_with(elem1, elem2):
    new_value = (
        elem1['value'] @ elem1['left'][0] @ elem2['value'] @ torch.linalg.inv(elem1['left'][1]) 
    )
    new_left = tuple(u1 @ u2 for u1, u2 in zip(elem1['left'], elem2['left']))
    new_right = tuple(d1 @ d2 for d1, d2 in zip(elem1['right'], elem2['right']))

    return {
        'up': elem2['up'],   # left = left_1
        'down': elem1['down'], # right = right_2
        'left': new_left,
        'right': new_right,
        'value': new_value
    }

def cal_aggregate(horizontal_compose_with, vertical_compose_with, elements):
    aggregate_horizontal = associative_scan(horizontal_compose_with, elements, dim=1)

    flipped = {key: (tuple(tensor.flip(0) for tensor in value) if isinstance(value, tuple) 
                              else value.flip(0))
                        for key, value in elements.items()}
    flipped_scanned = associative_scan(vertical_compose_with, flipped, dim=0)
    aggregate = {key: (tuple(tensor.flip(0) for tensor in value) if isinstance(value, tuple) 
                          else value.flip(0))
                    for key, value in flipped_scanned.items()}
    return aggregate

n = 2
p = 1
q = 1
batch_size = 2
images = torch.rand(batch_size, 5, 5)
elements = to_dictionary(n, p, q, images, from_vector, kernel_gl1)
aggregate = cal_aggregate(horizontal_compose_with, vertical_compose_with, elements)

print(aggregate)

# Game plan
# Data into dictionary 
# Define operation
# Associate Scan

# Problem
# 1. Parallization for lifting procedure - batch size and entire image or just batch size.
# 2. Vertical comp down to up