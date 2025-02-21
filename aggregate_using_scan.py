from imports import *
from torch._higher_order_ops.associative_scan import associative_scan
device = "cuda" if torch.cuda.is_available() else "cpu"

def from_vector(n, p, q, m, Xt, Xs):

    fV = torch.eye(n + p).repeat(m, 1, 1)
    fU = torch.eye(n + q).repeat(m, 1, 1)
    dX = (Xs - Xt).to(device)

    fV[:, 0, 0] = fU[:, 0, 0] = torch.exp(dX)
    fV[:, 1, 1] = fU[:, 1, 1] = torch.exp(dX ** 2)
    fV[:, 2, 0] = torch.sin(dX)
    fV[:, 2, 1] = dX ** 5
    fU[:, 0, 2] = dX ** 3
    fU[:, 1, 2] = 7 * dX

    return (fV.to(device), fU.to(device))

def kernel_gl1(p1, p2, p3, p4):
    return (p1+p3-p2-p4).to(device).unsqueeze(-1).unsqueeze(-1)

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

            if j == 0: 
                up[j].append(from_vector(n, p, q,batch_size, image[:, i, j], image[:, i, j + 1]))
            else:
                up[j].append(down[j-1][i])

            if i == 0:
                left[j].append(from_vector(n, p, q,batch_size, image[:, i + 1, j], image[:, i, j]))
            else:
                left[j].append(right[j][i-1])

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
    N_tensor = torch.stack(stacked_tensors).to(device)

    Edges_mul = (down_tuple[0] @ right_tuple[0] @ torch.linalg.inv(up_tuple[0]) @ torch.linalg.inv(left_tuple[0]),
                    down_tuple[0] @ right_tuple[0] @ torch.linalg.inv(up_tuple[0]) @ torch.linalg.inv(left_tuple[0]))

    face_tensor = reverse_feedback(n, Edges_mul, N_tensor)

    final_tuple = (up_tuple[0], up_tuple[1], down_tuple[0], down_tuple[1], left_tuple[0], left_tuple[1], right_tuple[0], right_tuple[1], face_tensor)

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

    tuple = (elem2[0], elem2[1], elem1[2], elem1[1], new_left_1, new_left_2, new_right_1, new_right_2, new_value)
    return tuple

@torch.compile
def cal_aggregate(horizontal_compose_with, vertical_compose_with, elements):

    aggregate_horizontal = associative_scan(horizontal_compose_with, elements, dim=1, combine_mode='generic')

    flipped = tuple(t.flip(0) for t in aggregate_horizontal)
    flipped_scanned = associative_scan(vertical_compose_with, flipped, dim=0, combine_mode='generic')
    aggregate = tuple(t.flip(0) for t in flipped_scanned)

    return aggregate

n = 2
p = 1
q = 1
batch_size = 2
images = torch.rand(batch_size, 5, 5).to(device)
elements = to_tuple(n, p, q, images, from_vector, kernel_gl1)


for i in range(100):
    start_time = time.time()
    aggregate = cal_aggregate(horizontal_compose_with, vertical_compose_with, elements)
    end_time = time.time()
    print(end_time - start_time)

print(aggregate[0].size())

# Game plan
# Data into dictionary 
# Define operation
# Associate Scan

# Problem
# 1. Parallization for lifting procedure - batch size and entire image or just batch size.
# 2. Vertical comp down to up