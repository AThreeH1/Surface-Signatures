from imports import *
from torch._higher_order_ops.associative_scan import associative_scan
from lifting import from_vector, kernel_gl1
device = "cuda" 
torch.set_default_dtype(torch.float64)

def reverse_feedback(n, tuple, N):
    """
    Maps multiplication edge elements(path around the surface) to face element. 
    Args:
        n: parameters of GL0 and GL1
        tuple: Edge element in GL0 using pd*pr*pu*pl (Edge elements of a face - down, right, up, left)
        N: Bottom right element of GL1 matrix
    Returns:
        Face element GL1 matrix(tensor)
    """
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

# @torch.compile
def to_tuple(n, p, q, image, from_vector, kernel_gl1):

    """
    Lifting Procedure - Maps elements from a batched image to the dictionary structure.
    Args:
        n, p, q: parameters of GL0 and GL1
        image (torch.Tensor): Batched tensor of shape (batch_size, row, col)
        from_vector and kernel_gl1: Lifting functions
    Returns:
        pytree structure containing all data, specifically in tensor form.

        returns [upU, upV, down1, down2, left1, left2, right1, right2, face_tensor]
            upU         ~ (row-1) x (col-1) x bs x (n+p) x (n+p)
            upV         ~ row x col x bs x (n+q) x (n+q) # TODO fix row col everywhere
            TODO
            face_tensor ~ row x col x bs x (n+p) x (n+q)
    """

    batch_size, rows, columns = image.shape

    up = [[] for _ in range(rows-1)]
    down = [[] for _ in range(rows-1)]
    left = [[] for _ in range(rows-1)]
    right = [[] for _ in range(rows-1)]

    N = [[] for _ in range(rows-1)]

    # TODO use 'global' tensor operations
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
        for row in grid:  
            a_items = []
            b_items = []
            for elem in row:  
                a, b = elem
                a_items.append(a)
                b_items.append(b)
            a_rows.append(torch.stack(a_items))  
            b_rows.append(torch.stack(b_items))
        return torch.stack(a_rows), torch.stack(b_rows)  

    upV, upU = stack_grid(up)
    down1, down2 = stack_grid(down) # TODO change for all
    left1, left2 = stack_grid(left)
    right1, right2 = stack_grid(right)

    # Stack N into a tensor preserving (rows, columns) structure
    N_tensor = torch.stack([torch.stack(Ni) for Ni in N])

    # Compute Edges_mul (order matches to_custom_matrix)
    Edges_mul = (
        down1 @ right1 @ torch.linalg.inv(upV) @ torch.linalg.inv(left1),
        down2 @ right2 @ torch.linalg.inv(upU) @ torch.linalg.inv(left2)
    )

    face_tensor = reverse_feedback(n, Edges_mul, N_tensor)

    # Return flat tuple of tensors
    final_tuple = (upV, upU, down1, down2, left1, left2, right1, right2, face_tensor)
    return final_tuple

def gl1_mul_tensor(mat_left, mat_right, n):
    """
    Performs batch GL1 multiplication between two tensors representing GL1 elements.
    The tensors have shape (batch_size, n+p, n+q) and be stored in block form:
      - Top-left block: P (which is offset, i.e. the actual block is P+I)
      - Top-right block: B
      - Bottom-left block: R
      - Bottom-right block: N
    The multiplication follows:
      new_P = (P_left+I) @ (P_right+I) - I
      new_B = (P_left+I) @ B_right + B_left
      new_R = R_left @ (P_right+I) + R_right
      new_N = R_left @ B_right + N_left + N_right
    """
    I = torch.eye(n, dtype=mat_left.dtype, device=mat_left.device)
    
    # cal batch dimension.
    batch_dims = mat_left.shape[:-2]
    # Reshape and expand I to match batch dimensions
    I_batch = I.view((1,)*len(batch_dims) + (n, n)).expand(*batch_dims, n, n)
    
    # Extract blocks from the right factor (the second argument)
    P_right = mat_right[..., :n, :n] + I_batch
    B_right = mat_right[..., :n, n:]
    R_right = mat_right[..., n:, :n]
    N_right = mat_right[..., n:, n:]
    
    # Extract blocks from the left factor (the first argument)
    P_left = mat_left[..., :n, :n] + I_batch
    B_left = mat_left[..., :n, n:]
    R_left = mat_left[..., n:, :n]
    N_left = mat_left[..., n:, n:]
    
    new_P = P_left @ P_right - I_batch
    new_B = P_left @ B_right + B_left
    new_R = R_left @ P_right + R_right
    new_N = R_left @ B_right + N_left + N_right
    
    top = torch.cat((new_P, new_B), dim=-1)
    bottom = torch.cat((new_R, new_N), dim=-1)
    return torch.cat((top, bottom), dim=-2)

def horizontal_compose_with(elem1, elem2, n):
    """
    Functional version of horizontal composition.
    The tuple structure is assumed as:
      (up1, up2, down1, down2, left1, left2, right1, right2, face_tensor)
    where face_tensor represents the GL1 element stored as a tensor.
    """
    
    # Check that the right of the left element matches the left of the right element.
    # print("A = ", elem1[6].size(), "B = ", elem2[4].size())
    # assert torch.allclose(elem1[6], elem2[4], atol=1e-5)
    # assert torch.allclose(elem1[7], elem2[5], atol=1e-5)
    
    # Compute the action: apply the down component of elem1 on elem2's value.
    acted = elem1[2] @ elem2[8] @ torch.linalg.inv(elem1[3])
    
    # Now multiply (in the GL1 sense) the result with the face tensor from elem1.
    new_value = gl1_mul_tensor(acted, elem1[8], n)
    
    # Compose the remaining blocks in the obvious way.
    new_up_1 = elem1[0] @ elem2[0]
    new_up_2 = elem1[1] @ elem2[1]
    new_down_1 = elem1[2] @ elem2[2]
    new_down_2 = elem1[3] @ elem2[3]
    
    return (new_up_1, new_up_2, new_down_1, new_down_2,
            elem1[4], elem1[5], elem2[6], elem2[7],
            new_value)


def vertical_compose_with(elem1, elem2, n):
    """
    Functional version of vertical composition.
    The tuple structure is assumed as:
      (up1, up2, down1, down2, left1, left2, right1, right2, face_tensor)
    where face_tensor represents the GL1 element stored as a tensor.
    """
    # assert torch.allclose(elem1[0], elem2[2], atol=1e-5)
    # assert torch.allclose(elem1[1], elem2[3], atol=1e-5)

    acted = elem1[4] @ elem2[8] @ torch.linalg.inv(elem1[5])
    new_value = gl1_mul_tensor(elem1[8], acted, n)

    new_left_1 = elem2[4] @ elem1[4]
    new_left_2 = elem2[5] @ elem1[5]

    new_right_1 = elem2[6] @ elem1[6]
    new_right_2 = elem2[7] @ elem1[7]

    tuple = (elem2[0], elem2[1], elem1[2], elem1[3], new_left_1, new_left_2, new_right_1, new_right_2, new_value)
    return tuple

def cal_aggregate(horizontal_compose_with, vertical_compose_with, elements, n):
    """
    Calculates final aggragate using associative scan
    Args:
        horizontal_compose_with and vertical_compose_with: Compose functions for two adjecent cells/faces
        elements: tuple of GL1 and GL0 elements of an image
        n: GL0 and GL1 parameter
    """
    combine_horizontal = partial(horizontal_compose_with, n=n)
    aggregate_horizontal = associative_scan(combine_horizontal, elements, dim=1, combine_mode='generic')
    # ( tensor(rows * columns * batch_size * n+p * n+p), tensor(rows * columns * ...), ..., tensor(rows * columns * batch_size * n+p * n+q) )

    flipped = tuple(t.flip(0) for t in aggregate_horizontal)
    combine_vertical = partial(vertical_compose_with, n=n)
    flipped_scanned = associative_scan(combine_vertical, flipped, dim=0, combine_mode='generic')
    aggregate = tuple(t.flip(0) for t in flipped_scanned)

    return aggregate

# Dims of face_tensor = rows * columns * batch_size * n+p * n+q
# ([[1, 2, 3],
# [2, 3, 4]], ....)

def scan_aggregate(n, p, q, images, torch_compile: bool = True):
    """
    Easy to import just this function in other files and get the complete aggregate.
    """
    elems = to_tuple(n, p, q, images, from_vector, kernel_gl1)
    if torch_compile:
        compiled_function = torch.compile(cal_aggregate)
    else:
        compiled_function = cal_aggregate

    aggregate = compiled_function(horizontal_compose_with, vertical_compose_with, elems, n)

    return aggregate

def scan_aggregate_benchmark(n, p, q, images, runs, torch_compile: bool = True):
    """
    To benchmark associative scan method
    """

    elems = to_tuple(n, p, q, images, from_vector, kernel_gl1)
    print("in progress...")
    if torch_compile:
        compiled_function = torch.compile(cal_aggregate)
    else:
        compiled_function = cal_aggregate

    Time = []
    for i in range(runs):
        start_time = time.time()
        aggregate = compiled_function(horizontal_compose_with, vertical_compose_with, elems, n)
        end_time = time.time()
        Time.append(end_time - start_time)

    final_time = Time[-1]
   
    print("Using associative scan in torch - ", f"Average time: {sum(Time)/runs},", f"Final time: {final_time},", f"Torch compile = {torch_compile}")

if __name__ == "__main__":
    n = 3
    p = 2
    q = 1
    batch_size = 2
    torch.manual_seed(42)
    images = torch.rand(batch_size, 5, 5).to(device)
    print(images)
    elements = to_tuple(n, p, q, images, from_vector, kernel_gl1)
    # print('elements=', elements, len(elements), [elements[i].shape for i in range(len(elements))])
    # xxx

    # associativity check
    elem1 = tuple(x[0][0] for x in elements)
    elem2 = tuple(x[0][1] for x in elements)
    elem3 = tuple(x[0][2] for x in elements)

    assert torch.allclose((horizontal_compose_with(horizontal_compose_with(elem1, elem2, n), elem3, n))[-1], (horizontal_compose_with(elem1, horizontal_compose_with(elem2, elem3, n), n))[-1], atol = 0.00001)

    aggregate = cal_aggregate(horizontal_compose_with, vertical_compose_with, elements, n)

    print("aggregate = ", aggregate[-1][0][-1])

# Game plan
# Data into dictionary 
# Define operation
# Associate Scan

# Problem
# 1. Parallization for lifting procedure - batch size and entire image or just batch size.
# 2. Vertical comp down to up