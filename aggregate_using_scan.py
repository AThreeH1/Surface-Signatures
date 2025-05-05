from imports import *
from torch._higher_order_ops.associative_scan import associative_scan
# from lifting import from_vector, kernel_gl1
import lifting
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
    F_V, F_U = tuple
    # Extract top and bottom rows for feedback
    rows, columns, m, _, _ = F_V.size()
    P = F_V[:, :, :, :n, :n].to(device) - torch.eye(n).repeat(rows, columns, m, 1, 1).to(device)
    B = F_U[:, :, :, :n, n:].to(device)
    R = F_V[:, :, :, n:, :n].to(device)

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

        returns [upU, upV, downV, downU, leftV, leftU, rightV, rightU, face_tensor]
            upU, downU, leftU, rightU         ~ (row-1) x (col-1) x bs x (n+p) x (n+p)
            upV, downV, leftV, rightV          ~ (row-1) x (col-1) x bs x (n+q) x (n+q)
            face_tensor ~ row x col x bs x (n+p) x (n+q)
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
    downV, downU = stack_grid(down) # TODO change for all
    leftV, leftU = stack_grid(left)
    rightV, rightU = stack_grid(right)

    # Stack N into a tensor preserving (rows, columns) structure
    N_tensor = torch.stack([torch.stack(Ni) for Ni in N])

    # Compute Edges_mul (order matches to_custom_matrix)
    Edges_mul = (
        downV @ rightV @ torch.linalg.inv(upV) @ torch.linalg.inv(leftV),
        downU @ rightU @ torch.linalg.inv(upU) @ torch.linalg.inv(leftU)
    )

    face_tensor = reverse_feedback(n, Edges_mul, N_tensor)

    # Return flat tuple of tensors
    final_tuple = (upV, upU, downV, downU, leftV, leftU, rightV, rightU, face_tensor)
    return final_tuple

def to_tuple_vectorized(n, p, q, image, from_vector, kernel_gl1):
    """
    Vectorized lifting procedure.
    Lifting Procedure - Maps elements from a batched image to the dictionary structure.
    Args:
        n, p, q: parameters of GL0 and GL1
        image (torch.Tensor): Batched tensor of shape (batch_size, row, col)
        from_vector and kernel_gl1: Lifting functions
    Returns:
        pytree structure containing all data, specifically in tensor form.

        returns [upU, upV, downV, downU, leftV, leftU, rightV, rightU, face_tensor]
            upU, downU, leftU, rightU         ~ (row-1) x (col-1) x bs x (n+p) x (n+p)
            upV, downV, leftV, rightV          ~ (row-1) x (col-1) x bs x (n+q) x (n+q)
            face_tensor ~ row x col x bs x (n+p) x (n+q)
    
    """
    Batch_size, rows, cols = image.shape
    height, width = rows - 1, cols - 1  # grid dimensions

    # --- Helper to batch calls: flatten extra dims into batch dimension
    def batch_call(func, *args):
        # Each argument is assumed to have shape (Batch_size, extra_dims, 1) or (Batch_size, extra_dims)
        # Flatten all dimensions after the batch dimension into one.
        extra_shape = args[0].shape[1:]
        flat_args = [a.reshape(Batch_size * int(torch.tensor(extra_shape).prod()), -1).squeeze(-1) for a in args]
        m = flat_args[0].shape[0]
        flat_out = func(n, p, q, m, *flat_args)
        # flat_out is a tuple (fV, fU) of shape (m, n+p, n+p)
        fV = flat_out[0].reshape(Batch_size, *extra_shape, n+p, n+p)
        fU = flat_out[1].reshape(Batch_size, *extra_shape, n+p, n+p)
        return fV, fU

    # --- Compute the direct edge blocks via from_vector
    # For "up": for each cell in the top row, use image[:, 0, j] and image[:, 0, j+1]
    up_direct_V, up_direct_U = batch_call(
        from_vector,
        image[:, 0, :width],   # shape: (Batch_size, width)
        image[:, 0, 1:]    # shape: (Batch_size, width)
    )
    
    # For "down": for each cell (i,j) with i in 0..height-1, j in 0..width-1:
    #   use image[:, i+1, j] and image[:, i+1, j+1]
    down_V, down_U = batch_call(
        from_vector,
        image[:, 1:, :width].reshape(Batch_size, height * width),   # shape: (Batch_size, height*width)
        image[:, 1:, 1:].reshape(Batch_size, height * width)      # shape: (Batch_size, height*width)
    )
    down_V = down_V.reshape(Batch_size, height, width, n+p, n+p)
    down_U = down_U.reshape(Batch_size, height, width, n+p, n+p)
    
    # For "right": for each cell (i,j):
    #   use image[:, i+1, j+1] and image[:, i, j+1]
    right_V, right_U = batch_call(
        from_vector,
        image[:, 1:, 1:].reshape(Batch_size, height * width),   # shape: (Batch_size, height*width)
        image[:, :height, 1:].reshape(Batch_size, height * width)      # shape: (Batch_size, height*width)
    )
    right_V = right_V.reshape(Batch_size, height, width, n+p, n+p)
    right_U = right_U.reshape(Batch_size, height, width, n+p, n+p)
    
    # For "left": for each cell (i,j) for column 0, use image[:, i+1, 0] and image[:, i, 0]
    left_direct_V, left_direct_U = batch_call(
        from_vector,
        image[:, 1:, 0].reshape(Batch_size, height),   # shape: (Batch_size, height)
        image[:, :height, 0].reshape(Batch_size, height)      # shape: (Batch_size, height)
    )
    left_direct_V = left_direct_V.reshape(Batch_size, height, n+p, n+p)
    left_direct_U = left_direct_U.reshape(Batch_size, height, n+p, n+p)
    
    # --- Compute kernel matrix N for each cell (i,j)
    # p1 = image[:, i, j], p2 = image[:, i+1, j], p3 = image[:, i+1, j+1], p4 = image[:, i, j+1]
    def batch_kernel(p1, p2, p3, p4):
        extra_shape = p1.shape[1:]
        flat_p1 = p1.reshape(Batch_size * int(torch.tensor(extra_shape).prod()))
        flat_p2 = p2.reshape(Batch_size * int(torch.tensor(extra_shape).prod()))
        flat_p3 = p3.reshape(Batch_size * int(torch.tensor(extra_shape).prod()))
        flat_p4 = p4.reshape(Batch_size * int(torch.tensor(extra_shape).prod()))
        flat_out = kernel_gl1(flat_p1, flat_p2, flat_p3, flat_p4, p, q)
        return flat_out.reshape(Batch_size, *extra_shape, p, q)
    
    N_tensor = batch_kernel(
        image[:, :height, :width],
        image[:, 1:, :width],
        image[:, 1:, 1:],
        image[:, :height, 1:]
    )
    # N_tensor has shape (Batch_size, height, width, p, q)

    # --- Assemble the full grid of edges.
    # up: for grid cells, for row 0 use up_direct; for i>=1 use down from previous row.
    up_V = torch.empty(Batch_size, height, width, n+p, n+p, device=image.device)
    up_U = torch.empty(Batch_size, height, width, n+p, n+p, device=image.device)
    up_V[:, 0, :, :, :] = up_direct_V
    up_U[:, 0, :, :, :] = up_direct_U
    if height > 1:
        up_V[:, 1:, :, :, :] = down_V[:, :height-1, :, :, :]
        up_U[:, 1:, :, :, :] = down_U[:, :height-1, :, :, :]

    # left: for grid cells, for col 0 use left_direct; for j>=1 use right from previous column.
    left_V = torch.empty(Batch_size, height, width, n+p, n+p, device=image.device)
    left_U = torch.empty(Batch_size, height, width, n+p, n+p, device=image.device)
    left_V[:, :, 0, :, :] = left_direct_V
    left_U[:, :, 0, :, :] = left_direct_U
    if width > 1:
        left_V[:, :, 1:, :, :] = right_V[:, :, :width-1, :, :]
        left_U[:, :, 1:, :, :] = right_U[:, :, :width-1, :, :]

    # --- Compute the edge multiplication (batched matrix multiplications)
    inv_up_V = torch.linalg.inv(up_V)
    inv_left_V = torch.linalg.inv(left_V)
    edges_mul_V = torch.matmul(torch.matmul(torch.matmul(down_V, right_V), inv_up_V), inv_left_V)
    
    inv_up_U = torch.linalg.inv(up_U)
    inv_left_U = torch.linalg.inv(left_U)
    edges_mul_U = torch.matmul(torch.matmul(torch.matmul(down_U, right_U), inv_up_U), inv_left_U)
    Edges_mul = (edges_mul_V, edges_mul_U)
    
    # --- Compute the face tensor via feedback (using the provided reverse_feedback)
    face_tensor = reverse_feedback(n, Edges_mul, N_tensor)
    
    return (up_V.permute(1, 2, 0, 3, 4).contiguous(), up_U.permute(1, 2, 0, 3, 4).contiguous(), down_V.permute(1, 2, 0, 3, 4).contiguous(), down_U.permute(1, 2, 0, 3, 4).contiguous(), left_V.permute(1, 2, 0, 3, 4).contiguous(), left_U.permute(1, 2, 0, 3, 4).contiguous(), right_V.permute(1, 2, 0, 3, 4).contiguous(), right_U.permute(1, 2, 0, 3, 4).contiguous(), face_tensor.permute(1, 2, 0, 3, 4).contiguous())

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
      (up1, up2, downV, downU, leftV, leftU, rightV, rightU, face_tensor)
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
    new_upV = elem1[0] @ elem2[0]
    new_upU = elem1[1] @ elem2[1]
    new_downV = elem1[2] @ elem2[2]
    new_downU = elem1[3] @ elem2[3]

    return (new_upV, new_upU, new_downV, new_downU,
            elem1[4], elem1[5], elem2[6], elem2[7],
            new_value)

def vertical_compose_with(elem1, elem2, n):
    """
    Functional version of vertical composition.
    The tuple structure is assumed as:
      (up1, up2, downV, downU, leftV, leftU, rightV, rightU, face_tensor)
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
    elems = to_tuple_vectorized(n, p, q, images, lifting.from_vector, lifting.kernel_gl1)
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

    elems = to_tuple_vectorized(n, p, q, images, lifting.from_vector, lifting.kernel_gl1)
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

import torch

def loop_aggregate(n, p, q, images):
    """
    Calculates final aggregate using loop-based scan.
    Args:
        horizontal_compose_with, vertical_compose_with: 
            Composition functions for adjacent cells/faces.
        elements: tuple of 9 tensors of shape (rows, cols, ...),
            where each tensor corresponds to one of the GL0/GL1 blocks.
        n: parameter for GL0 and GL1 operations.
    Returns:
        A tuple of 9 tensors of the same shape as the input (rows, cols, ...),
        representing the aggregated result.
    """
    elements = to_tuple(n, p, q, images, lifting.from_vector, lifting.kernel_gl1)
    # Get the number of rows and columns from the first tensor.
    rows, cols = elements[0].shape[:2]
    
    # Perform horizontal scan: for each row, compose left-to-right.
    # We create a 2D list "horiz_aggr" where horiz_aggr[r][c] is the horizontal
    # aggregate from (r,0) to (r,c).
    horiz_aggr = []
    for r in range(rows):
        # Initialize with the first cell in the row.
        cell = tuple(elements[i][r, 0] for i in range(9))
        row_aggr = [cell]
        for c in range(1, cols):
            current_cell = tuple(elements[i][r, c] for i in range(9))
            # Compose the last aggregate with the current cell horizontally.
            new_cell = horizontal_compose_with(row_aggr[-1], current_cell, n)
            row_aggr.append(new_cell)
        horiz_aggr.append(row_aggr)
    
    # Now perform vertical scan.
    # First flip the rows (so we can scan from bottom to top).
    flipped = horiz_aggr[::-1]
    
    # "vert_aggr" will be a 2D list with the same dimensions.
    vert_aggr = []
    for r in range(len(flipped)):
        new_row = []
        for c in range(cols):
            if r == 0:
                # The bottom row remains as is.
                new_cell = flipped[r][c]
            else:
                # Compose vertically: note that vertical_compose_with takes the cell above
                # (already aggregated) and the current flipped cell.
                new_cell = vertical_compose_with(vert_aggr[r-1][c], flipped[r][c], n)
            new_row.append(new_cell)
        vert_aggr.append(new_row)
    
    # Flip back the vertical aggregated result to recover original row order.
    final_aggr = vert_aggr[::-1]
    
    # Reassemble the final aggregation into a tuple of 9 tensors.
    # For each of the 9 components, we stack the corresponding cell entries.
    result_components = []
    for comp in range(9):
        # For each row, collect the comp-th entry from each cell and stack them horizontally.
        row_tensors = []
        for r in range(rows):
            # For row r, stack along the column axis.
            col_cells = [final_aggr[r][c][comp] for c in range(cols)]
            row_tensor = torch.stack(col_cells, dim=0)  # shape: (cols, ...)
            row_tensors.append(row_tensor)
        # Now stack all the rows to get a tensor of shape (rows, cols, ...)
        comp_tensor = torch.stack(row_tensors, dim=0)
        result_components.append(comp_tensor)
    
    return tuple(result_components)


if __name__ == "__main__":
    n = 2 # TODO test with different values!
    p = 1
    q = 1
    batch_size = 2
    torch.manual_seed(42)
    images = torch.rand(batch_size, 25, 25).to(device)
    A = time.time()
    elements = to_tuple(n, p, q, images, lifting.from_vector, lifting.kernel_gl1)
    B = time.time()
    elements_2 = to_tuple_vectorized(n, p, q, images, lifting.from_vector, lifting.kernel_gl1)
    C = time.time()
    print("For looped to tuple = ", B-A, "For parallel to tuple = ", C-B)
    # print('elements=', elements, len(elements), [elements[i].shape for i in range(len(elements))])
    print(elements[-1][0][-1],"spaceeeeeeeeeeeeeeeeeeeeeeeeeeee", elements_2[-1][0][-1])
    

    # associativity check
    elem1 = tuple(x[0][0] for x in elements)
    elem2 = tuple(x[0][1] for x in elements)
    elem3 = tuple(x[0][2] for x in elements)

    assert torch.allclose((horizontal_compose_with(horizontal_compose_with(elem1, elem2, n), elem3, n))[-1], (horizontal_compose_with(elem1, horizontal_compose_with(elem2, elem3, n), n))[-1], atol = 0.00001)

    aggregate = cal_aggregate(horizontal_compose_with, vertical_compose_with, elements, n)

    # print("aggregate = ", aggregate[-1][0][-1], "aggregate_loop = ", aggregate_loop[-1][0][-1])

# Game plan
# Data into dictionary 
# Define operation
# Associate Scan

# Problem
# 1. Parallization for lifting procedure - batch size and entire image or just batch size.
# 2. Vertical comp down to up