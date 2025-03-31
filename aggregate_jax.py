import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import torch
import time

jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnames=('p', 'q'))
def kernel_gl1(p1, p2, p3, p4, p, q):
    Y = p1 + p3 - p2 - p4  # shape: (batch, ...)
    Y = Y.reshape((-1, 1, 1))
    row_mult = jnp.arange(1, p+1).reshape(1, p, 1)
    col_exp = jnp.arange(1, q+1).reshape(1, 1, q)
    N = row_mult * (Y ** col_exp)
    return N

@partial(jax.jit, static_argnames=('n',))
def reverse_feedback(n, tuple_edges, N):
    M_V, M_U = tuple_edges
    P = M_V[..., :n, :n] - jnp.eye(n)
    B = M_U[..., :n, n:]
    
    R = M_V[..., n:, :n]
    top_row = jnp.concatenate([P, B], axis=-1)
    bottom_row = jnp.concatenate([R, N], axis=-1)
    result_matrix = jnp.concatenate([top_row, bottom_row], axis=-2)
    return result_matrix

@partial(jax.jit, static_argnames=('n', 'p', 'q'))
def to_tuple(n, p, q, image):
    batch_size, rows, cols = image.shape
    R_dim = rows - 1
    C_dim = cols - 1
    
    # Initialize empty lists for edges
    up = [[None for _ in range(C_dim)] for _ in range(R_dim)]
    down = [[None for _ in range(C_dim)] for _ in range(R_dim)]
    left = [[None for _ in range(C_dim)] for _ in range(R_dim)]
    right = [[None for _ in range(C_dim)] for _ in range(R_dim)]
    N = [[None for _ in range(C_dim)] for _ in range(R_dim)]
    
    # Precompute all edges
    for i in range(R_dim):
        for j in range(C_dim):
            # Compute up edge
            if i == 0:
                up[i][j] = from_vector(n, p, q, image[:, i, j], image[:, i, j+1])
            else:
                up[i][j] = down[i-1][j]
            
            # Compute left edge
            if j == 0:
                left[i][j] = from_vector(n, p, q, image[:, i+1, j], image[:, i, j])
            else:
                left[i][j] = right[i][j-1]
            
            # Compute down edge
            down[i][j] = from_vector(n, p, q, image[:, i+1, j], image[:, i+1, j+1])
            
            # Compute right edge
            right[i][j] = from_vector(n, p, q, image[:, i+1, j+1], image[:, i, j+1])
            
            # Compute N
            p1 = image[:, i, j]
            p2 = image[:, i+1, j]
            p3 = image[:, i+1, j+1]
            p4 = image[:, i, j+1]
            N[i][j] = kernel_gl1(p1, p2, p3, p4, p, q)
    
    # Stack grids into tensors for the “edge” components
    def stack(grid):
        fV_stack = []
        fU_stack = []
        for row in grid:
            fV_row = []
            fU_row = []
            for elem in row:
                fV, fU = elem
                fV_row.append(fV)
                fU_row.append(fU)
            fV_stack.append(jnp.stack(fV_row))
            fU_stack.append(jnp.stack(fU_row))
        return jnp.stack(fV_stack), jnp.stack(fU_stack)
    
    up_v, up_u = stack(up)
    down_v, down_u = stack(down)
    left_v, left_u = stack(left)
    right_v, right_u = stack(right)
    N_tensor = jnp.stack([jnp.stack(Ni) for Ni in N])
    
    # Compute Edges_mul (we work on the V and U parts separately)
    Edges_mul = (
        jnp.einsum('...ab,...bc->...ac', down_v, right_v) @ 
        jnp.linalg.inv(up_v) @ jnp.linalg.inv(left_v),
        jnp.einsum('...ab,...bc->...ac', down_u, right_u) @ 
        jnp.linalg.inv(up_u) @ jnp.linalg.inv(left_u)
    )
    face_tensor = reverse_feedback(n, Edges_mul, N_tensor)
    
    return (up_v, up_u, down_v, down_u, left_v, left_u, right_v, right_u, face_tensor)

@partial(jax.jit, static_argnames=('n', 'p', 'q'))
def to_tuple_vectorized(n, p, q, image):
    # image: (B, rows, cols)
    B, rows, cols = image.shape
    h, w = rows - 1, cols - 1  # grid dimensions

    # --- Nested vmap helper for from_vector calls
    # from_vector expects scalar (or (B,) vector) inputs; we use a double vmap for grid cells.
    fv_func = lambda xt, xs: from_vector(n, p, q, xt, xs)
    double_vmap = jax.vmap(jax.vmap(fv_func, in_axes=(0, 0)), in_axes=(0, 0))
    
    # --- Down edges: for each grid cell (i, j), use image[:, i+1, j] and image[:, i+1, j+1]
    down_v, down_u = double_vmap(image[:, 1:, :w], image[:, 1:, 1:])  # shape: (B, h, w, n+p, n+p)
    
    # --- Right edges: for each grid cell (i, j), use image[:, i+1, j+1] and image[:, i, j+1]
    right_v, right_u = double_vmap(image[:, 1:, 1:], image[:, :h, 1:])  # shape: (B, h, w, n+p, n+p)
    
    # --- Up edges:
    # For the top row (grid row 0), compute directly.
    up_direct_v, up_direct_u = jax.vmap(fv_func, in_axes=(0, 0))(
        image[:, 0, :w],   # shape: (B, w)
        image[:, 0, 1:]    # shape: (B, w)
    )
    # up_direct_*: shape (B, w, n+p, n+p). Expand to (B, 1, w, n+p, n+p)
    up_direct_v = up_direct_v[:, None, :, :, :]
    up_direct_u = up_direct_u[:, None, :, :, :]
    # For subsequent rows, use the down edge from the previous grid row.
    up_rest_v = down_v[:, :h-1, :, :, :]
    up_rest_u = down_u[:, :h-1, :, :, :]
    # Concatenate along the grid row dimension (axis=1)
    up_v = jnp.concatenate([up_direct_v, up_rest_v], axis=1)  # shape: (B, h, w, n+p, n+p)
    up_u = jnp.concatenate([up_direct_u, up_rest_u], axis=1)
    
    # --- Left edges:
    # For the left column, compute directly.
    left_direct_v, left_direct_u = jax.vmap(fv_func, in_axes=(0, 0))(
        image[:, 1:, 0],   # shape: (B, h)
        image[:, :h, 0]    # shape: (B, h)
    )
    # left_direct_*: shape (B, h, n+p, n+p). Expand to (B, h, 1, n+p, n+p)
    left_direct_v = left_direct_v[:, :, None, :, :]
    left_direct_u = left_direct_u[:, :, None, :, :]
    # For subsequent columns, left edge equals right edge from previous column.
    left_rest_v = right_v[:, :, :w-1, :, :]
    left_rest_u = right_u[:, :, :w-1, :, :]
    # Concatenate along the grid column dimension (axis=2)
    left_v = jnp.concatenate([left_direct_v, left_rest_v], axis=2)  # shape: (B, h, w, n+p, n+p)
    left_u = jnp.concatenate([left_direct_u, left_rest_u], axis=2)
    
    # --- Kernel tensor N:
    # p1 = image[:, :h, :w], p2 = image[:, 1:, :w], p3 = image[:, 1:, 1:], p4 = image[:, :h, 1:]
    double_kernel = jax.vmap(jax.vmap(
        lambda a, b, c, d: kernel_gl1(a, b, c, d, p, q),
        in_axes=(0, 0, 0, 0)
    ), in_axes=(0, 0, 0, 0))
    N_tensor = double_kernel(image[:, :h, :w],
                             image[:, 1:, :w],
                             image[:, 1:, 1:],
                             image[:, :h, 1:])  # shape: (B, h, w, p, q)
    
    # --- Batched edge multiplication:
    inv_up_v = jnp.linalg.inv(up_v)
    inv_left_v = jnp.linalg.inv(left_v)
    Edges_mul_v = jnp.matmul(jnp.matmul(jnp.matmul(down_v, right_v), inv_up_v), inv_left_v)
    
    inv_up_u = jnp.linalg.inv(up_u)
    inv_left_u = jnp.linalg.inv(left_u)
    Edges_mul_u = jnp.matmul(jnp.matmul(jnp.matmul(down_u, right_u), inv_up_u), inv_left_u)
    Edges_mul = (Edges_mul_v, Edges_mul_u)
    
    # --- Compute face tensor via reverse_feedback.
    face_tensor = reverse_feedback(n, Edges_mul, N_tensor)  # expected shape: (B, h, w, n+p, n+q)
    
    # --- Rearranging dimensions: from (B, h, w, ...) to (h, w, B, ...)
    def batch_to_grid(t):
        # If t has shape (B, h, w, a, b) then transpose to (h, w, B, a, b)
        return jnp.transpose(t, (1, 2, 0, 3, 4))
    
    def batch_to_grid_N(t):
        # For N_tensor: (B, h, w, p, q) -> (h, w, B, p, q)
        return jnp.transpose(t, (1, 2, 0, 3, 4))
    
    up_v = batch_to_grid(up_v)
    up_u = batch_to_grid(up_u)
    down_v = batch_to_grid(down_v)
    down_u = batch_to_grid(down_u)
    left_v = batch_to_grid(left_v)
    left_u = batch_to_grid(left_u)
    right_v = batch_to_grid(right_v)
    right_u = batch_to_grid(right_u)
    face_tensor = batch_to_grid(face_tensor)
    N_tensor = batch_to_grid_N(N_tensor)
    
    # Return the tuple with grid dims first: (h, w, B, ...)
    return (up_v, up_u, down_v, down_u, left_v, left_u, right_v, right_u, face_tensor)

@partial(jax.jit, static_argnames=('n',))
def gl1_mul_tensor(mat_left, mat_right, n):
    I = jnp.eye(n)
    batch_shape = mat_left.shape[:-2]
    I_batch = jnp.broadcast_to(I, batch_shape + I.shape)
    
    P_left = mat_left[..., :n, :n] + I_batch
    B_left = mat_left[..., :n, n:]
    R_left = mat_left[..., n:, :n]
    N_left = mat_left[..., n:, n:]
    
    P_right = mat_right[..., :n, :n] + I_batch
    B_right = mat_right[..., :n, n:]
    R_right = mat_right[..., n:, :n]
    N_right = mat_right[..., n:, n:]
    
    new_P = P_left @ P_right - I_batch
    new_B = P_left @ B_right + B_left
    new_R = R_left @ P_right + R_right
    new_N = R_left @ B_right + N_left + N_right
    
    top = jnp.concatenate([new_P, new_B], axis=-1)
    bottom = jnp.concatenate([new_R, new_N], axis=-1)
    return jnp.concatenate([top, bottom], axis=-2)

@partial(jax.jit, static_argnames=('n',))
def horizontal_compose(elem1, elem2, n):
    up1_v, up1_u, down1_v, down1_u, left1_v, left1_u, right1_v, right1_u, val1 = elem1
    up2_v, up2_u, down2_v, down2_u, left2_v, left2_u, right2_v, right2_u, val2 = elem2
    
    # Compute the action using down edges from elem1 on elem2's face tensor
    acted = down1_v @ val2 @ jnp.linalg.inv(down1_u)
    new_face = gl1_mul_tensor(acted, val1, n)
    
    new_up_v = up1_v @ up2_v
    new_up_u = up1_u @ up2_u
    new_down_v = down1_v @ down2_v
    new_down_u = down1_u @ down2_u
    
    return (new_up_v, new_up_u, new_down_v, new_down_u,
            left1_v, left1_u, right2_v, right2_u, new_face)

@partial(jax.jit, static_argnames=('n',))
def vertical_compose(elem1, elem2, n):
    up1_v, up1_u, down1_v, down1_u, left1_v, left1_u, right1_v, right1_u, val1 = elem1
    up2_v, up2_u, down2_v, down2_u, left2_v, left2_u, right2_v, right2_u, val2 = elem2
    
    # Compute the action using left edges from elem1 on elem2's face tensor
    acted = left1_v @ val2 @ jnp.linalg.inv(left1_u)
    new_face = gl1_mul_tensor(val1, acted, n)
    
    new_left_v = left2_v @ left1_v
    new_left_u = left2_u @ left1_u
    new_right_v = right2_v @ right1_v
    new_right_u = right2_u @ right1_u
    
    return (up2_v, up2_u, down1_v, down1_u,
            new_left_v, new_left_u, new_right_v, new_right_u, new_face)

def cal_aggregate(elements, n):


    # Define the horizontal composition function for the associative scan.
    def horizontal_associative_compose(cell1, cell2):
        return horizontal_compose(cell1, cell2, n)

    # Horizontal associative scan along columns (axis=1)
    horizontal_scanned = jax.lax.associative_scan(horizontal_associative_compose, elements, axis=1)
    # Shape (rows, cols, batch, ...)

    # Flip the grid along axis 0 (vertical axis) before vertical scan
    flipped = jax.tree_map(lambda x: jnp.flip(x, axis=0), horizontal_scanned)

    # Define the vertical composition function for the associative scan.
    def vertical_associative_compose(cell1, cell2):
        return vertical_compose(cell1, cell2, n)

    # Perform vertical associative scan along axis=0 on the flipped grid.
    flipped_scanned = jax.lax.associative_scan(vertical_associative_compose, flipped, axis=0)

    # Flip back along axis 0 to restore the original order.
    aggregated = jax.tree_map(lambda x: jnp.flip(x, axis=0), flipped_scanned)

    return aggregated

def jax_scan_aggregate(n, p, q, images, jax_jit: bool = True):

    # Each component has shape (rows, cols, batch, ...)
    up_v, up_u, down_v, down_u, left_v, left_u, right_v, right_u, face_tensor = to_tuple(n, p, q, images)
    elements = (up_v, up_u, down_v, down_u, left_v, left_u, right_v, right_u, face_tensor)

    if jax_jit:
        compiled_function = jax.jit(cal_aggregate, static_argnames=('n',))
    else:
        compiled_function = cal_aggregate

    aggregate = compiled_function(elements, n)

    return aggregate 

def jax_scan_aggregate_benchmark(n, p, q, images, runs, jax_jit: bool = True):

    up_v, up_u, down_v, down_u, left_v, left_u, right_v, right_u, face_tensor = to_tuple(n, p, q, images)
    elements = (up_v, up_u, down_v, down_u, left_v, left_u, right_v, right_u, face_tensor)
    # Shape (rows, cols, batch, ...)

    print("in progress...")
    if jax_jit:
        compiled_function = jax.jit(cal_aggregate, static_argnames=('n',))
    else:
        compiled_function = cal_aggregate

    Time = []
    for i in range(runs):
        start_time = time.time()
        aggregate = compiled_function(elements, n)
        end_time = time.time()
        Time.append(end_time - start_time)

    final_time = Time[-1]
   
    print("Using associative scan in JAX - ", f"Average time: {sum(Time)/runs},", f"Final time: {final_time},", f"jax_jit = {jax_jit}")


if __name__ == "__main__":
    batch_size = 2
    torch.manual_seed(42)
    images_torch = torch.rand(batch_size, 2000, 2000)
    image = jnp.asarray(images_torch.numpy()) 
    n, p, q = 2, 1, 1
    A = time.time()
    # to_tuple_loop = to_tuple(n, p, q, image)
    B = time.time()
    to_tuple_vector = to_tuple_vectorized(n, p, q, image)
    C = time.time()
    torch.manual_seed(41)
    images_torch = torch.rand(batch_size, 2000, 2000)
    image = jnp.asarray(images_torch.numpy())
    D = time.time()
    to_tuple_vector = to_tuple_vectorized(n, p, q, image)
    E = time.time()
    print("For looped to tuple = ", B-A, "For parallel to tuple = ", C-B, "for par = ", E - D)
    # print((to_tuple_loop[-1][0][-1]))
    # print((to_tuple_vector[-1][0][-1]))f 
    # aggregate = jax_scan_aggregate(n, p, q, image)
    # print(aggregate[-1][0][-1])