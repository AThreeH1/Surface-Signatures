import jax
import jax.numpy as jnp
from functools import partial
import time
from jax import profiler
import logging

# Configure JAX
jax.config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial

class FFNs(nn.Module):
    """Neural network modules for P, R, B matrices and kernel N."""
    n: int
    p: int
    q: int
    hidden_dim: int = 64  # Default hidden dimension
    
    @nn.compact
    def __call__(self, Xt, Xs):
        """
        Compute matrices fV and fU using neural networks.
        Args:
            Xt, Xs: Input points, shape (batch, ...)
        Returns:
            tuple: (fV, fU) matrices
        """
        batch_size = Xt.shape[0]
        
        # Prepare input: concatenate Xt and Xs
        inputs = jnp.concatenate([Xt.reshape(batch_size, -1), 
                                 Xs.reshape(batch_size, -1)], axis=1)
        
        # FFN for P diagonal values
        p_values = nn.Dense(self.hidden_dim)(inputs)
        p_values = nn.relu(p_values)
        p_values = nn.Dense(self.n)(p_values)
        p_values = jnp.exp(p_values)  # Ensure positive values
        
        # FFN for R matrix
        r_values = nn.Dense(self.hidden_dim)(inputs)
        r_values = nn.relu(r_values)
        r_values = nn.Dense(self.p * self.n)(r_values)
        R = r_values.reshape(batch_size, self.p, self.n)
        
        # FFN for B matrix
        b_values = nn.Dense(self.hidden_dim)(inputs)
        b_values = nn.relu(b_values)
        b_values = nn.Dense(self.n * self.q)(b_values)
        B = b_values.reshape(batch_size, self.n, self.q)
        
        # Create P as diagonal matrix
        P = jnp.zeros((batch_size, self.n, self.n))
        P = P.at[:, jnp.arange(self.n), jnp.arange(self.n)].set(p_values)
        
        # Create identity matrices S and D
        S = jnp.eye(self.p)[jnp.newaxis, :, :].repeat(batch_size, axis=0)
        D = jnp.eye(self.q)[jnp.newaxis, :, :].repeat(batch_size, axis=0)
        
        # Assemble fV and fU matrices
        zeros_np = jnp.zeros((batch_size, self.n, self.p))
        zeros_qn = jnp.zeros((batch_size, self.q, self.n))
        
        # Assemble fV
        top_fV = jnp.concatenate([P, zeros_np], axis=2)
        bottom_fV = jnp.concatenate([R, S], axis=2)
        fV = jnp.concatenate([top_fV, bottom_fV], axis=1)
        
        # Assemble fU
        top_fU = jnp.concatenate([P, B], axis=2)
        bottom_fU = jnp.concatenate([zeros_qn, D], axis=2)
        fU = jnp.concatenate([top_fU, bottom_fU], axis=1)
        
        return fV, fU
    
    @nn.compact
    def compute_kernel(self, p1, p2, p3, p4):
        """
        Compute kernel N using a neural network.
        Args:
            p1, p2, p3, p4: Corner points of shape (batch, ...)
        Returns:
            N: Kernel matrix of shape (batch, p, q)
        """
        batch_size = p1.shape[0]
        
        # Concatenate all four points as input
        inputs = jnp.concatenate([
            p1.reshape(batch_size, -1),
            p2.reshape(batch_size, -1),
            p3.reshape(batch_size, -1),
            p4.reshape(batch_size, -1)
        ], axis=1)
        
        # FFN for N
        n_values = nn.Dense(self.hidden_dim)(inputs)
        n_values = nn.relu(n_values)
        n_values = nn.Dense(self.p * self.q)(n_values)
        
        # Reshape to (batch, p, q)
        N = n_values.reshape(batch_size, self.p, self.q)
        
        return N

@partial(jax.jit, static_argnames=('n', 'p', 'q'))
def init_ffn_params(n, p, q, seed=7):
    """
    Initialize parameters for both FFN methods.
    
    Args:
        n, p, q: Matrix dimensions
        seed: Random seed
        
    Returns:
        Dictionary of parameters
    """
    model = FFNs(n=n, p=p, q=q)
    kernel_model = FFNs(n=1, p=p, q=q)  # For kernel computation
    
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    
    # Create dummy inputs for initialization
    batch_size = 1
    dummy_Xt = jnp.zeros((batch_size, 1))
    dummy_Xs = jnp.zeros((batch_size, 1))
    dummy_p1 = jnp.zeros((batch_size, 1))
    dummy_p2 = jnp.zeros((batch_size, 1))
    dummy_p3 = jnp.zeros((batch_size, 1))
    dummy_p4 = jnp.zeros((batch_size, 1))
    
    # Initialize parameters for both methods
    edge_params = model.init(key1, dummy_Xt, dummy_Xs)
    kernel_params = kernel_model.init(key2, method=FFNs.compute_kernel, 
                                    p1=dummy_p1, p2=dummy_p2, p3=dummy_p3, p4=dummy_p4)
    
    # Combine parameters
    return {'edge': edge_params, 'kernel': kernel_params}

@partial(jax.jit, static_argnames=('n', 'p', 'q'))
def from_vector_ffn(n, p, q, Xt, Xs, params):
    """
    Constructs matrices fV and fU using neural networks.
    
    Args:
        n, p, q: Matrix dimensions
        Xt, Xs: Input points (batch, ...)
        params: Neural network parameters
        
    Returns:
        tuple: (fV, fU) matrices
    """
    model = FFNs(n=n, p=p, q=q)
    return model.apply(params['edge'], Xt, Xs)

@partial(jax.jit, static_argnames=('p', 'q'))
def kernel_gl1_ffn(p1, p2, p3, p4, p, q, params):
    """
    Compute kernel N using neural network.
    
    Args:
        p1, p2, p3, p4: Corner points
        p, q: Matrix dimensions
        params: Neural network parameters
        
    Returns:
        N: Kernel matrix
    """
    model = FFNs(n=1, p=p, q=q)  # n doesn't matter for kernel computation
    return model.apply(params['kernel'], method=FFNs.compute_kernel, p1=p1, p2=p2, p3=p3, p4=p4)

@partial(jax.jit, static_argnames=('n', 'p', 'q'))
def from_vector(n, p, q, Xt, Xs, params):
    """
    Constructs the matrices fV and fU based on input parameters.

    Args:
        n, p, q : parameters
        Xt (array): Reference input points, shape (batch, ...).
        Xs (array): Target input points, shape (batch, ...).
        params (dict): Dictionary containing parameters.

    Returns:
        tuple: (fV, fU)
            - fV (array): Constructed matrix of shape (batch, n+p, n+p).
            - fU (array): Constructed matrix of shape (batch, n+q, n+q).

    Matrix Construction:
        fV = [
            P   0
            R   S
        ]

        fU = [
            P   B
            0   D
        ]

        Where:
            - **P (n x n)**: Diagonal matrix with elements:
              P_ii = a_i * exp(dX^i)

            - **R (p x n)**: Alternating sine and cosine terms:
              R_ij = b_j * sin(dX^j) + b'_j * sin(dX^j) * dX  (even i)
              R_ij = c_j * cos(dX^j) + c'_j * cos(dX^j) * dX  (odd i)

            - **B (n x q)**: Polynomial dependency on dX:
              B_ij = d_j * i * j * dX^i + d'_j

            - **S (p x p)**: Identity matrix of size p.
            - **D (q x q)**: Identity matrix of size q.
    """

    a = params['a']  # shape (n,)
    b = params['b']
    b_prime = params['b_prime']
    c = params['c']
    c_prime = params['c_prime']
    d = params['d']
    d_prime = params['d_prime']

    dX = Xs - Xt  # shape: (batch, ...)
    dX = dX.reshape((-1, 1, 1))  # shape: (batch, 1, 1)
    
    # Build block P (n x n)
    powers = jnp.arange(1, n+1).reshape(1, n)  # shape (1, n)
    dX_flat = dX.reshape((-1, 1))  # shape (batch, 1)
    diag_vals = a * jnp.exp(dX_flat ** powers)  # shape (batch, n), with broadcasting over (n,)
    P = jnp.zeros((diag_vals.shape[0], n, n))
    P = P.at[:, jnp.arange(n), jnp.arange(n)].set(diag_vals)
    
    # Build block R (p x n)
    dX_power = dX_flat ** jnp.arange(1, n+1).reshape(1, n)  # shape (batch, n)
    dX_power = jnp.expand_dims(dX_power, 1)  # shape (batch, 1, n)
    dX_factor = dX_flat.reshape(-1, 1, 1)  # shape (batch, 1, 1) 
    dX_factor = jnp.broadcast_to(dX_factor, dX_power.shape)  # Shape (batch, 1, n)

    row_idx = jnp.arange(p).reshape(p, 1)  # shape (p, 1)
    even_mask = (row_idx % 2 == 0).astype(float).reshape(1, p, 1)  # shape (1, p, 1)
 
    sin_term = jnp.sin(dX_power)
    even_block = (b * sin_term) + (b_prime * sin_term * dX_factor)

    cos_term = jnp.cos(dX_power)
    odd_block = (c * cos_term) + (c_prime * cos_term * dX_factor)

    R = even_mask * even_block + (1 - even_mask) * odd_block  # shape (batch, p, n)
    
    ### Block S (p x p)
    S = jnp.eye(p)[jnp.newaxis, :, :].repeat(dX.shape[0], axis=0)
    
    ### Assemble fV
    zeros_np = jnp.zeros((dX.shape[0], n, p))
    top_fV = jnp.concatenate([P, zeros_np], axis=2)
    bottom_fV = jnp.concatenate([R, S], axis=2)
    fV = jnp.concatenate([top_fV, bottom_fV], axis=1)
    
    # Build fU
    row_idx_B = jnp.arange(1, n+1).reshape(1, n, 1)  # shape (1, n, 1)
    d = d.reshape(1, n, 1)

    col_idx = jnp.arange(1, q+1).reshape(1, 1, q)       # shape (1, 1, q)
    B = d * row_idx_B * col_idx * (dX ** row_idx_B) + d_prime.reshape(1, n, 1)

    D = jnp.eye(q)[jnp.newaxis, :, :].repeat(dX.shape[0], axis=0)
    zeros_qn = jnp.zeros((dX.shape[0], q, n))
    top_fU = jnp.concatenate([P, B], axis=2)
    bottom_fU = jnp.concatenate([zeros_qn, D], axis=2)
    
    fU = jnp.concatenate([top_fU, bottom_fU], axis=1)

    return (fV, fU)

@partial(jax.jit, static_argnames=('p', 'q'))
def kernel_gl1(p1, p2, p3, p4, p, q, params):

    Y = p1 + p3 - p2 - p4  # shape: (batch, ...)
    Y = Y.reshape((-1, 1, 1))
    
    row_mult = jnp.arange(1, p+1).reshape(1, p, 1)
    col_exp = jnp.arange(1, q+1).reshape(1, 1, q)

    e_vec = params['e'].reshape(1, 1, q)
    N = row_mult * (e_vec * (Y ** col_exp))
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
def to_tuple(n, p, q, image, params):
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
                up[i][j] = from_vector(n, p, q, image[:, i, j], image[:, i, j+1], params)
            else:
                up[i][j] = down[i-1][j]
            
            # Compute left edge
            if j == 0:
                left[i][j] = from_vector(n, p, q, image[:, i+1, j], image[:, i, j], params)
            else:
                left[i][j] = right[i][j-1]
            
            # Compute down edge
            down[i][j] = from_vector(n, p, q, image[:, i+1, j], image[:, i+1, j+1], params)
            
            # Compute right edge
            right[i][j] = from_vector(n, p, q, image[:, i+1, j+1], image[:, i, j+1], params)
            
            # Compute N
            p1 = image[:, i, j]
            p2 = image[:, i+1, j]
            p3 = image[:, i+1, j+1]
            p4 = image[:, i, j+1]
            N[i][j] = kernel_gl1(p1, p2, p3, p4, p, q, params)
    
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
def to_tuple_vectorized(n, p, q, image, params):
    # image: (B, rows, cols)
    B, rows, cols = image.shape
    h, w = rows - 1, cols - 1  # grid dimensions

    # --- Nested vmap helper for from_vector calls
    # from_vector expects scalar (or (B,) vector) inputs; we use a double vmap for grid cells.
    ffn_params = init_ffn_params(n=5, p=3, q=3)
    fvu_func = lambda xt, xs: from_vector(n, p, q, xt, xs, params)
    double_vmap = jax.vmap(jax.vmap(fvu_func, in_axes=(0, 0)), in_axes=(0, 0))
    
    # --- Down edges: for each grid cell (i, j), use image[:, i+1, j] and image[:, i+1, j+1]
    down_v, down_u = double_vmap(image[:, 1:, :w], image[:, 1:, 1:])  # shape: (B, h, w, n+p, n+p)
    
    # --- Right edges: for each grid cell (i, j), use image[:, i+1, j+1] and image[:, i, j+1]
    right_v, right_u = double_vmap(image[:, 1:, 1:], image[:, :h, 1:])  # shape: (B, h, w, n+p, n+p)
    
    # --- Up edges:
    # For the top row (grid row 0), compute directly.
    up_direct_v, up_direct_u = jax.vmap(fvu_func, in_axes=(0, 0))(
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
    left_direct_v, left_direct_u = jax.vmap(fvu_func, in_axes=(0, 0))(
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
        lambda a, b, c, d: kernel_gl1(a, b, c, d, p, q, params),
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
    
    acted = down1_v @ val2 @ jnp.linalg.inv(down1_u)

    down1_u_t = jnp.transpose(down1_u, axes=(*range(down1_u.ndim-2), down1_u.ndim-1, down1_u.ndim-2))
    val2_t = jnp.transpose(val2, axes=(*range(val2.ndim-2), val2.ndim-1, val2.ndim-2))
    solved = jax.scipy.linalg.solve(down1_u_t, val2_t)
    solved_t = jnp.transpose(solved, axes=(*range(solved.ndim-2), solved.ndim-1, solved.ndim-2))
    acted = down1_v @ solved_t

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

@partial(jax.jit, static_argnames=('n',))
def cal_aggregate(elements, n):

    # Define the horizontal composition function for the associative scan.
    def horizontal_associative_compose(cell1, cell2):
        return horizontal_compose(cell1, cell2, n)

    # Horizontal associative scan along columns (axis=1)
    horizontal_scanned = jax.lax.associative_scan(horizontal_associative_compose, elements, axis=1)
    # Shape (rows, cols, batch, ...)
    # Flip the grid along axis 0 (vertical axis) before vertical scan
    flipped = jax.tree_util.tree_map(lambda x: jnp.flip(x, axis=0), horizontal_scanned)

    # Define the vertical composition function for the associative scan.
    def vertical_associative_compose(cell1, cell2):
        return vertical_compose(cell1, cell2, n)

    # Perform vertical associative scan along axis=0 on the flipped grid.
    flipped_scanned = jax.lax.associative_scan(vertical_associative_compose, flipped, axis=0)

    # Flip back along axis 0 to restore the original order.
    aggregated = jax.tree_util.tree_map(lambda x: jnp.flip(x, axis=0), flipped_scanned)

    return aggregated

def jax_scan_aggregate(n, p, q, images, params, jax_jit: bool = True):

    # Each component has shape (rows, cols, batch, ...)
    # calc_time = time.time()
    up_v, up_u, down_v, down_u, left_v, left_u, right_v, right_u, face_tensor = to_tuple_vectorized(n, p, q, images, params)
    elements = (up_v, up_u, down_v, down_u, left_v, left_u, right_v, right_u, face_tensor)
    # for element in elements:
    #     element.block_until_ready()
    # final_time = time.time()
    # print("Time to calculate elements = ", final_time - calc_time)

    if jax_jit:
        compiled_function = jax.jit(cal_aggregate, static_argnames=('n',))
    else:
        compiled_function = cal_aggregate

    # time1 = time.time()
    aggregate = compiled_function(elements, n)
    # for element in aggregate:
    #     element.block_until_ready()
    # time2 = time.time()
    # print("Time to calculate asso scan = ", time2 - time1)

    return aggregate 

def jax_scan_aggregate_benchmark(n, p, q, images, runs, jax_jit: bool = True):

    up_v, up_u, down_v, down_u, left_v, left_u, right_v, right_u, face_tensor = to_tuple_vectorized(n, p, q, images)
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
    batch_size = 64
    n, p, q = 5, 3, 3

    params = {
    'a': jnp.ones((n,)),
    'b': jnp.ones((n,)),
    'b_prime': jnp.ones((n,)),
    'c': jnp.ones((n,)),
    'c_prime': jnp.ones((n,)),
    'd': jnp.ones((n,)), 
    'd_prime': jnp.ones((n,)),
    'e': jnp.ones((q,))
    }
    key = jax.random.PRNGKey(42)
    image = jax.random.uniform(key, shape=(batch_size, 28, 28))
    
    A = time.time()
    # to_tuple_loop = to_tuple(n, p, q, image)
    B = time.time()
    agg1 = jax_scan_aggregate(n, p, q, image, jax_jit=True)
    C = time.time()
    key = jax.random.PRNGKey(41)
    image = jax.random.uniform(key, shape=(batch_size, 28, 28))
    D = time.time()
    agg2 = jax_scan_aggregate(n, p, q, image, jax_jit=True)
    E = time.time()
    # "For looped to tuple = ", B-A, 
    print("For aggregate = ", C-B, "using jit = ", E - D)
    # print((to_tuple_loop[-1][0][-1]))
    # print((to_tuple_vector[-1][0][-1]))f 
    # aggregate = jax_scan_aggregate(n, p, q, image)
    # print(aggregate[-1][0][-1])