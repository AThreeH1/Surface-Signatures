import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def from_vector(n, p, q, batch_size, Xt, Xs):
    """
    Constructs the matrices fV and fU based on input parameters using PyTorch.

    Args:
        n, p, q : integers
        Xt (tensor): Reference input points, shape (batch, ...).
        Xs (tensor): Target input points, shape (batch, ...).

    Returns:
        tuple: (fV, fU)
            - fV (tensor): Shape (batch, n+p, n+p)
            - fU (tensor): Shape (batch, n+q, n+q)
    """

    a = torch.ones(n).to(device)
    b = torch.ones(n).to(device)
    b_prime = torch.ones(n).to(device)
    c = torch.ones(n).to(device)
    c_prime = torch.ones(n).to(device)
    d = torch.ones(n).to(device)
    d_prime = torch.ones(n).to(device)

    batch_size = Xt.shape[0]
    dX = (Xs - Xt).reshape(batch_size, 1, 1)  # shape (batch, 1, 1)

    # Block P (n x n diagonal)
    powers = torch.arange(1, n + 1, dtype=dX.dtype, device=dX.device).reshape(1, n)  # (1, n)
    dX_flat = dX.view(batch_size, 1)  # (batch, 1)
    diag_vals = a * torch.exp(dX_flat ** powers)  # (batch, n)

    P = torch.zeros((batch_size, n, n), dtype=dX.dtype, device=dX.device)
    for i in range(n):
        P[:, i, i] = diag_vals[:, i]

    # Block R (p x n)
    dX_power = dX_flat ** torch.arange(1, n + 1, dtype=dX.dtype, device=dX.device).reshape(1, n)
    dX_power = dX_power.unsqueeze(1)  # (batch, 1, n)
    dX_factor = dX_flat.view(batch_size, 1, 1).expand_as(dX_power)  # (batch, 1, n)

    row_idx = torch.arange(p, device=dX.device).view(p, 1)  # (p, 1)
    even_mask = (row_idx % 2 == 0).float().view(1, p, 1)  # (1, p, 1)

    sin_term = torch.sin(dX_power)
    even_block = (b * sin_term) + (b_prime * sin_term * dX_factor)

    cos_term = torch.cos(dX_power)
    odd_block = (c * cos_term) + (c_prime * cos_term * dX_factor)

    R = even_mask * even_block + (1 - even_mask) * odd_block  # (batch, p, n)

    # Block S (identity, p x p)
    S = torch.eye(p, dtype=dX.dtype, device=dX.device).unsqueeze(0).expand(batch_size, -1, -1)  # (batch, p, p)

    # Assemble fV
    zeros_np = torch.zeros((batch_size, n, p), dtype=dX.dtype, device=dX.device)
    top_fV = torch.cat([P, zeros_np], dim=2)  # (batch, n, n+p)
    bottom_fV = torch.cat([R, S], dim=2)      # (batch, p, n+p)
    fV = torch.cat([top_fV, bottom_fV], dim=1)  # (batch, n+p, n+p)

    # Block B (n x q)
    row_idx_B = torch.arange(1, n + 1, dtype=dX.dtype, device=dX.device).view(1, n, 1)
    col_idx = torch.arange(1, q + 1, dtype=dX.dtype, device=dX.device).view(1, 1, q)
    d = d.view(1, n, 1)
    d_prime = d_prime.view(1, n, 1)

    B = d * row_idx_B * col_idx * (dX ** row_idx_B) + d_prime  # (batch, n, q)

    # Block D (identity, q x q)
    D = torch.eye(q, dtype=dX.dtype, device=dX.device).unsqueeze(0).expand(batch_size, -1, -1)  # (batch, q, q)
    zeros_qn = torch.zeros((batch_size, q, n), dtype=dX.dtype, device=dX.device)

    top_fU = torch.cat([P, B], dim=2)         # (batch, n, n+q)
    bottom_fU = torch.cat([zeros_qn, D], dim=2)  # (batch, q, n+q)
    fU = torch.cat([top_fU, bottom_fU], dim=1)  # (batch, n+q, n+q)

    return fV, fU

def kernel_gl1(p1, p2, p3, p4, p, q):
    """
    Given four tensors (typically scalars per batch) compute Y = X1 + X3 - X2 - X4.
    Then return the kernel matrix N of shape (m, p, q) defined by
      N[i,j] = (i+1) * Y^(j+1)
    for each batch element.
    """
    Y = (p1 + p3 - p2 - p4).to(device)  # shape: (m, ...)
    m = Y.shape[0]
    # Reshape Y to (m,1,1) for broadcasting
    Y = Y.view(m, 1, 1)
    # Row multipliers: shape (1, p, 1) with values [1, 2, ..., p]
    row_mult = torch.arange(1, p+1, device=device).view(1, p, 1).float()
    # Column exponents: shape (1, 1, q) with values [1, 2, ..., q]
    col_exp = torch.arange(1, q+1, device=device).view(1, 1, q).float()
    # Compute N so that N[i,j] = (i+1) * Y^(j+1)
    N = row_mult * (Y ** col_exp)  # shape: (m, p, q)
    return N.to(device)
