import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Assume device is defined (e.g., device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def from_vector(n, p, q, m, Xt, Xs):
    # Compute the batch of differences dX (assumed to be a scalar per batch element)
    dX = (Xs - Xt).to(device)  # shape: (m, ...), e.g. (m,)
    # Reshape to (m, 1, 1) for broadcasting in our operations
    dX = dX.view(m, 1, 1)
    
    ### Build block for fV and fU: the top-left block P (n x n)
    # For each batch element, we want a diagonal matrix with:
    # [exp(dX^1), exp(dX^2), ..., exp(dX^n)]
    powers = torch.arange(1, n+1, device=device, dtype = torch.float64).view(1, n)  # shape (1, n)
    # dX flattened to (m,1) so that each batch element is raised to each power:
    dX_flat = dX.view(m, 1)  # shape (m, 1)
    diag_vals = torch.exp(dX_flat ** powers)  # shape (m, n)
    # Create a batch of diagonal matrices
    P = torch.diag_embed(diag_vals)  # shape: (m, n, n)
    
    ### Build the bottom-left block R for fV (p x n)
    # For each batch element, create a matrix whose j-th column is:
    # row0: sin(dX^(j+1)), row1: cos(dX^(j+1)), row2: sin(dX^(j+1)), etc.
    dX_power = dX_flat ** torch.arange(1, n+1, device=device).view(1, n)  # shape (m, n)
    # Expand to (m, p, n)
    dX_power = dX_power.unsqueeze(1).expand(m, p, n)
    # Create a row index tensor for p rows
    row_idx = torch.arange(p, device=device).view(p, 1)
    # For even-indexed rows (0,2,…) use sin; for odd-indexed rows use cos.
    even_mask = (row_idx % 2 == 0).float().view(1, p, 1)  # shape (1, p, 1)
    R = even_mask * torch.sin(dX_power) + (1 - even_mask) * torch.cos(dX_power)
    # R shape: (m, p, n)
    
    ### Bottom-right block for fV: S as an identity of size p
    S = torch.eye(p, device=device).unsqueeze(0).repeat(m, 1, 1)  # shape: (m, p, p)
    
    ### Assemble fV as a block matrix
    # Top block: [P, 0] with P (m, n, n) and a zeros block (m, n, p)
    zeros_np = torch.zeros(m, n, p, device=device)
    top_fV = torch.cat([P, zeros_np], dim=2)  # shape: (m, n, n+p)
    # Bottom block: [R, S]
    bottom_fV = torch.cat([R, S], dim=2)  # shape: (m, p, n+p)
    # fV overall is (m, n+p, n+p)
    fV = torch.cat([top_fV, bottom_fV], dim=1)
    
    ### Now build fU:
    # Top-right block B (n x q): for each batch element, for row i and column j:
    # B[i,j] = (i+1) * (dX)^(j+1)
    row_factors = torch.arange(1, n+1, device=device).view(1, n, 1)  # shape (1, n, 1)
    col_exponents = torch.arange(1, q+1, device=device).view(1, 1, q)  # shape (1, 1, q)
    B = row_factors * (dX ** col_exponents)  # shape: (m, n, q)
    
    # Bottom-right block D for fU: identity (q x q)
    D = torch.eye(q, device=device).unsqueeze(0).repeat(m, 1, 1)  # shape: (m, q, q)
    
    # Assemble fU: top block is [P, B] (P: m x n x n, B: m x n x q)
    top_fU = torch.cat([P, B], dim=2)  # shape: (m, n, n+q)
    # Bottom block is [0, D] (zeros: (m, q, n))
    zeros_qn = torch.zeros(m, q, n, device=device)
    bottom_fU = torch.cat([zeros_qn, D], dim=2)  # shape: (m, q, n+q)
    # fU overall is (m, n+q, n+q)
    fU = torch.cat([top_fU, bottom_fU], dim=1)
    
    return (fV, fU)

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
