import torch
from torch._higher_order_ops.associative_scan import associative_scan

# Define the associative binary function
def custom_fn(x, y):
    """Example of a custom associative function between dictionary elements"""
    # a1, b1 = x  # Previous step (scan state)
    # a2, b2 = y  # Current step (new input)

    # return a1 + b2, a2 * b1  # Example interdependent operation
    # return a1 + 1, b1 + 10
    return x+y

# Create a dictionary (pytree) of CUDA tensors
input_dict = {
    'a': torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda'),
    'b': torch.tensor([4, 5, 6], dtype=torch.float32, device='cuda')
}

# Convert the dictionary into a tuple of tensors
input_tuple = (input_dict['a'], input_dict['b'])

input = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.float32, device='cuda')
# print(input_tuple)

# Apply associative_scan to the structured input
result_tuple = associative_scan(custom_fn, input, dim=0, combine_mode='pointwise')
print(result_tuple)

# Convert the tuple back into a dictionary
# result_dict = {'a': result_tuple[0], 'b': result_tuple[1]}

# Output the result
# print(result_dict)




# result_tuple = associative_scan(custom_fn, input_tuple, dim=0, combine_mode='pointwise')

#     # Now compute Y without the scalar factor.
#     Y = p1 + p3 - p2 - p4  # shape: (batch, ...)
#     Y = Y.reshape((-1, 1, 1))
    
#     # Create row multipliers (j = 1,...,p)
#     row_mult = jnp.arange(1, p+1).reshape(1, p, 1)
#     # Create column exponents (i = 1,...,q)
#     col_exp = jnp.arange(1, q+1).reshape(1, 1, q)
    
#     # e is now a vector of length q; reshape to allow broadcasting
#     e_vec = params['e'].reshape(1, 1, q)
    
#     # Compute N: each element is j * e_i * Y^i.
#     N = row_mult * (e_vec * (Y ** col_exp))
#     return N

