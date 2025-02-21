import torch
from torch._higher_order_ops.associative_scan import associative_scan

# Define the associative binary function
def custom_fn(x, y):
    """Example of a custom associative function between dictionary elements"""
    a1, b1 = x  # Previous step (scan state)
    a2, b2 = y  # Current step (new input)

    return a1 * b2, a2 * b1  # Example interdependent operation

# Create a dictionary (pytree) of CUDA tensors
input_dict = {
    'a': torch.tensor([1, 2, 3], dtype=torch.float32, device='cuda'),
    'b': torch.tensor([4, 5, 6], dtype=torch.float32, device='cuda')
}

# Convert the dictionary into a tuple of tensors
input_tuple = (input_dict['a'], input_dict['b'])

# Apply associative_scan to the structured input
result_tuple = associative_scan(custom_fn, input_tuple, dim=0, combine_mode='pointwise')

# Convert the tuple back into a dictionary
result_dict = {'a': result_tuple[0], 'b': result_tuple[1]}

# Output the result
print(result_dict)
