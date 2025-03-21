"""
# TODO runs in jax and jax compile
# TODO loops in functional 
# TODO Tensor sizes 

The repository contains two ways to calculate surface signature

1. Object oriented 

    This consists of classes and tests to validate GL0 and GL1 structure from a crossed module. There are number of files 
    in this repository the work together to calculate the net aggregate or surface signature.

        i. gl0_and_gl1_torch.py (gl0_and_gl1.py is the numpy implementation of the same): This file implements the GL0 and GL1 structure. 
                It validates, and tests various identities such as peiffer, equivarience and tau morphism. This file also defines the 
                multiplication, inverse, act on function, feedback, etc of elements in GL0 and GL1. 

        ii. custom_matrix_torch.py (custom_matrix.py is the numpy implementation of the same): This defines a way to map the input image onto 
                a custom matrix structure using lifting procedure which enables user to view the GL0 and GL1 elements in a systematic way.
                Also defines horizontal_compose_with and vertical_compose_with to aggregate to adjecent individual cells/faces.

        iii. aggregate_torch.py (aggregate_numpy.py is the numpy implementation of the same): This file calculates the final aggregate of the 
                input images. In this, we loop through the columns and rows of the custom matrix to calculate the aggregate/surface signature.

2. Functional - Runs only on gpu

        aggregate_using_scan.py: This file contains the functional implementation using tensors to calculate surface signature. This file does not test 
    identities of the crossed module and is assumed to hold true. The final outcome is cross verified with the object oriented implementation.
    GL0 and GL1 elements in this file is stored in a tuple of tensors instead of custom matrix.  
    Aggregate is calculated using associative scan to parallelise operations. (Given that horizontal_compose_with and vertical_compose_with functions
    are associative)
        This calculates the surface signature of image till every point in image instead of just net signature.

Other files in the repository:

    A. imports.py: Imports all necessary libraries.

    B. lifting.py: Defines from_vector and kernel_gl1 functions to lift input image elements (usually pixel data) to GL0 elements or part of GL1 element.

    C. classifier.py: A basic model to test out surface signature. Uses FNN after aggregate in order to classify.

(trial.py file can be ignored)

PLEASE RUN THIS README FILE FOR STATISTICS AND BENCHMARKING
"""

import torch
import jax
import jax.numpy as jnp
import aggregate_using_scan
from aggregate_using_scan import scan_aggregate_benchmark, scan_aggregate
from aggregate_torch import loop_aggregate_benchmark, loop_aggregate
from aggregate_jax import jax_scan_aggregate, jax_scan_aggregate_benchmark
device = "cuda" if torch.cuda.is_available() else "cpu"
jax.config.update("jax_enable_x64", True)

n = 2
p = 1
q = 1
batch_size = 1

# Higher numbers below will benefit more from associative scan
image_width = 5
image_height = 5

torch.manual_seed(42)
images = torch.rand(batch_size, image_height, image_width, device = device, dtype = torch.float64)
images_numpy = images.detach().cpu().numpy()  
images_jax = jax.device_put(jnp.array(images_numpy))

aggrgate_function_loop = aggregate_using_scan.loop_aggregate(n, p, q, images)
aggregate_using_scan = scan_aggregate(n, p, q, images, torch_compile = False)
aggregate_using_loop = loop_aggregate(n, p, q, images, torch_compile = False)
# aggregate_using_scan_jax = jax_scan_aggregate(n, p, q, images_jax, jax_jit = True)
print("aggregate_using_scan = ", aggregate_using_scan[-1][0][-1])
# print("aggregate_using_scan_JAX = ",aggregate_using_scan_jax[-1][0][-1])
print("aggregate_using_loop = ",aggregate_using_loop[0,0].value.matrix)
print("aggrgate_function_loop = ", aggrgate_function_loop[-1][0][-1])

# assert torch.allclose(aggregate_using_scan[-1][0][-1], aggregate_using_loop[0,0].value.matrix, atol = 0.1)

xxx
runs = 200
# Functions below auto prints benchmarks. Final time is time required for the 100th run.
# scan_aggregate_benchmark(n, p, q, images, runs=runs, torch_compile=True)
# loop_aggregate_benchmark(n, p, q, images, runs=runs, torch_compile=True)
jax_scan_aggregate_benchmark(n, p, q, images_jax, runs=runs, jax_jit=True)
# Associative scan internally also has torch compile - which is not disabled.
print()
print()
scan_aggregate_benchmark(n, p, q, images, runs=runs, torch_compile=False)
# loop_aggregate_benchmark(n, p, q, images, runs=runs, torch_compile=False)
jax_scan_aggregate_benchmark(n, p, q, images_jax, runs=runs, jax_jit=False)
print()
print()



