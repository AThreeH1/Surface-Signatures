"""
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
from aggregate_using_scan import scan_aggregate_benchmark
from aggregate_torch import loop_aggregate_benchmark
device = "cuda" if torch.cuda.is_available() else "cpu"

n = 2
p = 1
q = 1
batch_size = 5
image_width = 20
image_height = 20

torch.manual_seed(42)
images = torch.rand(batch_size, image_height, image_width).to(device)

scan_aggregate_benchmark(n, p, q, images, torch_compile=True)
loop_aggregate_benchmark(n, p, q, images, torch_compile=True)
scan_aggregate_benchmark(n, p, q, images, torch_compile=False)
loop_aggregate_benchmark(n, p, q, images, torch_compile=False)
