from imports import *
from gl0_and_gl1_torch import GL0Element, GL1Element
from custom_matrix_torch import TwoCell, GridOf2Cells, to_custom_matrix
from lifting import from_vector, kernel_gl1

device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO
# - [x]start overleaf and describe the 'lifting' procedure (from_...) for general shapes

# - just time-benchmark horizontal_first_aggregate
#    - [x] eager CPU 0.0011, 0.0008, 0.0014 seconds 
               # GPU 0.110, 0.112, 0.113 seconds
#    - [x] torch.compile CPU 0.001 seconds
               # GPU 0.003 seconds
#    - [x] associative scan



# [x] TODO before and after torch compile
# [x]TODO figure out classes data for pytorch parallization
# [x] TODO overleaf describe Rf 
# TODO Isolate every class and function and optimize 
# TODO time stamp with eager and compile w/parallel scan 

# [x] TODO torch.compile on disrete loops?
# [x] TODO general case for lifting procedure
# [x] TODO play with associative scan

def horizontal_first_aggregate(ImageBatch, a=None, b=None):
    """
    Computes the horizontal-first aggregate for a batch of images.
    Args:
        ImageBatch: Batch of images (type GridOf2Cells)
        a: Rows up to which the aggregation is performed
        b: Columns up to which the aggregation is performed

    Returns:
        Batch of GridOf2Cells of shape (batch_size, 1, 1)
    """
    batch_size = ImageBatch[0,0].value.matrix.shape[0]
    if a is None and b is None:
        a = ImageBatch.rows
        b = ImageBatch.cols

    Aggregate_horizontal = GridOf2Cells(batch_size, a, 1)

    # via Loop
    for i in range(a):
        Aggregate_temp = ImageBatch[i, 0]
        for j in range(1, b):
            Aggregate_temp = Aggregate_temp.horizontal_compose_with(ImageBatch[i, j])
        Aggregate_horizontal[i, 0] = Aggregate_temp.clone()

    # To set final aggregate
    Aggregate = GridOf2Cells(batch_size, 1, 1)

    # via Loop
    temp_value = Aggregate_horizontal[a - 1, 0]
    for i in range(1, a):
        temp_value = temp_value.vertical_compose_with(Aggregate_horizontal[a - (i + 1), 0])
    Aggregate[0,0] = temp_value.clone()

    return Aggregate



def loop_aggregate(n, p, q, images, torch_compile: bool = True):
    """
    Easy to import just this function in other files and get the complete aggregate.
    """    
    Images = to_custom_matrix(n, p, q, images, from_vector, kernel_gl1)

    if torch_compile:
        compiled_function = torch.compile(horizontal_first_aggregate)
    else:
        compiled_function = horizontal_first_aggregate
    
    return  compiled_function(Images)

def loop_aggregate_benchmark(n, p, q, images, runs, torch_compile: bool = False):
    """
    To benchmark loop method
    """
    Images = to_custom_matrix(n, p, q, images, from_vector, kernel_gl1)
    print("in progress...")
    if torch_compile:
        compiled_function = torch.compile(horizontal_first_aggregate)
    else:
        compiled_function = horizontal_first_aggregate

    Time = []
    for i in range(runs):
        start_time = time.time()
        aggregate = compiled_function(Images)
        end_time = time.time()
        Time.append(end_time - start_time)
        if i == 99:
            final_time = end_time - start_time

    print("Using loop - ", f"Average time: {sum(Time)/100},", f"Final time: {final_time},", f"Torch compile = {torch_compile}")




if __name__ == "__main__": 
    m = 5
    batch_size = 2  # Specify the batch size for testing
    n = 2
    p = 1
    q = 1
    torch.manual_seed(42)
    
    # Generate a batch of random images
    images = torch.rand(batch_size, m, m)

    # Map the batch of images
    Images = to_custom_matrix(n, p, q, images, from_vector, kernel_gl1)  # Assuming to_custom_matrix is batch-compatible
    
    # Compute horizontal-first aggregate for the batch
    Time = []
    for i in range(100):
        start_time = time.time()
        Aggregate_1 = horizontal_first_aggregate(Images)
        end_time = time.time()
        elapsed_time = end_time - start_time
        Time.append(elapsed_time)
        # print(f"Execution Time: {elapsed_time} seconds")

    print((sum(Time)/100))

    # print("Loop = ", Aggregate_1[0,0].value.matrix[0])
    # print("Horizontal first aggregate (Batch):")
    print(Aggregate_1[0, 0].value.matrix)
    
    # Compute vertical-first aggregate for the batch
    Aggregate_2 = vertical_first_aggregate(Images)
    # print("Vertical first aggregate (Batch):")
    # print(Aggregate_2[0, 0].value.matrix)
    
    # Ensure aggregates match
    for i in range(batch_size):
        assert torch.allclose(
            Aggregate_1[0, 0].value.matrix,
            Aggregate_2[0, 0].value.matrix
        ), f"Mismatch in aggregates for Image {i + 1}"
    
    print("Aggregates match for all images in the batch.")
    
    # Manual computation for the first image in the batch
    # print("H first manual (First Image in Batch):")
    Step1 = Images[1, 0].horizontal_compose_with(Images[1, 1])
    Step1.validate()
    Step2 = Images[0, 0].horizontal_compose_with(Images[0, 1])
    Step2.validate()
    Step3 = Step1.vertical_compose_with(Step2)
    Step3.validate()
    # print(Step3.value.matrix)
    
    # print("V first manual (First Image in Batch):")
    # print(
    #     Images[1, 0]
    #     .vertical_compose_with(Images[0, 0])
    #     .horizontal_compose_with(
    #         Images[1, 1].vertical_compose_with(Images[0, 1])
    #     )
    #     .value.matrix
    # )
    
    # Validation for another image
    torch.manual_seed(42)
    images = torch.rand(batch_size, 5, 5)
    # print("initial = ", images[0])
    Images = to_custom_matrix(2, 1, 1, images, from_vector, kernel_gl1)
    for i in range(Images.rows):
        for j in range(Images.cols):
            Images[i, j].validate()  # Validate all cells in the batch
    
    # checking associativity
    # print(Images[0, 1].down.tuple[1])
    # print("A = ", ((Images[0, 0].horizontal_compose_with(Images[0, 1])).horizontal_compose_with(Images[0, 2])).value.matrix)
    # print("B = ", (Images[0, 0].horizontal_compose_with(Images[0, 1].horizontal_compose_with(Images[0,2]))).value.matrix)
    # assert torch.allclose(((Images[0, 0].horizontal_compose_with(Images[0, 1])).horizontal_compose_with(Images[0,2])).value.matrix, (Images[0, 0].horizontal_compose_with(Images[0, 1].horizontal_compose_with(Images[0,2]))).value.matrix)
    
    # Check horizontal composition associativity for the batch
    assert torch.allclose(
        Images[0, 0]
        .horizontal_compose_with(Images[0, 1])
        .horizontal_compose_with(Images[0, 2])
        .value.matrix,
        Images[0, 0]
        .horizontal_compose_with(
            Images[0, 1].horizontal_compose_with(Images[0, 2])
        )
        .value.matrix,
        atol = 0.00001
    ), "Associativity check failed for horizontal composition in the batch."
