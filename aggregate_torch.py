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

# [x] TODO torch.compile on decrete loops?
# [x] TODO general case for lifting procedure
# [x] TODO play with associative scan


@torch.compile
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

                # Loop
    for i in range(a):
        Aggregate_temp = ImageBatch[i, 0]
        for j in range(1, b):
            Aggregate_temp = Aggregate_temp.horizontal_compose_with(ImageBatch[i, j])
        Aggregate_horizontal[i, 0] = Aggregate_temp.clone()

                # Tree Reduction
    # for i in range(a):
    #     result = [ImageBatch[i,j] for j in range(b)]
    #     i = 0
    #     # Tree reduction
    #     while len(result) > 1:
    #         n = len(result)
    #         n_pairs = n // 2  
          
    #         list1 = result[:2 * n_pairs:2]
    #         list2 = result[1:2 * n_pairs:2]
    #         print(i)
    #         new_result = TwoCell.horizontal_compose_with(list1, list2)
    #         i += 1
    #         # for odd elements
    #         if n % 2:
    #             new_result.append(result[-1])
            
    #         result = new_result
        
    #     Aggregate_horizontal[i, 0] = new_result[0].clone()

    # To set final aggregate
    Aggregate = GridOf2Cells(batch_size, 1, 1)

                    # Loop
    temp_value = Aggregate_horizontal[a - 1, 0]
    for i in range(1, a):
        temp_value = temp_value.vertical_compose_with(Aggregate_horizontal[a - (i + 1), 0])
    Aggregate[0,0] = temp_value.clone()

                    # Tree reduction
    # result_aggregate = [Aggregate_horizontal[i, 0] for i in range(a)]
    # while len(result_aggregate) > 1:
    #     n = len(result)
    #     n_pairs = n // 2  
        
    #     list1 = result_aggregate[:2 * n_pairs:2]
    #     list2 = result_aggregate[1:2 * n_pairs:2]
        
    #     new_result = TwoCell.vertical_compose_with(list1, list2)
        
    #     # for odd elements
    #     if n % 2:
    #         new_result.append(result[-1])
        
    #     result_aggregate = new_result

    # Aggregate[0, 0] = result_aggregate[0].clone()

    return Aggregate

@torch.compile
def vertical_first_aggregate(ImageBatch, a=None, b=None):
    """
    Computes the vertical-first aggregate for a batch of images.
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

    Aggregate_vertical = GridOf2Cells(batch_size, 1, b)

    for i in range(b):
        Aggregate_temp = ImageBatch[a - 1, i]
        for j in range(1, a):
            Aggregate_temp = Aggregate_temp.vertical_compose_with(ImageBatch[a - (j + 1), i])
        Aggregate_vertical[0, i] = Aggregate_temp.clone()

    Aggregate = GridOf2Cells(batch_size, 1, 1)
    temp_value = Aggregate_vertical[0, 0]

    for i in range(1, b):
        temp_value = temp_value.horizontal_compose_with(Aggregate_vertical[0, i])

    Aggregate[0, 0] = temp_value.clone()
    return Aggregate

if __name__ == "__main__": 
    m = 3
    batch_size = 2  # Specify the batch size for testing
    n = 2
    p = 1
    q = 1
    torch.manual_seed(42)
    
    # def from_vector(m, Xt, Xs):
    #     n, p, q = 2, 1, 1
    #     fV = torch.eye(n + p).repeat(m, 1, 1)
    #     fU = torch.eye(n + q).repeat(m, 1, 1)
    #     dX = Xs - Xt

    #     fV[:, 0, 0] = fU[:, 0, 0] = torch.exp(dX)
    #     fV[:, 1, 1] = fU[:, 1, 1] = torch.exp(dX ** 2)
    #     fV[:, 2, 0] = torch.sin(dX)
    #     fV[:, 2, 1] = dX ** 5
    #     fU[:, 0, 2] = dX ** 3
    #     fU[:, 1, 2] = 7 * dX
 
    #     return GL0Element(m, n, p, q, fV, fU)


    # def kernel_gl1(p1, p2, p3, p4):
    #     return (p1+p3-p2-p4).unsqueeze(-1).unsqueeze(-1)

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
    # print(Aggregate_1[0, 0].value.matrix)
    
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
    print("initial = ", images[0])
    Images = to_custom_matrix(2, 1, 1, images, from_vector, kernel_gl1)
    for i in range(Images.rows):
        for j in range(Images.cols):
            Images[i, j].validate()  # Validate all cells in the batch
    
    # checking associativity
    print(Images[0, 2].value.matrix)
    assert torch.allclose(((Images[0, 0].horizontal_compose_with(Images[0, 1])).horizontal_compose_with(Images[0,2])).value.matrix, (Images[0, 0].horizontal_compose_with(Images[0, 1].horizontal_compose_with(Images[0,2]))).value.matrix)
    
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
        .value.matrix
    ), "Associativity check failed for horizontal composition in the batch."
