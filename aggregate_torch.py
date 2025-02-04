from imports import *
from gl0_and_gl1_torch import GL0Element, GL1Element
from custom_matrix_torch import TwoCell, GridOf2Cells, to_custom_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO
# - [x]start overleaf and describe the 'lifting' procedure (from_...) for general shapes

# - just time-benchmark horizontal_first_aggregate
#    - [x] eager
#    - [x] torch.compile
#    - [ ] associative scan

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

    # for i in range(a):
    #     Aggregate_temp = ImageBatch[i, 0]
    #     for j in range(1, b):
    #         Aggregate_temp = Aggregate_temp.horizontal_compose_with(ImageBatch[i, j])
    #     Aggregate_horizontal[i, 0] = Aggregate_temp.clone()

    for i in range(a):
        temp_h = [ImageBatch[i, j] for j in range(b)]
        
        while len(temp_h) > 1:
            temp = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(lambda a, b: getattr(a, "horizontal_compose_with")(b), temp_h[i], temp_h[i+1]) 
                for i in range(0, len(temp_h)-1, 2)]

                for future in concurrent.futures.as_completed(futures):
                    temp.append(future.result()) 

            if len(temp_h) % 2 == 1:
                temp.append(temp_h[-1])

            temp_h = temp

        Aggregate_horizontal[i, 0] = temp_h[0].clone()

    Aggregate = GridOf2Cells(batch_size, 1, 1)
    
    # temp_value = Aggregate_horizontal[a - 1, 0]

    # for i in range(1, a):
    #     temp_value = temp_value.vertical_compose_with(Aggregate_horizontal[a - (i + 1), 0])

    temp_v = [Aggregate_horizontal[i, 0] for i in range(a-1, -1, -1)]
        
    while len(temp_v) > 1:
        temp = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda a, b: getattr(a, "vertical_compose_with")(b), temp_v[i], temp_v[i+1]) 
            for i in range(0, len(temp_v)-1, 2)]

            for future in concurrent.futures.as_completed(futures):
                temp.append(future.result()) 

        if len(temp_v) % 2 == 1:
            temp.append(temp_v[-1])

        temp_v = temp

    Aggregate[0, 0] = temp_v[0].clone()

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
    torch.manual_seed(42)
    
    def from_vector(m, Xt, Xs):
        n, p, q = 2, 1, 1
        fV = torch.eye(n + p).repeat(m, 1, 1)
        fU = torch.eye(n + q).repeat(m, 1, 1)
        dX = Xs - Xt

        fV[:, 0, 0] = fU[:, 0, 0] = torch.exp(dX)
        fV[:, 1, 1] = fU[:, 1, 1] = torch.exp(dX ** 2)
        fV[:, 2, 0] = torch.sin(dX)
        fV[:, 2, 1] = dX ** 5
        fU[:, 0, 2] = dX ** 3
        fU[:, 1, 2] = 7 * dX
 
        return GL0Element(m, n, p, q, fV, fU)


    def kernel_gl1(p1, p2, p3, p4):
        return (p1+p3-p2-p4).unsqueeze(-1).unsqueeze(-1)

    # Generate a batch of random images
    images = torch.rand(batch_size, m, m)
    
    # Map the batch of images
    Images = to_custom_matrix(images, from_vector, kernel_gl1)  # Assuming to_custom_matrix is batch-compatible
    
    # Compute horizontal-first aggregate for the batch
    Aggregate_1 = horizontal_first_aggregate(Images)
    # print("Horizontal first aggregate (Batch):")
    # print(Aggregate_1[0, 0].value.matrix)
    
    # Compute vertical-first aggregate for the batch
    Aggregate_2 = vertical_first_aggregate(Images)
    # print("Vertical first aggregate (Batch):")
    # print(Aggregate_2[0, 0].value.matrix)
    
    # Ensure aggregates match
    for i in range(batch_size):
        assert np.allclose(
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
    images = torch.rand(batch_size, 5, 5)
    Images = to_custom_matrix(images, from_vector, kernel_gl1)
    for i in range(Images.rows):
        for j in range(Images.cols):
            Images[i, j].validate()  # Validate all cells in the batch
    
    # Check horizontal composition associativity for the batch
    assert np.allclose(
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
