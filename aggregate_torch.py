from imports import *
from gl0_and_gl1_torch import GL0Element, GL1Element
from custom_matrix_torch import TwoCell, GridOf2Cells, mapping


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

    for i in range(a):
        Aggregate_temp = ImageBatch[i, 0]
        for j in range(1, b):
            Aggregate_temp = Aggregate_temp.horizontal_compose_with(ImageBatch[i, j])
        Aggregate_horizontal[i, 0] = Aggregate_temp.clone()

    Aggregate = GridOf2Cells(batch_size, 1, 1)
    temp_value = Aggregate_horizontal[a - 1, 0]

    for i in range(1, a):
        temp_value = temp_value.vertical_compose_with(Aggregate_horizontal[a - (i + 1), 0])

    Aggregate[0, 0] = temp_value.clone()
    return Aggregate

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
    
    # Generate a batch of random images
    images = torch.rand(batch_size, m, m)
    
    # Map the batch of images
    Images = mapping(images)  # Assuming mapping is batch-compatible
    
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
    Images = mapping(images)
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
