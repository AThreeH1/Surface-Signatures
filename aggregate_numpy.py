from imports import *
from gl0_and_gl1 import GL0Element, GL1Element
from custom_matrix import TwoCell, GridOf2Cells, mapping

def horizontal_first_aggregate(Image, a=None, b=None):
    """
    Takes in a Image of input image and returns the signature of the image
    Args:
        Image = Image of the image (type GridOf2Cells)
        a = signature till row a
        b = signature till column b

    Returns:
        GridOf2Cells of shape 1,1

    Workings:
        Image(2 dims) --> Image(1 dim: Horizontal) --> Signature matrix
    """

    if a == None and b == None:
        a = Image.rows
        b = Image.cols

    Aggregate_horizontal = GridOf2Cells(a, 1)
    # print('XXX temp_value=', temp_value, type(temp_value))

    for i in range(a):
        # print('i=', i) 
        Aggregate_temp = Image[i, 0]
        for j in range(b-1):
            # print('j=', i)
            Aggregate_temp = TwoCell.horizontal_compose_with(Aggregate_temp, Image[i, j+1])
            # print('Aggregate_temp=', Aggregate_temp, type(Aggregate_temp))
        # print('Aggregate_temp=', Aggregate_temp, type(Aggregate_temp))
        clone = Aggregate_temp.clone()
        # print('clone=', clone, type(clone))
        Aggregate_horizontal[i, 0] = clone
    
    # print('a-1=',a-1)
    temp_value = Aggregate_horizontal[a - 1, 0]
    # print('temp_value=', temp_value, type(temp_value))

    Aggregate = GridOf2Cells(1,1)
    for i in range(1, a):
        # print('i=', i)
        temp_value = TwoCell.vertical_compose_with(temp_value, Aggregate_horizontal[a - (i + 1), 0])
        # print('temp_value=', temp_value, type(temp_value))

    clone = temp_value.clone()
    # print('clone=', clone, type(clone))
    Aggregate[0,0] = clone

    return Aggregate

def vertical_first_aggregate(Image, a = None, b = None):

    if a == None and b == None:
        a = Image.rows
        b = Image.cols

    Aggregate_vertical = GridOf2Cells(1, b)

    for i in range(b):
        Aggregate_temp = Image[a-1, i]
        for j in range(1, a):
            Aggregate_temp = TwoCell.vertical_compose_with(Aggregate_temp, Image[a-(j+1), i])
        Aggregate_vertical[0, i] = Aggregate_temp.clone()
    
    Aggregate = GridOf2Cells(1,1)

    temp_value = Aggregate_vertical[0, 0]

    for i in range(1, b):
        temp_value = TwoCell.horizontal_compose_with(temp_value, Aggregate_vertical[0, i])

    Aggregate[0,0] = temp_value.clone()
    return Aggregate

if __name__ == "__main__":
    m = 3
    np.random.seed(42)
    image = np.random.rand(m, m)

    Image = mapping(image)
    print("Starts here")
    # print('image=\n', image, type(image))
    # print('Image=\n', Image, type(Image))
    Aggregate_1 = horizontal_first_aggregate(Image)
    # print()
    # print()
    print("Horizontal first aggregate =\n", Aggregate_1[0,0].value.matrix)
    # print()
    Aggregate_2 = vertical_first_aggregate(Image)
    print("Vertical first aggregate =\n",Aggregate_2[0,0].value.matrix)
    # assert np.allclose(Aggregate_1[0,0].value.matrix, Aggregate_2[0,0].value.matrix)

    print("H first manual = ")
    
    Step1 = Image[1,0].horizontal_compose_with( Image[1,1] )
    check_1 = Step1.validate()
    Step2 = Image[0,0].horizontal_compose_with( Image[0,1] )
    check_2 = Step2.validate()
    Step3 = Step1.vertical_compose_with( Step2 )
    check_3 = Step3.validate()
    print( Step3.value.matrix )

    print("V first manual = ")
    print( Image[1,0].vertical_compose_with( Image[0,0] ).horizontal_compose_with(  Image[1,1].vertical_compose_with( Image[0,1] ) ).value.matrix )
    # print( Image[0,0].vertical_compose_with( Image[1,])

    image = np.random.rand(5, 5)
    Image = mapping(image)
    for i in range(Image.rows):
        for j in range(Image.cols):
            Image[i,j].validate()
    assert np.allclose( Image[0,0].horizontal_compose_with(Image[0,1]).horizontal_compose_with( Image[0,2]).value.matrix, 
                        Image[0,0].horizontal_compose_with(Image[0,1].horizontal_compose_with( Image[0,2])).value.matrix )

# TODO sanity checks (order doesn't matter)
# TODO image to image
# TODO train on mnist dataset