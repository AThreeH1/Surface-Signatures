def composition(Map):
    """
    Takes in a Map of input image and returns the signature of the image
    Args:
        Map = Map of the image

    Workings:
        Map(2 dims) --> Map(1 dim: Horizontal) --> Signature matrix
    """
    Aggregate_horizontal = CustomMatrix(Map.rows, 1)

    for i in range(Map.rows):
        Aggregate_dim1 = Map[i, 0].value
        DOWN = Map[i, 0].down
        UP = Map[i, 0].up
        LEFT = Map[i, 0].left
        Right = Map[i, Map.cols-1]. right
        for j in range(Map.cols - 1 ):
            Aggregate_dim1 = (DOWN.act_on(Aggregate_dim1)) * (Map[i, j+1].value)
            DOWN = DOWN * Map[i, j+1].down
            UP = UP * Map[i, j+1].up
        Aggregate_horizontal[i, 0].value = Aggregate_dim1
        Aggregate_horizontal[i, 0].down = DOWN
        Aggregate_horizontal[i, 0].up = UP
        Aggregate_horizontal[i, 0].left = LEFT
        Aggregate_horizontal[i, 0].right = Right

# TODO change this to a method, horizontal and vertical

    Aggregate = CustomMatrix(1,1)
    Aggregate_dim2 = Aggregate_horizontal[0,0].value
    DOWN_1 = Aggregate_horizontal[Aggregate_horizontal.rows - 1, 0].down
    UP_1 = Aggregate_horizontal[0,0].up
    LEFT_1 = Aggregate_horizontal[0,0].left
    RIGHT_1 = Aggregate_horizontal[0,0].right

    for i in range(Aggregate_horizontal.rows-1):
        Aggregate_dim2 = (LEFT_1.act_on(Aggregate_dim2)) * (Aggregate_horizontal[i+1, 0].value)
        LEFT_1 = LEFT_1 * Aggregate_horizontal[i+1, 0].left
        RIGHT_1 = RIGHT_1 * Aggregate_horizontal[i+1, 0].right

    Aggregate[0, 0].value = Aggregate_dim2
    Aggregate[0, 0].left = LEFT_1
    Aggregate[0, 0].right = RIGHT_1
    Aggregate[0, 0].up = UP_1
    Aggregate[0, 0].down = DOWN_1

    return Aggregate

def vertical_first_composition(Map):
    """
    Takes in a Map of input image and returns the signature of the image
    Args:
        Map = Map of the image

    Workings:
        Map(2 dims) --> Map(1 dim: Vertical) --> Signature matrix
    """
    # Step 1: Aggregate Vertically
    Aggregate_vertical = CustomMatrix(1, Map.cols)

    for j in range(Map.cols):
        Aggregate_dim1 = Map[0, j].value
        LEFT = Map[0, j].left
        RIGHT = Map[0, j].right
        UP = Map[0, j].up
        DOWN = Map[Map.rows - 1, j].down
        for i in range(Map.rows - 1):
            Aggregate_dim1 = (LEFT.act_on(Aggregate_dim1)) * (Map[i + 1, j].value)
            LEFT = LEFT * Map[i + 1, j].left
            RIGHT = RIGHT * Map[i + 1, j].right

        Aggregate_vertical[0, j].value = Aggregate_dim1
        Aggregate_vertical[0, j].left = LEFT
        Aggregate_vertical[0, j].right = RIGHT
        Aggregate_vertical[0, j].up = UP
        Aggregate_vertical[0, j].down = DOWN

    # Step 2: Aggregate Horizontally
    Aggregate = CustomMatrix(1, 1)
    Aggregate_dim2 = Aggregate_vertical[0, 0].value
    LEFT_1 = Aggregate_vertical[0, 0].left
    RIGHT_1 = Aggregate_vertical[0, Aggregate_vertical.cols - 1].right
    UP_1 = Aggregate_vertical[0, 0].up
    DOWN_1 = Aggregate_vertical[0, 0].down

    for j in range(Aggregate_vertical.cols - 1):
        Aggregate_dim2 = (UP_1.act_on(Aggregate_dim2)) * (Aggregate_vertical[0, j + 1].value)
        UP_1 = UP_1 * Aggregate_vertical[0, j + 1].up
        DOWN_1 = DOWN_1 * Aggregate_vertical[0, j + 1].down

    Aggregate[0, 0].value = Aggregate_dim2
    Aggregate[0, 0].left = LEFT_1
    Aggregate[0, 0].right = RIGHT_1
    Aggregate[0, 0].up = UP_1
    Aggregate[0, 0].down = DOWN_1

    return Aggregate

if __name__ == "__main__":
    m = 5
    image = np.random.rand(m, m)

    Map = mapping(image)
    Aggregate = composition(Map)
    Aggregate_2 = vertical_first_composition(Map)
    assert np.allclose(Aggregate[0,0].value.matrix, Aggregate_2[0,0].value.matrix)

# TODO sanity checks (order doesn't matter)
# TODO image to image
# TODO train on mnist dataset