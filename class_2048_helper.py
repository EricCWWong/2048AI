import numpy as np


def merge_row(row):
    '''
        This function merges a given row to the right.
        It is same as swiping right in the game 2048.

        Example:
        >>> # our input...
        >>> row = [2,0,2,0]
        >>> new_row = merge_row(row)
        >>> new_row
        >>> # our output...
        >>> [0,0,0,4]
    '''
    no_zero_row = []
    for el in row:
        if el != 0:
            no_zero_row.append(el)

    new_row = []
    
    if len(no_zero_row) == 2:
        if no_zero_row[0] == no_zero_row[1]:
            new_row.append(no_zero_row[0] + no_zero_row[1])
        else:
            new_row = no_zero_row
    elif len(no_zero_row) == 3:
        if no_zero_row[0] == no_zero_row[1]:
            new_row.append(no_zero_row[0] + no_zero_row[1])
            new_row.append(no_zero_row[2])
        elif no_zero_row[1] == no_zero_row[2]:
            new_row.append(no_zero_row[0])
            new_row.append(no_zero_row[1] + no_zero_row[2])
        else:
            new_row = no_zero_row
    elif len(no_zero_row) == 4:
        if no_zero_row[0] == no_zero_row[1]:
            new_row.append(no_zero_row[0] + no_zero_row[1])
            if no_zero_row[2] == no_zero_row[3]:
                new_row.append(no_zero_row[2] + no_zero_row[3])
            else:
                new_row.append(no_zero_row[2])
                new_row.append(no_zero_row[3])
        
        elif no_zero_row[1] == no_zero_row[2]:
            new_row.append(no_zero_row[0])
            new_row.append(no_zero_row[1] + no_zero_row[2])
            new_row.append(no_zero_row[3])
        elif no_zero_row[2] == no_zero_row[3]:
            new_row.append(no_zero_row[0])
            new_row.append(no_zero_row[1])
            new_row.append(no_zero_row[2] + no_zero_row[3])
        else:
            new_row = no_zero_row
    else:
        new_row = no_zero_row

    if len(new_row) != 4:
        no_empty = 4 - len(new_row)
        for _ in range(no_empty):
            new_row = [0] + new_row

    return new_row

def merge(grid):
    '''
        This uses the merge row function above and calculate
        the new grid after merging.

        Example:
        >>> # our input...
        >>> grid = [[2,0,2,0]
                    [0,0,4,0]
                    [4,4,0,0]
                    [2,8,0,2]]
        >>> new_grid = merge(grid)
        >>> new_grid
        >>> # our output...
        >>> [[0,0,0,4]
             [0,0,0,4]
             [0,0,0,8]
             [0,2,8,2]]
    '''
    
    tiles = grid
    new_grid = []
    for row in tiles:
        new_tiles = merge_row(row)
        new_grid.append(new_tiles)
    return np.array(new_grid)

def neighbour_difference(grid, axis=None):
    '''
        This calculates the differences between neighbouring tiles.

        Example:
        >>> # our input...
        >>> grid = [[2,0,2,0]
                    [0,0,4,0]
                    [4,4,0,0]
                    [2,8,0,2]]
        >>> neig_diff = neighbour_difference(grid)
        >>> neig_diff
        >>> [2,-2, 2, 0,-4, 4, 0, 4, 0, -6, 8, -2]
    '''
    
    hor_diff = []
    vert_diff = []
    for row in grid:
        for i in range(3):
            hor_diff.append(row[i] - row[i + 1])
    
    for row in np.transpose(grid):
        for i in range(3):
            vert_diff.append(row[i] - row[i + 1])
    
    if axis == 0:
        return np.array(hor_diff)
    elif axis == 1:
        return np.array(vert_diff)
    else:
        return np.insert(hor_diff, len(hor_diff), vert_diff)
