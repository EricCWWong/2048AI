import numpy as np


def merge_row(row):
    '''
        This function merges a given row element to the right.
        It is same as swiping right in the game 2048.
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
    '''
    tiles = grid
    new_grid = []
    for row in tiles:
        new_tiles = merge_row(row)
        new_grid.append(new_tiles)
    return np.array(new_grid)

def neighbour_difference(grid):
    '''
        This calculates the differences between neighbouring tiles.
    '''
    hor_diff = []
    vert_diff = []
    for row in grid:
        for i in range(3):
            hor_diff.append(row[i] - row[i + 1])
    
    for row in np.transpose(grid):
        for i in range(3):
            vert_diff.append(row[i] - row[i + 1])
    
    diff = hor_diff + vert_diff

    return np.array(diff)

def tiles_on_edge(grid):
    '''
        This counts the number of non-empty tiles at each edge of the grid.
    '''
    
    count = []
    count_top = 0
    for el in grid[0]:
        if el != 0:
            count_top += 1
    count.append(count_top)

    count_bot = 0
    for el in grid[3]:
        if el != 0:
            count_bot += 1
    count.append(count_bot)

    count_left = 0
    for el in grid[:,0]:
        if el != 0:
            count_left += 1
    count.append(count_left)

    count_right = 0
    for el in grid[:,0]:
        if el != 0:
            count_right += 1
    count.append(count_right)

    return np.array(count)