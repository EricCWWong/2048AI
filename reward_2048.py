import numpy as np
from class_2048_helper import neighbour_difference


def reward_merge(old, new, max_tile=2048):
    '''
        This reward scheme gives reward by the number of new merged tiles.
    '''
    # initialise reward.
    reward = 0

    # flatten the vector.
    old = old.flatten()
    new = new.flatten()

    # create a dictonary that will store the values of the tiles.
    old_dict = {}
    old_dict[0] = 0
    new_dict = {}
    new_dict[0] = 0

    for i in range(int(np.log2(max_tile))):
        old_dict[int(2**(i+1))] = 0
        new_dict[int(2**(i+1))] = 0

    # counting the number of each values.
    for el in old:
        old_dict[int(el)] = old_dict[int(el)] + 1

    for el in new:
        new_dict[int(el)] += 1

    # now we check the difference between grids.
    diff = {}

    for i in range(int(np.log2(max_tile))):
        difference = new_dict[int(2**(i+1))] - old_dict[int(2**(i+1))]
        if difference > 0:
            diff[int(2**(i+1))] = difference
        else:
            diff[int(2**(i+1))] = 0
    
    # now we calculate the rewards, we only care about 4 or above.
    for key in diff:
        if key > 2:
            reward = reward + key * diff[key]

    return reward

def reward_align(grid):
    diff = neighbour_difference(grid)

    freq_zeros = 0

    for el in diff:
        if int(el) == 0:
            freq_zeros += 1

    return freq_zeros

def reward_empty(grid):
    count = 0
    for el in grid.flatten():
        if el == 0:
            count = count + 1

    return count

def reward_corner(state):
    weighted_sum = 0
    for i, el in enumerate(state):
        if i == 0 or i == 3 or i == 12 or i == 15:
            weighted_sum = weighted_sum + el * 10
        elif i == 5 or i == 6 or i == 9 or i == 10:
            weighted_sum = weighted_sum + el
        else:
            weighted_sum = weighted_sum + el * 5 

    return weighted_sum