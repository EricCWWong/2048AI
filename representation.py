import numpy as np


def relationship_row(row):
    diff = []
    for i, el in enumerate(row):
        for j, other in enumerate(row):
            if i != j:
                diff.append(el - other)
    return np.array(diff)

def relationship(grid):
    diff_grid = []
    for row in grid:
        diff_grid.append(relationship_row(row)) 
    diff_grid = np.array(diff_grid)
    diff_grid = diff_grid.flatten()
    return diff_grid

def relationship_representation(grid):

    state_action_mat = []
    for i in range(4):
        if i == 0:
            relat = relationship(grid) * (-1)
            grid_vec = grid.flatten()
            state_vec = np.insert(grid_vec, len(grid_vec), relat)
        elif i == 1:
            relat = relationship(grid)
            grid_vec = grid.flatten()
            state_vec = np.insert(grid_vec, len(grid_vec), relat)
        elif i == 2:
            relat = relationship(np.transpose(grid)) * (-1)
            grid_vec = grid.flatten()
            state_vec = np.insert(grid_vec, len(grid_vec), relat)
        else:
            relat = relationship(np.transpose(grid))
            grid_vec = grid.flatten()
            state_vec = np.insert(grid_vec, len(grid_vec), relat)

        state_action_vec = np.zeros(len(state_vec)*i)
        state_action_vec = np.insert(state_action_vec, len(state_action_vec), state_vec)
        state_action_vec = np.insert(state_action_vec, len(state_action_vec), np.zeros(len(state_vec)*(3-i)))
        state_action_vec = np.insert(state_action_vec, len(state_action_vec), np.array([1]))

        state_action_mat.append(state_action_vec)

    return np.array(state_action_mat)