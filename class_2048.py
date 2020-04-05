import numpy as np
import random
from class_2048_helper import merge, neighbour_difference
from reward_2048 import reward_merge, reward_align, reward_empty
from representation import relationship_representation

class Game_2048:
    def __init__(self):
        # initialise the game attributes.
        self.new_game()
        self.max_value = self.max()

        # initialise allowed moves.
        self.set_allowed_moves()

    def new_game(self):
        '''
            This will create a new game.
        '''
        # randomly generating a new grid.
        grid = np.zeros((16))
        index = np.random.choice(16, 2, replace=False)
        grid[index[0]] = random.choice([2,4])
        grid[index[1]] = random.choice([2,4])
        grid = np.reshape(grid, (4,4))

        # resetting the attributes of the game.
        self.grid = grid
        self.terminal = False
        self.reward = 0
        self.set_state_vector()
        self.set_state_action_vector()

        # resetting the allowed moves of the game.
        self.set_allowed_moves()
        return grid

    def set_allowed_moves(self):
        '''
            This method will check what moves are allowed for the current board.
        '''
        # get the potential grid after a move.
        old = self.grid
        left = self.left_move()
        right = self.right_move()
        up = self.up_move()
        down = self.down_move()

        # if the potential move gives the same grid,
        # that move is not allowed.
        if np.array_equal(old, left):
            self.left = False
        else:
            self.left = True

        if np.array_equal(old, right):
            self.right = False
        else:
            self.right = True
        
        if np.array_equal(old, up):
            self.up = False
        else:
            self.up = True
        
        if np.array_equal(old, down):
            self.down = False
        else:
            self.down = True

        # put all the data into one array.
        self.actions = np.array([self.left, self.right, self.up, self.down])

    def set_state_vector(self):
        '''
            This will create our state feature vector.
            At the moment, we will represent our state as
            the grid flatten as a 1D array.
        '''

        # this part stores the value of each grid
        vec = self.grid.flatten()
        state_vec = []
        for el in vec:
            if el == 0:
                state_vec.append(el)
            else:
                state_vec.append(np.log2(el))
        
        self.state = np.array(state_vec)
        return self.state

    def set_state_action_vector(self):
        self.state_action = relationship_representation(self.grid)
        return self.state_action
    
    def max(self):
        '''
            This will find the maximum value on the baord.
        '''
        return np.max(self.grid)

    def right_move(self):
        '''
            Swipe right.
        '''
        grid = merge(self.grid)
        return grid
    
    def left_move(self):
        '''
            Swipe left.
        '''
        grid = np.flip(self.grid,1)
        grid = merge(grid)
        grid = np.flip(grid,1)
        return grid

    def down_move(self):
        '''
            Swipe down.
        '''
        grid = np.transpose(self.grid)
        grid = merge(grid)
        grid = np.transpose(grid)
        return grid

    def up_move(self):
        '''
            Swipe up.
        '''
        grid = np.transpose(self.grid)
        grid = np.flip(grid,1)
        grid = merge(grid)
        grid = np.flip(grid,1)
        grid = np.transpose(grid)
        return grid

    def is_terminal(self):
        '''
            Check if we have reached the terminal state.
        '''

        # if any moves are still allowed, we can continue to play the game.
        if np.any(self.actions):
            self.terminal = False
        # if no moves are allowed, we lost and we terminate the game.
        else:
            self.terminal = True

    def new_tile_generation(self):
        '''
            This generates new tiles on empty tiles.
        '''
        grid = self.grid.flatten()
        index = []
        for i, el in enumerate(grid):
            if el == 0:
                index.append(i)
        new_tile_index = random.choice(index)
        grid[new_tile_index] = random.choice([2,4])
        grid = np.reshape(grid, (4,4))
        return grid
        
    def make_move(self, move):     
        '''
            This allows us to make a move. It will calculate
            the new grid, and update all attributes.
        '''
        old = self.grid

        if move == 0 and self.actions[0]: # left
            grid = self.left_move()
        elif move == 1 and self.actions[1]: # right
            grid = self.right_move()
        elif move == 2 and self.actions[2]: # up
            grid = self.up_move()
        elif move == 3 and self.actions[3]: #down
            grid = self.down_move()
        else:
            print('incorrect input or move is not allowed')
            grid = self.grid
        
        self.grid = grid
        self.reward = reward_merge(old, self.grid)
        
        self.grid = self.new_tile_generation()

        #self.reward = self.reward + reward_align(self.grid)

        self.set_allowed_moves()   
        self.is_terminal()
        self.set_state_vector()
        self.set_state_action_vector()
        
        self.max_value = self.max()
            
            
            
    



if __name__ == "__main__":


    game1=Game_2048()
    
    left = 0
    right = 1
    up = 2
    down = 3

    game1.new_game()
    print(game1.grid)
    game1.make_move(up)
    print(game1.reward)
    print(game1.state_action)
    print(game1.grid)
    game1.make_move(right)
    print(game1.reward)
    print(game1.grid)
    game1.make_move(down)
    print(game1.reward)
    print(game1.grid)
    game1.make_move(left)
    print(game1.reward)
    print(game1.grid)
    game1.make_move(down)
    print(game1.reward)
    print(game1.grid)
