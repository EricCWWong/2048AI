import numpy as np
from class_2048 import Game_2048
from epsilon_greedy import epsilon_greedy
import tqdm


def play_2048(weight, num_episodes, max_steps=np.inf):
    '''
        Play 2048 using a weight learnt from q-learning policy.
    '''

    # create a game.
    env = Game_2048()

    # initialise the weight vector for all action.
    w = weight

    # initialise an array to store the maximum value of the tiles.
    max_value = []
    score = []

    # run episodes.
    for i in tqdm.tqdm(range(num_episodes)):

        # record the trace for this episode.
        trace = []

        # we start a new game.
        env.new_game()

        # add the initial grid into the trace.
        trace.append(env.grid)

        # while we haven't lost yet...
        while env.terminal is False:            
            # calculate the q value for current state for all actions.
            q = np.sum(env.state_action*w, 1)

            # get a new q vector to eliminate the unavailable actions.
            Q_index = []
            for i, el in enumerate(env.actions):
                if el:
                    Q_index.append(np.array([q[i],i]))
            Q_index = np.array(Q_index)

            # choose an action with this policy.
            max_Q_index = np.argmax(Q_index[:,0])
            action = Q_index[max_Q_index][1]

            # make a move with this action.
            env.make_move(action)

            # add the new grid into the trace
            trace.append(env.grid)
        
        # record the maximum value achieved.
        max_value.append(env.max_value)
        score.append(env.score)

        # if we achieved the best record, we save the trace.
        if env.max_value >= np.max(max_value):
            trace_max = trace

    # report the frequency of each max tiles.
    unique, counts = np.unique(max_value, return_counts=True)
    max_count = dict(zip(unique, counts))

    print(max_count)
    return trace_max, score
