import numpy as np
from class_2048 import Game_2048
import tqdm

def q_learning_2048(gamma, alpha, epsilon, policy, num_episodes, max_steps=np.inf):
    '''
        q-learning algorithm for 2048
    '''

    # create a game.
    env = Game_2048()

    # initialise weight vector for a single action.
    w_a = np.zeros(len(env.state))

    # initialise the weight vector for all action.
    w = []
    for _ in range(4):
        w.append(w_a)
    w = np.array(w)

    # initialise an array to store the maximum value of the tiles.
    max_value = []

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
            q = np.sum(env.state*w, 1)

            # get a new q vector to eliminate the unavailable actions.
            Q_index = []
            for i, el in enumerate(env.actions):
                if el:
                    Q_index.append(np.array([q[i],i]))
            Q_index = np.array(Q_index)

            # choose a policy with this new q.
            prob = policy(epsilon, Q_index[:,0], num_episodes, i)

            # choose an action with this policy.
            action = int(np.random.choice(Q_index[:,1], p=prob))

            # store the current state.
            state = env.state

            # make a move with this action.
            env.make_move(action)

            # calculate the q-value for the next state.
            q_next = np.sum(env.state*w, 1)

            # calculate the delta correction for our weight.
            delta = alpha * (env.reward + gamma * np.max(q_next) - q[action]) * state
            
            # update our weight
            w[action] = w[action] + delta

            # add the new grid into the trace
            trace.append(env.grid)
        
        # record the maximum value achieved.
        max_value.append(env.max_value)

        # if we achieved the best record, we save the trace.
        if env.max_value == np.max(max_value):
            trace_max = trace

    # report the frequency of each max tiles.
    unique, counts = np.unique(max_value, return_counts=True)
    max_count = dict(zip(unique, counts))

    print(max_count)
    return w, trace_max
