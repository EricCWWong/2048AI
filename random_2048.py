import numpy as np
from class_2048 import Game_2048
import tqdm


def random_2048(num_episodes, max_steps=np.inf):
    '''
        This plays the game as a random agent. Decision
        are chosen randomly.
    '''

    # create a game.
    env = Game_2048()

    # initialise an array to store the maximum value of the tiles
    # and score.
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

            # removing unavailable action.
            allowed_action = np.sum(env.actions)
            probability = 1/allowed_action
            prob = np.zeros(4)

            for i, el in enumerate(env.actions):
                if el:
                    prob[i] = probability

            # choose an action with this policy.
            action = int(np.random.choice(np.arange(4), p=prob))

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

if __name__ == "__main__":
    trace_max, score = random_2048(1000)
    for el in trace_max:
        print(el)
    print(np.max(np.array(score)))