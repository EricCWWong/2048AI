import numpy as np


'''
    This files creates the epsilon greedy policy.

    All function has to have consistent format:

    def policy(epsilon, q, num_episodes, episodes)
'''

def decay_epsilon_greedy(epsilon, q, num_episodes, episodes):
    '''
        This is a step-wise decay epsilon greedy policy generator.
    '''
    num_actions = len(q)
    percentage = episodes/num_episodes
    
    diff = epsilon/4
    if percentage > 0.97: 
        epsilon = epsilon - 4 * diff
    elif percentage > 0.9: 
        epsilon = epsilon - 3 * diff
    elif percentage > 0.8:
        epsilon = epsilon = 2 * diff
    elif percentage > 0.7:
        epsilon = epsilon = 1 * diff
    elif percentage > 0.6:
        epsilon = epsilon = 0.5 * diff

    policy = np.zeros(num_actions) + epsilon/num_actions
    max_q_ind = np.argmax(q)
    policy[max_q_ind] = 1 - epsilon + epsilon/num_actions
    return policy

def epsilon_greedy(epsilon, q, num_episodes, episodes):
    '''
        This is a fixed epsilon greedy policy generator.

        Note that num_episodes and episodes are not used.
    '''
    num_actions = len(q)
    policy = np.zeros(num_actions) + epsilon/num_actions
    max_q_ind = np.argmax(q)
    policy[max_q_ind] = 1 - epsilon + epsilon/num_actions
    return policy