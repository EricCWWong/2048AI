from class_2048 import Game_2048
from epsilon_greedy import decay_epsilon_greedy, epsilon_greedy
from learning import q_learning_2048
import numpy as np



def main():
    policy = epsilon_greedy

    print("Would you like to manually set the parameters? (Y/n)")
    yes_no = str(input())
    if yes_no == "Y" or yes_no == "y":
        print("Please input learning rate, alpha: (float)")
        alpha = float(input())
        print("Please input gamma: (float)")
        gamma = float(input())
        print("Please input epsilon: (float)")
        epsilon = float(input())
        print("Please input number of episodes: (int)")
        num_epsiodes = int(input())
    elif yes_no == "N" or yes_no == "n":
        gamma = 0.9
        alpha = 0.000005
        epsilon = 0.4
        num_epsiodes = 1000
    else:
        raise ValueError('Wrong input!')

    print("Learning begins...")
    w, trace_max, score = q_learning_2048(gamma, alpha, epsilon, policy, num_epsiodes, 1000)

    print("Would you like to see the weigth vector? (Y/n)")
    yes_no = str(input())
    if yes_no == "Y" or yes_no == "y":
        print(w)
    elif yes_no == "N" or yes_no == "n":
        pass
    else:
        raise ValueError('Wrong input!')
    
    print("Would you like to see the trace of the best game? (Y/n)")
    yes_no = str(input())
    if yes_no == "Y" or yes_no == "y":
        for el in trace_max:
            print(el)
    elif yes_no == "N" or yes_no == "n":
        pass
    else:
        raise ValueError('Wrong input!')

    print("See you next time!")

if __name__ == "__main__":
    main()
