from class_2048 import Game_2048
from epsilon_greedy import decay_epsilon_greedy, epsilon_greedy
from learning import q_learning_2048, q_learning_sa_2048
import numpy as np
from plotter import scatter_plotter
from learnt import play_2048



def main():
    policy = epsilon_greedy

    print("Learning begins...")
    w, trace_max, score = q_learning_sa_2048(0.9, 0.0000001, 0.1, policy, 1000, 1000)
    print("Would you like to see the weigth vector? (Y/n)")
    yes_no = str(input())
    if yes_no == "Y" or yes_no == "y":
        print(w)
    elif yes_no == "N" or yes_no == "n":
        pass
    else:
        print("Wrong input!")
    
    print("Would you like to see the trace of the best game? (Y/n)")
    yes_no = str(input())
    if yes_no == "Y" or yes_no == "y":
        for el in trace_max:
            print(el)
    elif yes_no == "N" or yes_no == "n":
        pass
    else:
        print("Wrong input!")

    print("Would you like to play the game with decision made based on this vector? (Y/n)")
    yes_no = str(input())
    if yes_no == "Y" or yes_no == "y":
        trace_max, score = play_2048(w, 1000, 1000)
        for el in trace_max:
            print(el)
    elif yes_no == "N" or yes_no == "n":
        pass
    else:
        print("Wrong input!")
    print("See you next time!")



if __name__ == "__main__":
    main()
