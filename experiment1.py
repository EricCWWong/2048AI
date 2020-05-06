from class_2048 import Game_2048
from epsilon_greedy import decay_epsilon_greedy, epsilon_greedy
from learning import q_learning_2048
import numpy as np



def main():
    policy = epsilon_greedy

    print("Learning begins...")
    w, trace_max, score = q_learning_2048(0.9, 0.000005, 0.4, policy, 1000, 1000)

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

    print("See you next time!")

if __name__ == "__main__":
    main()
