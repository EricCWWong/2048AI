from class_2048 import Game_2048
from epsilon_greedy import decay_epsilon_greedy, epsilon_greedy
from learning import q_learning_2048
import numpy as np



def main():
    policy = decay_epsilon_greedy
    w, trace_max, score = q_learning_2048(0.9, 0.000005, 0.4, policy, 1000, 1000)
    print(w)

    for el in trace_max:
        print(el)

    print(np.max(np.array(score)))

if __name__ == "__main__":
    main()
