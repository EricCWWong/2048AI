from class_2048 import Game_2048
from epsilon_greedy import decay_epsilon_greedy, epsilon_greedy
from learning import q_learning_2048, q_learning_sa_2048
import numpy as np
from plotter import scatter_plotter
from learnt import play_2048



def main():
    policy = epsilon_greedy
    w, _, _ = q_learning_sa_2048(0.9, 0.0000001, 0.1, policy, 1000, 1000)
    trace_max, score = play_2048(w, 1000, 1000)

    # for el in trace_max:
    #     print(el)


if __name__ == "__main__":
    main()
