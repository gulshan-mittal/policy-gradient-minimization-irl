"""
Run linear reward policy gradient inverse reinforcement learning on the gridworld.

"""

import numpy as np
import matplotlib.pyplot as plt

import irl.pgirl_linear as linear_irl
import irl.envs.gridworld as gridworld

def main(grid_size, discount):

    wind = 0.3
    trajectory_length = 3*grid_size

    gw = gridworld.Gridworld(grid_size, wind, discount)

    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    print(ground_r)
    policy = [gw.optimal_policy_deterministic(s) for s in range(gw.n_states)]
    r = linear_irl.irl(gw.n_states, gw.n_actions, gw.transition_probability, policy, gw.discount, 1, 5)
    print(r)
    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    # plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

if __name__ == '__main__':
    main(5, 0.2)
