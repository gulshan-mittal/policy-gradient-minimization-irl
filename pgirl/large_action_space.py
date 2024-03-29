
import numpy as np
import matplotlib.pyplot as plt

import irl.pgirl_linear as linear_irl
import irl.envs.gridworld as gridworld
from irl.cal_val import value

def main(grid_size, discount):
    

    wind = 0.3
    trajectory_length = 3*grid_size

    gw = gridworld.Gridworld(grid_size, wind, discount)

    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    policy = [gw.optimal_policy_deterministic(s) for s in range(gw.n_states)]

    # Need a value function for each basis function.
    feature_matrix = gw.feature_matrix()
    values = []
    for dim in range(feature_matrix.shape[1]):
        reward = feature_matrix[:, dim]
        values.append(value(policy, gw.n_states, gw.transition_probability,
                            reward, gw.discount))
    values = np.array(values)

    r = linear_irl.large_irl(values, gw.transition_probability,
                        feature_matrix, gw.n_states, gw.n_actions, policy)

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    # plt.colorbar()
    plt.title("Recovered reward")
    plt.xlabel("Grid Size = 10")
    plt.show()

if __name__ == '__main__':
    main(10, 0.9)
