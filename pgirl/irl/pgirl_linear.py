
import random

import numpy as np
from cvxopt import matrix, solvers

def irl(n_states, n_actions, transition_probability, policy, discount, Rmax, l1):
   
    A = set(range(n_actions))
    transition_probability = np.transpose(transition_probability, (1, 0, 2))

    def T(a, s):
        return np.dot(transition_probability[policy[s], s] -
                      transition_probability[a, s],
                      np.linalg.inv(np.eye(n_states) -
                        discount*transition_probability[policy[s]]))


    c = -np.hstack([np.zeros(n_states), np.ones(n_states),
                    -l1*np.ones(n_states)])
    zero_stack1 = np.zeros((n_states*(n_actions-1), n_states))
    T_stack = np.vstack([
        -T(a, s)
        for s in range(n_states)
        for a in A - {policy[s]}
    ])
    I_stack1 = np.vstack([
        np.eye(1, n_states, s)
        for s in range(n_states)
        for a in A - {policy[s]}
    ])
    I_stack2 = np.eye(n_states)
    zero_stack2 = np.zeros((n_states, n_states))

    D_left = np.vstack([T_stack, T_stack, -I_stack2, I_stack2])
    D_middle = np.vstack([I_stack1, zero_stack1, zero_stack2, zero_stack2])
    D_right = np.vstack([zero_stack1, zero_stack1, -I_stack2, -I_stack2])

    D = np.hstack([D_left, D_middle, D_right])
    b = np.zeros((n_states*(n_actions-1)*2 + 2*n_states, 1))
    bounds = np.array([(None, None)]*2*n_states + [(-Rmax, Rmax)]*n_states)

    D_bounds = np.hstack([
        np.vstack([
            -np.eye(n_states),
            np.eye(n_states)]),
        np.vstack([
            np.zeros((n_states, n_states)),
            np.zeros((n_states, n_states))]),
        np.vstack([
            np.zeros((n_states, n_states)),
            np.zeros((n_states, n_states))])])
    b_bounds = np.vstack([Rmax*np.ones((n_states, 1))]*2)
    D = np.vstack((D, D_bounds))
    b = np.vstack((b, b_bounds))
    A_ub = matrix(D)
    b = matrix(b)
    c = matrix(c)
    results = solvers.lp(c, A_ub, b)
    r = np.asarray(results["x"][:n_states], dtype=np.double)

    return r.reshape((n_states,))

def v_tensor(value, transition_probability, feature_dimension, n_states,
             n_actions, policy):
    """
    Finds the v tensor used in large linear IRL.

    feature_dimension: Dimension of the feature matrix. int.
    n_states: Number of states sampled. int.
    n_actions: Number of actions. int.
    policy: NumPy array mapping state ints to action ints.
    -> v helper tensor.
    """

    v = np.zeros((n_states, n_actions-1, feature_dimension))
    for i in range(n_states):
        a1 = policy[i]
        exp_on_policy = np.dot(transition_probability[i, a1], value.T)
        seen_policy_action = False
        for j in range(n_actions):
            if a1 == j:
                seen_policy_action = True
                continue

            exp_off_policy = np.dot(transition_probability[i, j], value.T)
            if seen_policy_action:
                v[i, j-1] = exp_on_policy - exp_off_policy
            else:
                v[i, j] = exp_on_policy - exp_off_policy
    return v

def large_irl(value, transition_probability, feature_matrix, n_states,
              n_actions, policy):
    """
    Find the reward in a large state space.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state. 
    n_states: Number of states sampled. int.
    n_actions: Number of actions. int.
    policy: NumPy array mapping state ints to action ints.
    -> Reward for each state in states.
    """

    D = feature_matrix.shape[1]

    v = v_tensor(value, transition_probability, D, n_states, n_actions, policy)


    x_size = n_states + (n_actions-1)*n_states*2 + D

    # c is a big stack of ones and zeros; there's N ones and the rest is zero.
    c = -np.hstack([np.ones(n_states), np.zeros(x_size - n_states)])
    assert c.shape[0] == x_size

    A = np.hstack([
        np.zeros((n_states*(n_actions-1), n_states)),
        np.eye(n_states*(n_actions-1)),
        -np.eye(n_states*(n_actions-1)),
        np.vstack([v[i, j].T for i in range(n_states)
                             for j in range(n_actions-1)])])
    assert A.shape[1] == x_size

    # b is just zeros!
    b = np.zeros(A.shape[0])

    # Break G up into the bottom row and other rows to construct it.
    bottom_row = np.vstack([
                    np.hstack([
                        np.ones((n_actions-1, 1)).dot(np.eye(1, n_states, l)),
                        np.hstack([-np.eye(n_actions-1) if i == l
                                   else np.zeros((n_actions-1, n_actions-1))
                         for i in range(n_states)]),
                        np.hstack([2*np.eye(n_actions-1) if i == l
                                   else np.zeros((n_actions-1, n_actions-1))
                         for i in range(n_states)]),
                        np.zeros((n_actions-1, D))])
                    for l in range(n_states)])
    assert bottom_row.shape[1] == x_size
    G = np.vstack([
            np.hstack([
                np.zeros((D, n_states)),
                np.zeros((D, n_states*(n_actions-1))),
                np.zeros((D, n_states*(n_actions-1))),
                np.eye(D)]),
            np.hstack([
                np.zeros((D, n_states)),
                np.zeros((D, n_states*(n_actions-1))),
                np.zeros((D, n_states*(n_actions-1))),
                -np.eye(D)]),
            np.hstack([
                np.zeros((n_states*(n_actions-1), n_states)),
                -np.eye(n_states*(n_actions-1)),
                np.zeros((n_states*(n_actions-1), n_states*(n_actions-1))),
                np.zeros((n_states*(n_actions-1), D))]),
            np.hstack([
                np.zeros((n_states*(n_actions-1), n_states)),
                np.zeros((n_states*(n_actions-1), n_states*(n_actions-1))),
                -np.eye(n_states*(n_actions-1)),
                np.zeros((n_states*(n_actions-1), D))]),
            bottom_row])
    assert G.shape[1] == x_size

    h = np.vstack([np.ones((D*2, 1)), np.zeros((n_states*(n_actions-1)*2+bottom_row.shape[0], 1))])

    from cvxopt import matrix, solvers
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    results = solvers.lp(c, G, h, A, b)
    alpha = np.asarray(results["x"][-D:], dtype=np.double)
    return np.dot(feature_matrix, -alpha)
