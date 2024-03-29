"""
Unit tests for the gridworld MDP.

"""

import unittest

import numpy as np
import numpy.random as rn

import gridworld


def make_random_gridworld():
    grid_size = rn.randint(2, 15)
    wind = rn.uniform(0.0, 1.0)
    discount = rn.uniform(0.0, 1.0)
    return gridworld.Gridworld(grid_size, wind, discount)


class TestTransitionProbability(unittest.TestCase):

    def test_sums_to_one(self):
        # This is a simple fuzz-test.
        for _ in range(40):
            gw = make_random_gridworld()
            self.assertTrue(
                np.isclose(gw.transition_probability.sum(axis=2), 1).all(),
                'Probabilities don\'t sum to 1: {}'.format(gw))

    def test_manual_sums_to_one(self):
        gw = gridworld.Gridworld(5, 0.3, 0.2)
        self.assertTrue(
            np.isclose(gw.transition_probability.sum(axis=2), 1).all())

if __name__ == '__main__':
    unittest.main()
