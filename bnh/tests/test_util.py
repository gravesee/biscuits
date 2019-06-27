from bnh.tabulate import *
import unittest
import numpy as np

class TestTabulate(unittest.TestCase):

    def test_tabulate(self):
        uniq, table = tabulate(np.array([1,2]), np.array([0,1]))
        np.testing.assert_array_equal(uniq, np.array([[1],[2]]))
        np.testing.assert_array_equal(table, np.array([[1,0],[0,1]], dtype='float64'))