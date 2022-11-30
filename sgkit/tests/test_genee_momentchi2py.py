# hbe tests are from https://github.com/deanbodenham/momentchi2py/blob/master/tests/test_momentchi2.py

import unittest

import numpy as np

from sgkit.stats.genee_momentchi2py import hbe


class HbeTests(unittest.TestCase):
    def test_hbe1(self):
        """hbe with x float, coeff list"""
        x = 10.203
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(coeff, x)
        soln = 0.949
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)

    def test_hbe2(self):
        """hbe with x float, coeff list, specifying arguments"""
        x = 10.203
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(x=x, coeff=coeff)
        soln = 0.949
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)

    def test_hbe3(self):
        """hbe with x float, coeff list, specifying arguments"""
        x = 0.627
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(x=x, coeff=coeff)
        soln = 0.0285
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)

    def test_hbe4(self):
        """hbe with x list, coeff list, specifying arguments"""
        x = [0.627, 10.203]
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(coeff=coeff, x=x)
        soln = [0.0285, 0.949]
        # check it is a list
        self.assertTrue(isinstance(ans, list))
        # check lists are equal length and almost equal values
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=2, msg=None, delta=None)

    def test_hbe5(self):
        """hbe with x float, coeff numpy array"""
        x = 10.203
        coeff = np.array([1.5, 1.5, 0.5, 0.5])
        ans = hbe(coeff, x)
        soln = 0.949
        self.assertAlmostEqual(ans, soln, places=3, msg=None, delta=None)

    def test_hbe6(self):
        """hbe with x numpy array, coeff numpy array"""
        x = np.array([0.627, 10.203])
        coeff = np.array([1.5, 1.5, 0.5, 0.5])
        ans = hbe(coeff, x)
        soln = np.array([0.0285, 0.949])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=3, msg=None, delta=None)

    def test_hbe7(self):
        """hbe with x numpy array one element, coeff numpy array"""
        x = np.array([0.627])
        coeff = np.array([1.5, 1.5, 0.5, 0.5])
        ans = hbe(coeff, x)
        soln = np.array([0.0285])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=3, msg=None, delta=None)

    def test_hbe8(self):
        """hbe with x numpy array, coeff list"""
        x = np.array([0.627, 10.203])
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(coeff, x)
        soln = np.array([0.0285, 0.949])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=3, msg=None, delta=None)

    def test_hbe9(self):
        """hbe with x list, coeff numpy array"""
        x = np.array([0.627, 10.203])
        coeff = [1.5, 1.5, 0.5, 0.5]
        ans = hbe(coeff, x)
        soln = np.array([0.0285, 0.949])
        self.assertEqual(len(ans), len(soln))
        for i in range(len(ans)):
            self.assertAlmostEqual(ans[i], soln[i], places=3, msg=None, delta=None)
