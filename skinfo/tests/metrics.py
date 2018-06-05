import unittest

import numpy as np

from skinfo import entropy


class TestEntropy(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n = np.random.normal(6, 2.5, 1000)
        np.random.seed(2409)
        self.g = np.random.gamma(32, 7, 1000)

    def test_normal_10bin(self):
        self.assertAlmostEqual(entropy(self.n, 10), 2.5497, places=4)

    def test_normal_100bin(self):
        self.assertAlmostEqual(entropy(self.n, 100), 5.7577, places=4)

    def test_normal_freedman(self):
        self.assertAlmostEqual(entropy(self.n, 'fd'), 3.9848, places=4)

    def test_normal_discrete(self):
        self.assertAlmostEqual(
            entropy(self.n, [-1, 2, 5, 10, 16]),
            1.3981, places=4
        )

    def test_gamma_10bin(self):
        self.assertAlmostEqual(entropy(self.g, 10), 2.6201, places=4)

    def test_gamma_100bin(self):
        self.assertAlmostEqual(entropy(self.g, 100), 5.8359, places=4)

    def test_gamma_freedman(self):
        self.assertAlmostEqual(entropy(self.g, 'fd'), 3.9026, places=4)

    def test_gamma_discrete(self):
        with self.assertRaises(ValueError) as e:
            entropy(self.g, [-1, 2, 5, 10, 16])
            self.assertEqual(
                str(e),
                'The histogram cannot be empty. Adjust the bins to fit the data'
            )


if __name__ == '__main__':
    unittest.main()
