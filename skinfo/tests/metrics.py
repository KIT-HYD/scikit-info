import unittest

import numpy as np

from skinfo import entropy, joint_entropy


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


class TestJointEntropy(unittest.TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.n = np.random.normal(4, 9, 1000)
        np.random.seed(5432)
        self.g = np.random.gamma(40, 12, 1000)

    def test_10bins(self):
        self.assertAlmostEqual(
            joint_entropy(self.n, self.g, 10),
            5.4051, places=4
        )

    def test_100bins(self):
        self.assertAlmostEqual(
            joint_entropy(self.n, self.g, 100),
            9.683, places=4
        )

    def test_discrete(self):
        self.assertAlmostEqual(
            joint_entropy(self.n, self.g, np.linspace(-20, 500, 30)),
            4.3187, places=4
        )

    def test_sparse(self):
        self.assertAlmostEqual(
            joint_entropy(self.n, self.g, 10000),
            9.9658, places=4
        )

    def test_assertion_error(self):
        with self.assertRaises(AssertionError):
            joint_entropy(self.n[1:], self.g, 10)


class TestConditionalEntropy(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
