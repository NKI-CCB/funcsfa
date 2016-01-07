import unittest

import numpy as np

from sfamd._sfamd import Factorization, np_size_t


class TestTwoDataTypes(unittest.TestCase):

    def setUp(self):
        self.n_features_split = [1000, 1243]
        self.n_features_split = np.asarray(self.n_features_split, np_size_t)
        self.n_features = np.sum(self.n_features_split)
        self.n_samples = 121
        self.n_factors = 10
        data_shape = (self.n_samples, self.n_features)
        self.data = dict()
        self.data['zeros'] = np.zeros(data_shape)
        self.data['random'] = np.random.normal(0, 1, data_shape)
        self.Z = np.random.normal(0, 1, (self.n_factors, self.n_samples))
        self.B = np.random.normal(0, .5, (self.n_features, self.n_factors))
        X = np.dot(self.B, self.Z).T
        X = X - np.mean(X, 0, keepdims=True)
        self.data['factorized'] = X
        self.data = {n: np.require(d, np.dtype('float64'), 'FA') for
                     n, d in self.data.items()}

    def check_factorization_type_and_shape(self, f):
        self.assertEqual(f.n_datatypes, len(self.n_features_split))
        coefficients = np.asarray(f.coefficients)
        self.assertEqual(coefficients.shape,
                         (self.n_features, self.n_factors))
        self.assertEqual(coefficients.dtype, np.dtype('float64'))
        factors = np.asarray(f.factors)
        self.assertEqual(factors.dtype, np.dtype('float64'))
        self.assertEqual(factors.shape,
                         (self.n_factors, self.n_samples))
        factors_cov = np.asarray(f.factors_cov)
        self.assertEqual(factors_cov.dtype, np.dtype('float64'))
        self.assertEqual(factors_cov.shape,
                         (self.n_factors, self.n_factors))
        residual_var = np.asarray(f.residual_var)
        self.assertEqual(residual_var.dtype, np.dtype('float64'))
        self.assertEqual(residual_var.shape,
                         (self.n_features,))

    def test_init(self):
        for d_name, d in self.data.items():
            with(self.subTest(data_name=d_name)):
                f = Factorization(d, self.n_features_split, self.n_factors)
                self.assertEqual(np.byte_bounds(d), np.byte_bounds(f.data))
                self.check_factorization_type_and_shape(f)

    def test_factorize(self):
        l = np.array([0.01, 1, 0.01, 1], np.dtype('float64'))
        max_iter = 10
        max_diff = 1e-6
        for d_name, d in self.data.items():
            with(self.subTest(data_name=d_name)):
                f = Factorization(d, self.n_features_split, self.n_factors)
                n_iter, diff = f.sfa(max_diff, max_iter, l)
                assert(n_iter < max_iter+1)
                assert(diff >= 0.0)
                if (diff > max_diff):
                    self.assertEqual(n_iter, max_iter)
                self.assertEqual(np.byte_bounds(d), np.byte_bounds(f.data))
                self.check_factorization_type_and_shape(f)

    def test_factorize_max_iter(self):
        l = np.array([0.01, 1, 0.01, 1], np.dtype('float64'))
        max_iter = 10
        max_diff = 0.0
        for d_name, d in self.data.items():
            with(self.subTest(data_name=d_name)):
                f = Factorization(d, self.n_features_split, self.n_factors)
                n_iter, diff = f.sfa(max_diff, max_iter, l)
                assert(n_iter < max_iter+1)
                assert(diff >= 0.0)
                if diff == 0.0:
                    assert d_name == 'zeros'
                if (diff > max_diff):
                    self.assertEqual(n_iter, max_iter)
                self.assertEqual(np.byte_bounds(d), np.byte_bounds(f.data))
                self.check_factorization_type_and_shape(f)

    def test_factorize_gets_close(self):
        l2 = 0.1
        l = np.array([0.01, l2, 0.02, l2], np.dtype('float64'))
        max_iter = 1000
        max_diff = 1e-6
        d_name = 'factorized'
        d = self.data[d_name]
        f = Factorization(d, self.n_features_split, self.n_factors)
        n_iter, diff = f.sfa(max_diff, max_iter, l)
        self.assertEqual(np.byte_bounds(d), np.byte_bounds(f.data))
        self.check_factorization_type_and_shape(f)
        coef = f.coefficients * (1+l2)
        data_rec = np.dot(coef, f.factors).T
        np.testing.assert_allclose(d, data_rec, atol=1)

    def test_factorize_converges(self):
        l = np.array([1e-12, 1e-3, 1e-12, 1e-3], np.dtype('float64'))
        max_iter = 1000
        max_diff = 0.1
        d_name = 'factorized'
        d = self.data[d_name]
        f = Factorization(d, self.n_features_split, self.n_factors)
        n_iter, diff = f.sfa(max_diff, max_iter, l)
        assert(n_iter < max_iter)
        assert(diff < max_diff)
        self.assertEqual(np.byte_bounds(d), np.byte_bounds(f.data))
        self.check_factorization_type_and_shape(f)


class TestThreeDataTypes(unittest.TestCase):

    def setUp(self):
        self.n_features_split = [1000, 800, 243]
        self.n_features_split = np.asarray(self.n_features_split, np_size_t)
        self.n_features = np.sum(self.n_features_split)
        self.n_samples = 121
        self.n_factors = 10
        data_shape = (self.n_samples, self.n_features)
        self.data = dict()
        self.data['zeros'] = np.zeros(data_shape)
        self.data['random'] = np.random.normal(0, 1, data_shape)
        self.Z = np.random.normal(0, 1, (self.n_factors, self.n_samples))
        self.B = np.zeros((self.n_features, self.n_factors))
        self.B[0:1000, :] = np.random.normal(0, 0.5, (1000, self.n_factors))
        self.B[1000:1800, :] = np.random.normal(0, 2, (800, self.n_factors))
        self.B[1800:, :] = np.random.normal(0, 0.01, (243, self.n_factors))
        X = np.dot(self.B, self.Z).T
        X = X - np.mean(X, 0, keepdims=True)
        self.data['factorized'] = X
        self.data = {n: np.require(d, np.dtype('float64'), 'FA') for
                     n, d in self.data.items()}

    def check_factorization_type_and_shape(self, f):
        self.assertEqual(f.n_datatypes, len(self.n_features_split))
        coefficients = np.asarray(f.coefficients)
        self.assertEqual(coefficients.shape,
                         (self.n_features, self.n_factors))
        self.assertEqual(coefficients.dtype, np.dtype('float64'))
        factors = np.asarray(f.factors)
        self.assertEqual(factors.dtype, np.dtype('float64'))
        self.assertEqual(factors.shape,
                         (self.n_factors, self.n_samples))
        factors_cov = np.asarray(f.factors_cov)
        self.assertEqual(factors_cov.dtype, np.dtype('float64'))
        self.assertEqual(factors_cov.shape,
                         (self.n_factors, self.n_factors))
        residual_var = np.asarray(f.residual_var)
        self.assertEqual(residual_var.dtype, np.dtype('float64'))
        self.assertEqual(residual_var.shape,
                         (self.n_features,))

    def test_init(self):
        for d_name, d in self.data.items():
            with(self.subTest(data_name=d_name)):
                f = Factorization(d, self.n_features_split, self.n_factors)
                self.assertEqual(np.byte_bounds(d), np.byte_bounds(f.data))
                self.check_factorization_type_and_shape(f)

    def test_factorize(self):
        l = np.array([0.01, 1] * len(self.n_features_split),
                     np.dtype('float64'))
        max_iter = 10
        max_diff = 1e-6
        for d_name, d in self.data.items():
            with(self.subTest(data_name=d_name)):
                f = Factorization(d, self.n_features_split, self.n_factors)
                n_iter, diff = f.sfa(max_diff, max_iter, l)
                assert(n_iter < max_iter+1)
                assert(diff >= 0.0)
                if (diff > max_diff):
                    self.assertEqual(n_iter, max_iter)
                self.assertEqual(np.byte_bounds(d), np.byte_bounds(f.data))
                self.check_factorization_type_and_shape(f)

    def test_factorize_max_iter(self):
        l = np.array([0.01, 1] * len(self.n_features_split),
                     np.dtype('float64'))
        max_iter = 10
        max_diff = 0.0
        for d_name, d in self.data.items():
            with(self.subTest(data_name=d_name)):
                f = Factorization(d, self.n_features_split, self.n_factors)
                n_iter, diff = f.sfa(max_diff, max_iter, l)
                assert(n_iter < max_iter+1)
                assert(diff >= 0.0)
                if diff == 0.0:
                    assert d_name == 'zeros'
                if (diff > max_diff):
                    self.assertEqual(n_iter, max_iter)
                self.assertEqual(np.byte_bounds(d), np.byte_bounds(f.data))
                self.check_factorization_type_and_shape(f)

    def test_factorize_gets_close(self):
        l2 = 0.1
        l = np.array([0.01, l2, 0.02, l2, 1e-6], np.dtype('float64'))
        max_iter = 1000
        max_diff = 1e-6
        d_name = 'factorized'
        d = self.data[d_name]
        f = Factorization(d, self.n_features_split, self.n_factors)
        n_iter, diff = f.sfa(max_diff, max_iter, l)
        self.assertEqual(np.byte_bounds(d), np.byte_bounds(f.data))
        self.check_factorization_type_and_shape(f)
        coef = f.coefficients * (1+l2)
        data_rec = np.dot(coef, f.factors).T
        np.testing.assert_allclose(d, data_rec, atol=1)

    def test_factorize_converges(self):
        l = np.array([1e-12, 1e-3] * len(self.n_features_split),
                     np.dtype('float64'))
        max_iter = 1000
        max_diff = 0.1
        d_name = 'factorized'
        d = self.data[d_name]
        f = Factorization(d, self.n_features_split, self.n_factors)
        n_iter, diff = f.sfa(max_diff, max_iter, l)
        assert(n_iter < max_iter)
        assert(diff < max_diff)
        self.assertEqual(np.byte_bounds(d), np.byte_bounds(f.data))
        self.check_factorization_type_and_shape(f)
