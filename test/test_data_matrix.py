import unittest

import numpy as np

import funcsfa


class TestRoundtrip(unittest.TestCase):

    def setUp(self):
        self.rand = np.random.RandomState(1273641113)
        self.n_factors = 9
        self.f = funcsfa.SFA()
        self.n_samples = 514
        samples = [f'n{i}' for i in range(self.n_samples)]
        self.n_features = 107+49
        f1 = [f'f1_{i}' for i in range(107)]
        f2 = [f'f2_{i}' for i in range(49)]
        self.X_a = self.rand.normal(0, 1, (self.n_samples, 107))
        self.Xw_a = self.rand.normal(0, 1, (self.n_samples, 107))
        self.X_b = self.rand.normal(0, 1, (self.n_samples, 49))
        self.Xw_b = self.rand.normal(0, 1, (self.n_samples, 49))
        self.data = funcsfa.StackedDataMatrix([
            funcsfa.DataMatrix(self.X_a, samples, f1, self.Xw_a),
            funcsfa.DataMatrix(self.X_b, samples, f2, self.Xw_b)],
            ['a', 'b'],
        )
        self.data.to_netcdf('test_roundtrip.nc')
        self.data2 = funcsfa.StackedDataMatrix.from_netcdf('test_roundtrip.nc')

    def test_same_data(self):
        np.testing.assert_allclose(self.data.data, self.data2.data)

    def test_same_weighted_data(self):
        np.testing.assert_allclose(self.data.dataW, self.data2.dataW)

    def test_same_samples(self):
        assert self.data.samples == self.data2.samples

    def test_same_features(self):
        assert self.data.features == self.data2.features

    def test_size(self):
        assert self.data.dt_n_features == self.data2.dt_n_features
