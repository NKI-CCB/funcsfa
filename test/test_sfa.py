import unittest

import numpy as np

from . import plot
import funcsfa


class TestInvalidInputs(unittest.TestCase):

    def setUp(self):
        self.rand = np.random.RandomState(1968486074)
        self.n_factors = 9
        self.f = funcsfa.SFA()
        self.n_samples = 221
        self.n_features = 37
        self.X_a = self.rand.normal(0, 1, (self.n_samples, 30))
        self.X_b = self.rand.normal(0, 1, (self.n_samples, 7))
        self.data_one = funcsfa.DataMatrix(self.X_a)
        self.data_two = funcsfa.StackedDataMatrix([
            funcsfa.DataMatrix(self.X_a),
            funcsfa.DataMatrix(self.X_b)])

    def test_l1_penalty_length_one_dt(self):
        self.f.fit(self.data_one, self.n_factors, max_iter=0, l1=0.0)
        self.f.fit(self.data_one, self.n_factors, max_iter=0, l1=0.0, l2=0.0)
        self.f.fit(self.data_one, self.n_factors, max_iter=0, l1=[0.0], l2=0.0)
        self.f.fit(self.data_one, self.n_factors, max_iter=0, l1=[0.0],
                   l2=[0.0])
        with self.assertRaises(Exception):
            self.f.fit(self.data_one, self.n_factors, max_iter=0,
                       l1=[0.0, 0.1])
        with self.assertRaises(Exception):
            self.f.fit(self.data_one, self.n_factors, max_iter=0,
                       l1=[])
        with self.assertRaises(Exception):
            self.f.fit(self.data_one, self.n_factors, max_iter=0,
                       l1=[0.1, 0.2], l2=[0.1])
        with self.assertRaises(Exception):
            self.f.fit(self.data_one, self.n_factors, max_iter=0,
                       l1=[0.1, 0.2], l2=0.1)

    def test_l2_penalty_length_one_dt(self):
        self.f.fit(self.data_one, self.n_factors, max_iter=0, l1=0.0, l2=[0.0])
        with self.assertRaises(Exception):
            self.f.fit(self.data_one, self.n_factors, max_iter=0,
                       l2=[0.0, 0.1])
        with self.assertRaises(Exception):
            self.f.fit(self.data_one, self.n_factors, max_iter=0,
                       l2=[])
        with self.assertRaises(Exception):
            self.f.fit(self.data_one, self.n_factors, max_iter=0,
                       l2=[0.1, 0.2], l1=[0.1])
        with self.assertRaises(Exception):
            self.f.fit(self.data_one, self.n_factors, max_iter=0,
                       l2=[0.1, 0.2], l1=0.1)

    def test_more_factors_than_features(self):
        with self.assertRaises(Exception):
            self.f.fit(self.data_two, self.data_two.dt_n_features[0]+1,
                       max_iter=0)
        with self.assertRaises(Exception):
            self.f.fit(self.data_two, self.data_two.dt_n_features[1]+1,
                       max_iter=0)

    def test_invalid_transform(self):
        f = funcsfa.SFA()

        with self.assertRaises(Exception):
            f.transform(self.data_one)

        f.fit(self.data_one, self.n_factors, max_iter=10)
        f.transform(self.data_one)
        f.transform(self.data_one)
        with self.assertRaises(Exception):
            f.transform(self.data_two)
        with self.assertRaises(Exception):
            f.transform(self.data_two.data)


class TestSingleDatatypeReproduceRandom(unittest.TestCase):

    def setUp(self):
        self.n_samples = 400
        self.n_features = 2000
        self.n_factors = 10
        self.rand = np.random.RandomState(1968486074)
        self.B = self.rand.normal(0, 1, (self.n_features, self.n_factors))
        Zvar = np.linspace(10, 1, self.n_factors)
        Zvar = Zvar / np.mean(Zvar)
        self.Z = self.rand.normal(0, np.sqrt(Zvar),
                                  (self.n_samples, self.n_factors))
        self.X = np.dot(self.Z, self.B.T)
        self.data = funcsfa.DataMatrix(self.X)
        self.f = funcsfa.SFA()

    def test_init_full_factors_output_shapes(self):
        Z_estimated = self.f.fit_transform(self.data, self.n_factors,
                                           max_iter=0)
        self.assertEqual(Z_estimated.shape, (self.n_samples, self.n_factors))
        self.assertEqual(self.f.coefficients[0].shape,
                         (self.n_features, self.n_factors))
        assert not np.any(np.isnan(Z_estimated))
        assert not np.any(np.isnan(self.f.coefficients[0]))

    def test_init_full_factors_reconstruction(self):
        Z_estimated = self.f.fit_transform(self.data, self.n_factors,
                                           max_iter=0)
        self.assertAlmostEqual(self.f.reconstruction_error, 0.0)
        X_reconstructed = (np.dot(Z_estimated, self.f.coefficients[0].T) +
                           np.mean(self.X, 0, keepdims=True))
        np.testing.assert_allclose(self.X, X_reconstructed)

    def test_init_full_factors_constraints(self):
        Z_estimated = self.f.fit_transform(self.data, self.n_factors,
                                           max_iter=0)
        np.testing.assert_allclose(1, np.mean(Z_estimated ** 2))

    def test_init_part_factors_output_shapes(self):
        n_factors = self.n_factors // 2
        Z_estimated = self.f.fit_transform(self.data, n_factors, max_iter=0)
        self.assertEqual(Z_estimated.shape, (self.n_samples, n_factors))
        self.assertEqual(self.f.coefficients[0].shape,
                         (self.n_features, n_factors))
        assert not np.any(np.isnan(Z_estimated))
        assert not np.any(np.isnan(self.f.coefficients[0]))

    def test_init_part_factors_constraints(self):
        n_factors = self.n_factors // 2
        Z_estimated = self.f.fit_transform(self.data, n_factors, max_iter=0)
        np.testing.assert_allclose(1, np.mean(Z_estimated ** 2))
        Z_estimated2 = self.f.transform(self.data)
        np.testing.assert_allclose(1, np.mean(Z_estimated2 ** 2), rtol=1e-2)

    def test_init_part_factors_reconstruction(self):
        n_factors = self.n_factors // 2
        max_error = np.sum(self.X ** 2) / self.n_samples
        Z_estimated = self.f.fit_transform(self.data, n_factors, max_iter=0)
        X_reconstructed = (np.dot(Z_estimated, self.f.coefficients[0].T) +
                           np.mean(self.X, 0, keepdims=True))
        err = np.sum((self.X - X_reconstructed) ** 2) / self.n_samples
        assert self.f.reconstruction_error / max_error < 0.5
        np.testing.assert_allclose(err, self.f.reconstruction_error)

    def test_init_fit_transform_consistent(self):
        self.f.fit_transform(self.data, self.n_factors, max_iter=0)
        coef1 = self.f.coefficients
        self.f.fit(self.data, self.n_factors, max_iter=0)
        coef2 = self.f.coefficients
        for c1, c2 in zip(coef1, coef2):
            np.testing.assert_allclose(c1, c2)

    def test_init_fit_fit_consistent(self):
        self.f.fit(self.data, self.n_factors, max_iter=0)
        coef1 = self.f.coefficients
        self.f.fit(self.data, self.n_factors, max_iter=0)
        coef2 = self.f.coefficients
        for c1, c2 in zip(coef1, coef2):
            np.testing.assert_allclose(c1, c2)

    def test_init_array_data_matrix_consistent(self):
        self.f.fit(self.data, self.n_factors, max_iter=0)
        coef1 = self.f.coefficients
        self.f.fit(self.data.data, self.n_factors, max_iter=0)
        coef2 = self.f.coefficients
        for c1, c2 in zip(coef1, coef2):
            np.testing.assert_allclose(c1, c2)

    def test_fit_transform_consistent(self):
        n_factors = self.n_factors // 2
        self.f.fit_transform(self.data, n_factors, max_iter=5)
        coef1 = self.f.coefficients
        self.f.fit(self.data, n_factors, max_iter=5)
        coef2 = self.f.coefficients
        for c1, c2 in zip(coef1, coef2):
            np.testing.assert_allclose(c1, c2)


class TestTwoDatatypesReproduceRandom(unittest.TestCase):

    def setUp(self):
        self.n_samples = 400
        self.n_features = [2000, 180]
        self.n_factors = 10
        self.rand = np.random.RandomState(1502575153)
        self.B_a = self.rand.normal(0, 0.9,
                                    (self.n_features[0], self.n_factors))
        self.B_b = self.rand.normal(0, 1.1,
                                    (self.n_features[1], self.n_factors))
        Zvar = np.linspace(10, 1, self.n_factors)
        Zvar = Zvar / np.mean(Zvar)
        self.Z = self.rand.normal(0, np.sqrt(Zvar),
                                  (self.n_samples, self.n_factors))
        self.data = funcsfa.StackedDataMatrix([
            funcsfa.DataMatrix(np.dot(self.Z, self.B_a.T)),
            funcsfa.DataMatrix(np.dot(self.Z, self.B_b.T))])
        self.f = funcsfa.SFA()

    def test_init_full_factors_output_shapes(self):
        Z_estimated = self.f.fit_transform(self.data, self.n_factors,
                                           max_iter=0)
        self.assertEqual(Z_estimated.shape, (self.n_samples, self.n_factors))
        self.assertEqual(self.f.coefficients[0].shape,
                         (self.n_features[0], self.n_factors))
        self.assertEqual(self.f.coefficients[1].shape,
                         (self.n_features[1], self.n_factors))
        assert not np.any(np.isnan(Z_estimated))
        assert not np.any(np.isnan(self.f.coefficients[0]))
        assert not np.any(np.isnan(self.f.coefficients[1]))

    def test_init_full_factors_reconstruction(self):
        Z_estimated = self.f.fit_transform(self.data, self.n_factors,
                                           max_iter=0)
        self.assertAlmostEqual(self.f.reconstruction_error, 0.0)
        X_reconstructed = [np.dot(Z_estimated, self.f.coefficients[0].T),
                           np.dot(Z_estimated, self.f.coefficients[1].T)]
        np.testing.assert_allclose(self.data.dt(0), X_reconstructed[0])
        np.testing.assert_allclose(self.data.dt(1), X_reconstructed[1])

    def test_init_full_factors_constraints(self):
        Z_estimated = self.f.fit_transform(self.data, self.n_factors,
                                           max_iter=0)
        np.testing.assert_allclose(1, np.mean(Z_estimated ** 2))

    def test_init_part_factors_output_shapes(self):
        n_factors = self.n_factors // 2
        Z_estimated = self.f.fit_transform(self.data, n_factors, max_iter=0)
        self.assertEqual(Z_estimated.shape, (self.n_samples, n_factors))
        self.assertEqual(self.f.coefficients[0].shape,
                         (self.n_features[0], n_factors))
        self.assertEqual(self.f.coefficients[1].shape,
                         (self.n_features[1], n_factors))
        assert not np.any(np.isnan(Z_estimated))
        assert not np.any(np.isnan(self.f.coefficients[0]))
        assert not np.any(np.isnan(self.f.coefficients[1]))

    def test_init_part_factors_constraints(self):
        n_factors = self.n_factors // 2
        Z_estimated = self.f.fit_transform(self.data, n_factors, max_iter=0)
        np.testing.assert_allclose(1, np.mean(Z_estimated ** 2))
        Z_estimated2 = self.f.transform(self.data)
        np.testing.assert_allclose(1, np.mean(Z_estimated2 ** 2), rtol=1e-2)

    def test_init_part_factors_reconstruction(self):
        n_factors = self.n_factors // 2
        max_error = [np.sum(self.data.dt(0) ** 2) / self.n_samples,
                     np.sum(self.data.dt(1) ** 2) / self.n_samples]
        Z_estimated = self.f.fit_transform(self.data, n_factors, max_iter=0)
        X_reconstructed = [np.dot(Z_estimated, self.f.coefficients[0].T),
                           np.dot(Z_estimated, self.f.coefficients[1].T)]
        err = [np.sum((self.data.dt(0) - X_reconstructed[0]) ** 2) /
               self.n_samples,
               np.sum((self.data.dt(1) - X_reconstructed[1]) ** 2) /
               self.n_samples]
        assert err[0] / max_error[0] < 0.5
        assert err[1] / max_error[1] < 0.5
        np.testing.assert_allclose(sum(err), self.f.reconstruction_error)

    def test_fit_transform_consistent(self):
        n_factors = self.n_factors // 2
        self.f.fit_transform(self.data, n_factors, max_iter=0)
        coef1 = self.f.coefficients
        self.f.fit(self.data, n_factors, max_iter=0)
        coef2 = self.f.coefficients
        for c1, c2 in zip(coef1, coef2):
            np.testing.assert_allclose(c1, c2)


def _plot_convergence(m, title, filename):
    fig = plot.fig((7, 7))
    fig.suptitle(title)

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.plot(m['iteration'], m['explained_variance'], clip_on=True)
    ax1.set_xlim(left=-1)
    if np.max(m['explained_variance']) > 1:
        ax1.set_ylim(top=1)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('explained variance')

    ax2.plot(m['iteration'][1:], m['max_diff_factors'][1:], clip_on=True)
    ax2.set_xlim(left=-1)
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Max diff (Factors)')

    ax3.plot(m['iteration'][1:], m['max_diff_coefficients'][1:], clip_on=True)
    ax3.set_xlim(left=-1)
    ax3.set_yscale('log')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Max diff (Coefficients)')

    plot.save(fig, filename)


class TestFunctionalNonSparse(unittest.TestCase):
    def setUp(self):
        self.n_samples = 400
        self.n_features = [2000, 180]
        self.n_factors = 10
        self.rand = np.random.RandomState(856273263)
        self.B_a = self.rand.normal(0, 0.9,
                                    (self.n_features[0], self.n_factors))
        self.B_b = self.rand.normal(0, 1.1,
                                    (self.n_features[1], self.n_factors))
        Zvar = np.linspace(10, 1, self.n_factors)
        Zvar = Zvar / np.mean(Zvar)
        self.Z = self.rand.normal(0, np.sqrt(Zvar),
                                  (self.n_samples, self.n_factors))
        self.data = funcsfa.StackedDataMatrix([
            funcsfa.DataMatrix(np.dot(self.Z, self.B_a.T)),
            funcsfa.DataMatrix(np.dot(self.Z, self.B_b.T))])
        self.f = funcsfa.SFA()

    def testConvergenceFull(self):
        m = self.f.monitored_fit(self.data, self.n_factors, max_iter=50)
        _plot_convergence(m, 'Monitor of SFA\n 10 / 10 factors, no sparsity',
                          'testConvergenceFull.svg')

    def testConvergencePart(self):
        m = self.f.monitored_fit(self.data, self.n_factors // 2, max_iter=50)
        _plot_convergence(m, 'Monitor of SFA\n 5 / 10 factors, no sparsity',
                          'testConvergencePart.svg')


class TestFunctionalSparse(unittest.TestCase):

    def setUp(self):
        self.n_samples = 400
        self.n_features = [2000, 180]
        self.n_factors = 10
        self.rand = np.random.RandomState(856273263)
        self.B_a = self.rand.normal(0, 0.9,
                                    (self.n_features[0], self.n_factors))
        self.B_a[self.rand.rand(*self.B_a.shape) < 0.5] = 0
        self.B_b = self.rand.normal(0, 1.1,
                                    (self.n_features[1], self.n_factors))
        self.B_b[self.rand.rand(*self.B_b.shape) < 0.1] = 0
        Zvar = np.linspace(10, 1, self.n_factors)
        Zvar = Zvar / np.mean(Zvar)
        self.Z = self.rand.normal(0, np.sqrt(Zvar),
                                  (self.n_samples, self.n_factors))
        self.data = funcsfa.StackedDataMatrix([
            funcsfa.DataMatrix(np.dot(self.Z, self.B_a.T)),
            funcsfa.DataMatrix(np.dot(self.Z, self.B_b.T))])
        self.f = funcsfa.SFA()

    def testConvergenceFull(self):
        m = self.f.monitored_fit(self.data, self.n_factors, max_iter=50)
        _plot_convergence(m, title="Monitor of SFA\n 10 / 10 factors, .5/.1 "
                                   "generation sparsity",
                          filename='testConvergenceSparseFull.svg')

    def testConvergencePart(self):
        m = self.f.monitored_fit(self.data, self.n_factors // 2, max_iter=50)
        _plot_convergence(m, title="Monitor of SFA\n 5 / 10 factors, .5/.1 "
                                   "generation sparsity",
                          filename='testConvergenceSparsePart.svg')

    def testConvergencePartPenalized(self):
        m = self.f.monitored_fit(self.data, self.n_factors // 2, [.13, .06],
                                 .1, max_iter=500)
        _plot_convergence(m, title="Monitor of SFA\n 5 / 10 factors, .5/.1 "
                                   "generation sparsity, penalized fit",
                          filename='testConvergenceSparsePartPenalized.svg')
