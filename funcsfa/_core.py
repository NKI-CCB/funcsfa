from itertools import chain
import numbers

import numpy as np

from funcsfa._lib import Factorization, np_size_t
from funcsfa._data_matrix import DataMatrix, StackedDataMatrix


class SFA():
    """Sparse factor analysis of multiple data types.

    This class is an implementation of the sparse factor analysis method that
    finds common factors of feature in data. Groups of features can be from
    different data types. Sparsity and residual variance can be different per
    data type.
    """

    def __init__(self):
        self._factorization = None

    @staticmethod
    def _penalties_to_array(l1, l2, n_dt):
        """Converts inputs of l1 and l2 penalties in various formats to
        arrays.

        A single value is repeated to be the same for all data types. Lists of
        numbers are converted to arrays and should have a length of
        ``n_dt``.

        Arguments:
            l1 (float or list of float or ndarray):
                l1 penalty(s)
            l2 (float or list of float or ndarray):
                l2 penalty(s)
            n_dt(int):
                Number of data types to extend array to if just one l1 or l2
                penalty is given.

        Returns:
            Tuple of arrays of length ``n_dt`` with l1 and l2 penalties.
        """
        if isinstance(l1, numbers.Real):
            l1 = [float(l1)] * n_dt
        if isinstance(l2, numbers.Real):
            l2 = [float(l2)] * n_dt
        l1 = np.array(l1, dtype='f8')
        l2 = np.array(l2, dtype='f8')
        if not l1.shape == l2.shape == (n_dt, ):
            raise Exception("Number of data types not consistent among "
                            "parameters.")
        lambdas = np.array(list(chain(*zip(l1, l2))), dtype='f8')
        return lambdas

    def fit(self, data, n_factors, l1=0, l2=0, max_iter=5000, eps=1e-6,
            do_monitor=False):
        """Fit coefficients from data.

        Runs an EM algorithm to find the best fit.

        Arguments:
            data:
                Input data to find factors in. A `~sfamd.StackedDataMatrix`
                or list of `~sfamd.DataMatrix`.
            n_factors:
                Number of factors to estimate
            l1:
                :math:`\ell_1` penalties, possibly per data type (lasso)
            l2:
                :math:`\ell_2` penalties, possibly per data type (ridge)
            max_iter:
                Maximum number of em iterations to perform.
            eps:
                Convergence criterion. If the estimated coefficients don't
                change more that this, the algorithm stops.
        """
        if not isinstance(data, DataMatrix):
            data = DataMatrix(data)
        if not isinstance(data, StackedDataMatrix):
            data = StackedDataMatrix([data])
        if any([n < n_factors for n in data.dt_n_features]):
            raise Exception("Number of features in each data type needs to be "
                            "higher than the number of factors")
        lambdas = SFA._penalties_to_array(l1, l2, data.n_dt)
        self._data = data
        d = np.require(data.dataW, '=f8', 'F')
        nfs = np.asarray(data.dt_n_features, np_size_t)
        self._factorization = Factorization(d, nfs, n_factors)
        self.monitor = self._factorization.sfa(eps, max_iter, lambdas,
                                               do_monitor)

    @property
    def coefficients(self):
        """Coefficients of this model, per data type."""
        return [self._factorization.coefficients[s, :]
                for s in self._data.slices]

    @property
    def reconstruction_error(self):
        B = self._factorization.coefficients
        Z = self._factorization.factors
        rec = np.dot(B, Z)
        err = np.sum((self._data.dataW.T - rec)**2, 0)
        return float(np.mean(err))

    def fit_transform(self, data, n_factors, l1=0, l2=0, max_iter=5000,
                      eps=1e-6):
        self.fit(data, n_factors, l1, l2, max_iter, eps)
        return self._factorization.factors.T

    def transform(self, data: DataMatrix) -> np.ndarray:
        """Gives best estimate of factors for data given coefficients.

        Arguments:
            data:
                Data to determine factors for.
        """
        Psi_inv = np.reciprocal(self._factorization.residual_var)
        B = self._factorization.coefficients
        Psi_inv_B = Psi_inv[:, np.newaxis] * B
        btb = (Psi_inv_B.T).dot(B)
        btbi = np.linalg.inv(btb + np.identity(btb.shape[0]))
        eq1 = np.dot(Psi_inv_B, btbi)
        Z = np.dot(data.dataW, eq1)
        return Z

    def monitored_fit(self, data, n_factors, l1=0, l2=0, max_iter=5000,
                      eps=1e-6):
        self.fit(data, n_factors, l1, l2, max_iter, eps, True)
        return self._monitor_as_dict()

    def _monitor_as_dict(self):
        mon = dict()
        mon['iteration'] = list(self.monitor.iteration)
        mon['max_diff_factors'] = list(self.monitor.max_diff_factors)
        mon['max_diff_coefficients'] = list(self.monitor.max_diff_coefficients)
        mon['reconstruction_error'] = list(self.monitor.reconstruction_error)
        mon['explained_variance'] = list(self.monitor.explained_variance)
        return mon
