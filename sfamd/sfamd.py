import numpy as np
from itertools import chain, accumulate
import numbers
from sfamd import _sfamd


class DataMatrix():
    """A matrix of data, with names for features and samples, and weights.

    Arguments:
        data(numpy.ndarray):
            Initialization of `.data`, the actual data to store. Shape
            should be samples by features.
        samples(List[str]):
            Initialization of sample names (`.samples`). Should be same length
            as first dimension of data. Default is a list of empty strings.
        features(List[str]):
            Initialization of features names (`.features`). Should be same
            length as second dimension of data. Default is a list of empty
            strings.
        weights(ndarray):
            Initialization of `.weights`. Should be same shape as data. Default
            is a weight of one for every datapoint.

    Attributes:
        data(numpy.ndarray):
            Array with the actual data.
        weights(numpy.ndarray):
            Array with precision weights for the data, same shape as `data`.
        samples(List[str]):
            List with sample names.
        features(List[str]):
            List with feature names.
    """

    def __init__(self, data, samples=None, features=None, weights=None):
        if samples is None:
            samples = ["" for _ in range(data.shape[0])]
        if data.shape[0] != len(samples):
            raise Exception("Data dimension 0 should be the same as the "
                            "number of sample names")

        if features is None:
            features = ["" for _ in range(data.shape[1])]
        if data.shape[1] != len(features):
            raise Exception("Data dimension 1 should be the same as the "
                            "number of feature names")

        if weights is None:
            weights = np.ones_like(data)
        if weights.shape != data.shape:
            raise Exception("Dimensions of weights does not equal "
                            "dimensions of data")

        self.data = data
        self.samples = samples
        self.features = features
        self.weights = weights


class StackedDataMatrix(DataMatrix):
    """A `.DataMatrix` containing multiple data types.

    Data and weights matrices are concatenated along the feature dimension (1,
    columns). Lists of feature names are concatenated. Sample names should be
    equal and in the same order for all data types. Data types of all data and
    weight matrices should match.

    Arguments:
        matrices(List[DataMatrix]):
            List of `.DataMatrix` to stack together.
        dt_names(List[str]):
            Names of data types, same order and length as matrices. Defaults
            to empty strings.

    Attributes:
        dt_names(List[str]):
            names of data types.
        dt_n_features(List[str]):
            number of features per data type.
        slices(List[slice]):
            slices to use for indexing `~.data`.
    """

    def __init__(self, matrices, dt_names=None):
        if dt_names is None:
            dt_names = ["" for _ in range(len(matrices))]

        samples = matrices[0].samples
        for m in matrices:
            for n1, n2 in zip(m.samples, samples):
                if n1 != n2:
                    raise Exception("Samples names don't match.")

        dtype = matrices[0].data.dtype
        for m in matrices:
            if m.data.dtype != dtype:
                raise Exception("dtypes of matrices don't match")

        dt_n_features = [len(m.features) for m in matrices]
        slices = self._compute_slices(dt_n_features)

        data = np.zeros((len(samples), sum(dt_n_features)), dtype=dtype)
        weights = np.ones((len(samples), sum(dt_n_features)), dtype=dtype)
        for m, sl in zip(matrices, slices):
            data[:, sl] = m.data
            weights[:, sl] = m.weights
        features = list(chain(*[m.features for m in matrices]))

        super().__init__(data, samples, features, weights)

        self.dt_names = dt_names
        self.n_dt = len(dt_names)
        self.slices = slices
        self.dt_n_features = dt_n_features

    def dt(self, idx) -> np.ndarray:
        """Index the data to get one datatype.

        Attributes:
            idx(int or str):
                Name or integer index of datatype.
        Returns:
            Numpy array with the data of one data type.
        """
        if isinstance(idx, str):
            idx = self.dt_names.index(idx)
        return self.data[:, self.slices[idx]]

    @staticmethod
    def _compute_slices(feature_lengths):
        return [slice(i_start, i_stop) for i_start, i_stop in
                zip(chain([0], accumulate(feature_lengths)),
                    accumulate(feature_lengths))]


class SFA():
    """Sparse factor analysis of multiple data types.

    This class is an implementation of the sparse factor analysis method that
    finds common factors of feature in data. Groups of features can be from
    different data types. Sparsity and residual variance can be different per
    data type.
    """

    def __init__(self):
        self._factorization = None

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
        d = np.require(data.data, '=f8', 'F')
        nfs = np.asarray(data.dt_n_features, _sfamd.np_size_t)
        self._factorization = _sfamd.Factorization(d, nfs, n_factors)
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
        err = np.sum((self._data.data.T - rec)**2, 0)
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
        Z = np.dot(data.data, eq1)
        return Z

    def monitored_fit(self, data, n_factors, l1=0, l2=0, max_iter=5000,
                      eps=1e-6):
        self.fit(data, n_factors, l1, l2, max_iter, eps, True)
        return self.monitor
