import numpy as np
from itertools import chain, accumulate


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
        pass
