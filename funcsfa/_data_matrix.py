from itertools import chain, accumulate

import numpy as np


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

    @property
    def dataW(self):
        d = self.data * self.weights
        d -= np.mean(d, 0, keepdims=True)
        return d


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

    def dt(self, idx, weighted=True) -> np.ndarray:
        """Index the data to get one datatype.

        Attributes:
            idx(int or str):
                Name or integer index of datatype.
        Returns:
            Numpy array with the data of one data type.
        """
        if isinstance(idx, str):
            idx = self.dt_names.index(idx)
        d = self.data[:, self.slices[idx]]
        if weighted:
            d = d * self.weights[:, self.slices[idx]]
            d -= np.mean(d, 0, keepdims=True)
        return d

    @staticmethod
    def _compute_slices(feature_lengths):
        return [slice(i_start, i_stop) for i_start, i_stop in
                zip(chain([0], accumulate(feature_lengths)),
                    accumulate(feature_lengths))]

    def to_netcdf(self, fn, feature_names=None, sample_dim='sample'):
        import netCDF4

        if feature_names is None:
            feature_names = [f'{dt}_feature' for dt in self.dt_names]

        with netCDF4.Dataset(fn, 'w') as ds:
            ds.createDimension(sample_dim, len(self.samples))
            samples_a = np.array(self.samples)
            samples = ds.createVariable(sample_dim, samples_a.dtype,
                                        sample_dim)
            for i, s in enumerate(self.samples):
                samples[i] = s

            ds.setncattr_string('data_types', self.dt_names)
            for dt_i, dt in enumerate(self.dt_names):
                ds.createDimension(feature_names[dt_i],
                                   self.dt_n_features[dt_i])
                fvar_a = np.array(self.features[self.slices[dt_i]])
                fvar = ds.createVariable(feature_names[dt_i], fvar_a.dtype,
                                         feature_names[dt_i])
                for i, v in enumerate(self.features[self.slices[dt_i]]):
                    fvar[i] = v

                var = ds.createVariable(dt, 'f8',
                                        (sample_dim, feature_names[dt_i]))
                var[:] = self.data[:, self.slices[dt_i]]

                wvar = ds.createVariable(f'{dt}_weights', 'f8',
                                         (sample_dim, feature_names[dt_i]))
                wvar[:] = self.weights[:, self.slices[dt_i]]

    @classmethod
    def from_netcdf(cls, fn, sample_dim='sample'):
        import netCDF4

        with netCDF4.Dataset(fn, 'r') as ds:
            dt_names = ds.getncattr('data_types')
            samples = list(ds[sample_dim])
            dms = dict()
            for dt in dt_names:
                assert(len(ds[dt].dimensions) == 2)
                assert(ds[dt].dimensions[0] == sample_dim)
                feature_dim = ds[dt].dimensions[1]
                dms[dt] = DataMatrix(
                    np.array(ds[dt]),
                    samples,
                    list(ds[feature_dim]),
                    np.array(ds[f'{dt}_weights']),
                )

        return cls([dms[dt] for dt in dt_names], dt_names)
