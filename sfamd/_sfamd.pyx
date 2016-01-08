import numpy as np
from itertools import chain, accumulate

cimport sfamd_c
cimport numpy as np
cimport cython
cimport cython.view

np_size_t = np.dtype('u' + str(sizeof(size_t)))

c_version = sfamd_c.sfamd_version.decode()

cdef class Factorization:

    cdef sfamd_c.sfamd_Factorization* f
    cdef double[::1, :] data

    def __cinit__ (self, double[::1, :] data, size_t[::1] n_features_split,
                   int n_factors):
        cdef int n_samples = data.shape[0]
        cdef int n_features = data.shape[1]
        cdef int n_datatypes = n_features_split.shape[0]

        n_features_split = np.copy(n_features_split)

        if sum(n_features_split) != n_features:
            raise ValueError("Total Number of features in n_features_split "
                             "should equal n_features")

        self.data = data

        self.f = sfamd_c.sfamd_Factorization_alloc(
            n_features=n_features,
            n_factors=n_factors,
            n_samples=n_samples,
            n_datatypes=n_datatypes,
            n_features_split=&n_features_split[0],
            data=&self.data[0, 0])
        if not self.f:
            raise MemoryError()

    def  __dealloc__(self):
        sfamd_c.sfamd_Factorization_free(self.f)

    property coefficients:
        def __get__(self):
            res = (<double[:self.f.n_factors, :self.f.n_features]>
                   self.f.coefficients)
            res = np.asarray(res).T
            np.set_array_base(res, self)
            return res

    property factors:
        def __get__(self):
            res = (<double[:self.f.n_samples, :self.f.n_factors]> 
                   self.f.factors)
            res = np.asarray(res).T
            np.set_array_base(res, self)
            return res

    property factors_cov:
        def __get__(self):
            res = (<double[:self.f.n_factors, :self.f.n_factors]>
                   self.f.factor_cov)
            res = np.asarray(res).T
            np.set_array_base(res, self)
            return res

    property residual_var:
        def __get__(self):
            return <double[:self.f.n_features]> self.f.residual_var
            res = np.asarray(res)
            np.set_array_base(res, self)
            return res

    property n_datatypes:
        def __get__(self):
            return self.f.n_datatypes

    property data:
        def __get__(self):
            return np.asarray(self.data)

    cpdef sfa(self, double eps, int max_iter, double[::1] lambdas):
        cdef int n_iter = 0
        cdef double diff = 0

        # Make a character array of 'e', to indicate elastic net
        # regularization for every data type.
        cdef char[::1] regularization = cython.view.array(
                shape=(self.f.n_datatypes,),
                itemsize=sizeof(char),
                format='c')
        for i in range(self.f.n_datatypes):
            regularization[i] = b'e'

        with nogil:
            ret = sfamd_c.sfamd(self.f, eps, max_iter,
                                <char *> &regularization[0],
                                &lambdas[0], &n_iter, &diff)
        if (ret != 0):
            raise Exception('SFA Error: ' + str(ret))
        return (n_iter, diff)
