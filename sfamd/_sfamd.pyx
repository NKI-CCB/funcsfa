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
            data=&data[0, 0])
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
            res = <double[:self.f.n_features]> self.f.residual_var
            res = np.asarray(res)
            np.set_array_base(res, self)
            return res

    property n_datatypes:
        def __get__(self):
            return self.f.n_datatypes

    property data:
        def __get__(self):
            return np.asarray(self.data)

    cpdef sfa(self, double eps, int max_iter, double[::1] lambdas,
              int do_monitor=False):
        cdef int n_iter = 0
        cdef double diff = 0
        cdef sfamd_c.sfamd_Monitor *mon
        cdef Monitor monitor

        if (do_monitor):
            X = np.asarray(self.data)
            data_var = np.sum(X**2)
            monitor = Monitor(max_iter, data_var)
            mon = monitor.mon
        else:
            monitor = None
            mon = NULL

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
                                &lambdas[0], mon)
        if (ret != 0):
            raise Exception('SFA Error: ' + str(ret))
        return monitor

cdef class Monitor:

    cdef sfamd_c.sfamd_Monitor* mon
    cdef double data_var

    def __cinit__(self, int max_iter, double data_var):
        self.mon = sfamd_c.sfamd_Monitor_alloc(max_iter=max_iter)
        self.data_var = data_var

    def __dealloc__(self):
        sfamd_c.sfamd_Monitor_free(self.mon)

    property reconstruction_error:
        def __get__(self):
            res = (<double[:self.mon.n_iter+2]> self.mon.reconstruction_error)
            res = np.asarray(res)
            np.set_array_base(res, self)
            return res

    property explained_variance:
        def __get__(self):
            res = (<double[:self.mon.n_iter+2]> self.mon.reconstruction_error)
            res = np.asarray(res)
            np.set_array_base(res, self)
            res = 1 - (res / self.data_var)
            return res

    property max_diff_factors:
        def __get__(self):
            res = (<double[:self.mon.n_iter+2]> self.mon.max_diff_factors)
            res = np.asarray(res)
            np.set_array_base(res, self)
            return res

    property max_diff_coefficients:
        def __get__(self):
            res = (<double[:self.mon.n_iter+2]> self.mon.max_diff_coefficients)
            res = np.asarray(res)
            np.set_array_base(res, self)
            return res

    property iteration:
        def __get__(self):
            return list(range(-1, self.mon.n_iter+1))

    property n_iter:
        def __get__(self):
            return int(self.mon.n_iter)+1
