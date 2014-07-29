import numpy as np
import enum

cimport sfa_c
cimport numpy as np
cimport cython

size_t = np.dtype('u' + str(sizeof(size_t)))
cdef sfa_c.regularization_t x = <sfa_c.regularization_t> 0
x = <sfa_c.regularization_t> (x - 1)
if (x > 0):
    regularization_t = np.dtype('u' + str(sizeof(sfa_c.regularization_t)))
else:
    regularization_t = np.dtype('i' + str(sizeof(sfa_c.regularization_t)))

class Regularization(enum.Enum):
    """Type of regularization"""
    Lasso = sfa_c.Lasso
    Enet = sfa_c.Enet

@cython.boundscheck(False)
cdef class Factorization:
    """A factorization of a specific dataset.
    Factorization(data, n_features_split, n_factors)
        data: Array of data to factorize, feature of same datype should be
            together.
        n_features_split: Number of features per datatype, in same order as the
            features in data.
        n_factors: Maximum number of factors this factorization should make.
    """

    cdef sfa_c.Factorization* f
    cdef double[::1, :] data


    def __cinit__ (self, double[::1, :] data, size_t[::1] n_features_split,
                   int n_factors):
        cdef int n_samples = data.shape[0]
        cdef int n_features = data.shape[1]
        cdef int n_datatypes = n_features_split.shape[0]

        self.data = data

        self.f = sfa_c.sfa_Factorization_alloc(
            n_features=n_features,
            n_factors=n_factors,
            n_samples=n_samples,
            n_datatypes=n_datatypes,
            n_features_split=&n_features_split[0],
            data=&self.data[0, 0])
        if not self.f:
            raise MemoryError()

    def  __dealloc__(self):
        sfa_c.sfa_Factorization_free(self.f)

    property coefficients:
        def __get__(self):
            return np.asarray(<double[:self.f.n_factors, :self.f.n_features]>
                              self.f.coefficients).T

    property factors:
        def __get__(self):
            return np.asarray(<double[:self.f.n_samples, :self.f.n_factors]>
                              self.f.factors).T

    property factors_cov:
        def __get__(self):
            return np.asarray(<double[:self.f.n_factors, :self.f.n_factors]>
                              self.f.factor_cov).T

    property residual_var:
        def __get__(self):
            return np.asarray(<double[:self.f.n_features]>
                              self.f.residual_var)

    property n_datatypes:
        def __get__(self):
            return self.f.n_datatypes

    @cython.boundscheck(False)
    cpdef sfa(self, double eps, int max_iter,
              char[::1] regularization,
              double[::1] lambdas):
        """Perform factorization.

        Args:
            eps:    When smallest change in paramters is smaller than
                eps, EM is assumed to be converged and stopped.
            max_iter:  EM stops after max_iter, even when not converged.
            regularization: List of Regularization to apply per datatype.
            lambas:     List of list of float, giving regularization parameters
                per datatype.
        """
        cdef int n_iter = 0
        cdef double diff = 0
        with nogil:
            ret = sfa_c.sfa(self.f, eps, max_iter,
                            <sfa_c.regularization_t *> &regularization[0],
                            &lambdas[0], &n_iter, &diff)
        if (ret != 0):
            raise Exception('SFA Error: ' + str(ret))
        return (n_iter, diff)

n_lambdas_per_reg = {
    Regularization.Lasso: 1,
    Regularization.Enet: 2
}
