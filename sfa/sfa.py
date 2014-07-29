from sfa import _sfa
import numpy as np


class Factorization():

    def __init__(self, data, n_features_split, n_factors):
        data = np.require(data, 'float64', 'FA')
        n_features_split = np.require(n_features_split, _sfa.size_t, 'A')
        n_factors = np.require(n_factors, int, 'A')

        if sum(n_features_split) != data.shape[1]:
            raise ValueError('Total number of features in n_features_split: ' +
                             str(sum(n_features_split)) +
                             ', is not equal to the number of features in' +
                             ' data: ' + str(data.shape[1]))

        self._f = _sfa.Factorization(data, n_features_split, n_factors)

    @property
    def coefficients(self):
        return self._f.coefficients

    @property
    def factors(self):
        return self._f.factors

    @property
    def factors_cov(self):
        return self._f.factors_cov

    @property
    def residual_var(self):
        return self._f.residual_var

    def sfa(self, eps, max_iter, regularization, lambdas):
        if (len(regularization) != self._f.n_datatypes):
            raise ValueError('Number of regularization types should match '
                             'number of data datypes')
        if (len(lambdas) != self._f.n_datatypes):
            raise ValueError('Number of lambdas should match number of '
                             'datatypes')
        for i in range(self._f.n_datatypes):
            if (len(lambdas[i]) != _sfa.n_lambdas_per_reg[regularization[i]]):
                raise ValueError('Number of lambdas of datatype ' + str(i) +
                                 ': ' + str(len(lambdas[i])) + ', expected: ' +
                                 str(self.n_lambdas_per_reg[regularization[i]]))
        regularization = np.array([r.value for r in regularization],
                                  _sfa.regularization_t)

        lambdas = np.array(sum(lambdas, []), 'float64')

        return(self._f.sfa(eps, max_iter, regularization.view(np.int8),
               lambdas))
