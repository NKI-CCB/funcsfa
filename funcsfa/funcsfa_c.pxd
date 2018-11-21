cdef extern from "funcsfa.h":

    ctypedef struct funcsfa_Factorization:
        size_t n_features
        size_t n_factors
        size_t n_samples
        size_t n_datatypes
        size_t *n_features_split
        double *data
        double *coefficients
        double *factors
        double *factor_cov
        double *residual_var

    funcsfa_Factorization* funcsfa_Factorization_alloc(size_t n_features,
        size_t n_factors, size_t n_samples, size_t n_datatypes,
        size_t *n_features_split, double *data)

    void funcsfa_Factorization_free(funcsfa_Factorization *m)

    ctypedef struct funcsfa_Monitor:
        int n_iter
        double *reconstruction_error
        double *max_diff_factors
        double *max_diff_coefficients

    funcsfa_Monitor* funcsfa_Monitor_alloc(int max_iter)

    void funcsfa_Monitor_free(funcsfa_Monitor *mon)

    int funcsfa(funcsfa_Factorization *m, double eps, int max_iter,
        char regularizations[],
        double *lambdas, funcsfa_Monitor *mon) nogil
