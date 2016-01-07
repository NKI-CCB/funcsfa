cdef extern from "sfamd.h":

    const char *sfamd_version;

    ctypedef struct sfamd_Factorization:
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

    sfamd_Factorization* sfamd_Factorization_alloc(size_t n_features,
        size_t n_factors, size_t n_samples, size_t n_datatypes,
        size_t *n_features_split, double *data)

    void sfamd_Factorization_free(sfamd_Factorization *m)

    int sfamd(sfamd_Factorization *m, double eps, int max_iter,
        char regularizations[],
        double *lambdas, int *n_iter_p, double *diff) nogil
