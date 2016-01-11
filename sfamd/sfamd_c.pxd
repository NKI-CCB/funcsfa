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

    ctypedef struct sfamd_Monitor:
        int n_iter
        double *reconstruction_error
        double *max_diff_factors
        double *max_diff_coefficients

    sfamd_Monitor* sfamd_Monitor_alloc(int max_iter)

    void sfamd_Monitor_free(sfamd_Monitor *mon)

    int sfamd(sfamd_Factorization *m, double eps, int max_iter,
        char regularizations[],
        double *lambdas, sfamd_Monitor *mon) nogil
