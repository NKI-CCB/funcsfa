cdef extern from "sfa.h":

    const char *sfa_version;

    enum regularization_t:
        Lasso, Enet

    ctypedef struct Factorization:
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

    Factorization* sfa_Factorization_alloc(size_t n_features, size_t n_factors,
        size_t n_samples, size_t n_datatypes, size_t *n_features_split,
        double *data)

    void sfa_Factorization_free(Factorization *m)

    int sfa(Factorization *m, double eps, int max_iter,
        char regularizations[],
        double *lambdas, int *n_iter_p, double *diff) nogil
