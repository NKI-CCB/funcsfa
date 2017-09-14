/** \file sfamd.h
 * Sparse Factor Analysis in C.
 *
 */
#include <stdlib.h>

extern const char *sfamd_version;

/** Data and its factorization.
 *
 * @attention All matrices are in column-major (Fortran) order
 */
typedef struct
{
  /** Number of features in the data */
  size_t n_features;
  /** Number of factors this factorization makes */
  size_t n_factors;
  /** Number of samples in the data */
  size_t n_samples;
  /** Number of datatypes the data is split in */
  size_t n_datatypes;
  /** Array of number of features per datatype, should sum up to n_datatypes */
  size_t *n_features_split;
  /** Transformed matrix of data (n_samples x n_features) */
  double *data;
  /** Coefficients of the factorization (n_features x n_factors) */
  double *coefficients;
  /** Factors of the factorization (n_factors x n_samples) */
  double *factors;
  /** Covariance matrix of factors (n_factors x n_factors) */
  double *factor_cov;
  /** Residuals of factorization (n_features) */
  double *residual_var;
} sfamd_Factorization;

/** Allocate a factorization, do not forget to call sfamd_Factorization_free() later
 * to free allocated memory
 *
 * @memberof Factorization
 */
sfamd_Factorization*
sfamd_Factorization_alloc(size_t n_features, size_t n_factors, size_t n_samples,
                        size_t n_datatypes, size_t *n_features_split,
                        double *data);
/** Free a facotorization previously allocated with sfamd_Factorization_alloc()
 *
 * @memberof Factorization
 */
void
sfamd_Factorization_free(sfamd_Factorization *m);

typedef struct
{
  /** Number of iterations */
  int n_iter;
  /** reconstruction error of data (n_iter+1) */
  double *reconstruction_error;
  /** Max elementwise difference of factors between this and last iteration
   * (n_iter+1) */
  double *max_diff_factors;
  /** Max elementwise difference of coefficients between this and last
   * iteration (n_iter+1) */
  double *max_diff_coefficients;
} sfamd_Monitor;

/** Allocate a monitor, do not forget to call sfamd_Monitor_free() later
 * to free allocated memory
 *
 * @memberof Monitor
 */
sfamd_Monitor*
sfamd_Monitor_alloc(int max_iter);

/** Free a monitor previously allocated with sfamd_Monitor_alloc()
 *
 * @memberof Monitor
 */
void
sfamd_Monitor_free(sfamd_Monitor *mon);

/** Sparse factor analysis of multiple datatypes
 *
 * @memberof Factorization
 *
 * @param[out]  m         Initialized factorization to do.
 * @param[in]   eps       When smallest change in paramters is smaller than
 *    eps, EM is assumed to be converged and stopped.
 * @param[in]   max_iter  EM stops after max_iter, even when not converged.
 * @param[in]   regularizations  Array of regularization to do per datatype.
 * @param[in]   lambda    Array of lambdas, per datatype stacked after each
 *    other.
 * @param[out]  mon       Monitor of solutions over iterations. Not filled when
 *    passed NULL, otherwise should be initalizd Monitor object.
 * @return      0 on succes, non-zero on error.
 */
int
sfamd(sfamd_Factorization *m, double eps, int max_iter,
      char regularizations[],
      double *lambdas, sfamd_Monitor *mon);
