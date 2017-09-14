#include <math.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <lapacke.h>
#include <cblas.h>

#include "sfamd.h"

#define min(x, y) ((x)>(y)?(y):(x))
#define max(x, y) ((x)>(y)?(x):(y))

#define DEBUG 0
#define debug_print(fmt, ...) \
        do { if (DEBUG) fprintf(stderr, "%s:%4d:%25s(): " fmt, __FILE__, \
                                __LINE__, __func__, __VA_ARGS__); } while (0)

double eps_ev = 1e-20;

// **************************************
// Objects
// **************************************

sfamd_Factorization*
sfamd_Factorization_alloc(size_t n_features, size_t n_factors,
                          size_t n_samples, size_t n_datatypes,
                          size_t *n_features_split, double *data)
{
  size_t i;
  sfamd_Factorization *m;
  size_t sum_n_features_split = 0;

  debug_print("%s\n", "Allocating SFA object");

  m = malloc(sizeof(sfamd_Factorization));
  m->n_features = n_features;
  m->n_factors = n_factors;
  m->n_samples = n_samples;
  m->n_datatypes = n_datatypes;
  m->n_features_split = malloc(sizeof(size_t)*n_datatypes);
  for(i=0; i<n_datatypes; i++)
  {
    m->n_features_split[i] = n_features_split[i];
    sum_n_features_split += n_features_split[i];
  }
  if (sum_n_features_split != n_features)
  {
    debug_print("Total number of features (%zd) does not match sum (%zd)\n",
        n_features, sum_n_features_split);
    free(m->n_features_split);
    free(m);
    return NULL;
  }
  m->data = data;
  m->coefficients = malloc(sizeof(double)*n_features*n_factors);
  m->factors = malloc(sizeof(double)*n_factors*n_samples);
  m->factor_cov = malloc(sizeof(double)*n_factors*n_factors);
  m->residual_var = malloc(sizeof(double)*n_features);

  return(m);
}

void
sfamd_Factorization_free(sfamd_Factorization *m)
{
  debug_print("%s\n", "Freeing SFA object");

  if (m != NULL)
  {
    free(m->n_features_split);
    free(m->coefficients);
    free(m->factors);
    free(m->factor_cov);
    free(m->residual_var);
  }
  free(m);
}

sfamd_Monitor*
sfamd_Monitor_alloc(int max_iter)
{
  sfamd_Monitor *mon;
  int i;

  debug_print("%s\n", "Allocating Monitor object");

  mon = malloc(sizeof(sfamd_Monitor));
  if (mon == NULL) {
    return NULL;
  }
  mon->n_iter = 0;
  mon->reconstruction_error = malloc(sizeof(double)*(max_iter+1));
  mon->max_diff_factors = malloc(sizeof(double)*(max_iter+1));
  mon->max_diff_coefficients = malloc(sizeof(double)*(max_iter+1));
  if ((mon->reconstruction_error == NULL) ||
      (mon->max_diff_factors == NULL) ||
      (mon->max_diff_coefficients == NULL))
  {
    sfamd_Monitor_free(mon);
    return NULL;
  }
  for (i = 0; i < max_iter+1; i++)
  {
    mon->reconstruction_error[i] = NAN;
    mon->max_diff_factors[i] = NAN;
    mon->max_diff_coefficients[i] = NAN;
  }

  return mon;
}

void
sfamd_Monitor_free(sfamd_Monitor *mon)
{
  debug_print("%s\n", "Freeing Monitor object");

  if (mon != NULL)
  {
    free(mon->reconstruction_error);
    free(mon->max_diff_factors);
    free(mon->max_diff_coefficients);
  }
  free(mon);
}

/** Bunch of memory for temporary matrices.
 * @private
 */
typedef struct
{
  /** Integer vector n_factors long */
  int *factors_int;
  /** Double vector n_features long */
  double *features_double;
  /** Double matrix n_features x n_factors 1 */
  double *features_factors_double_a;
  /** Double matrix n_features x n_factors 2 */
  double *features_factors_double_b;
  /** Double matrix n_factors x n_factors 1 */
  double *factors_factors_double_a;
  /** Double matrix n_factors x n_factors 2 */
  double *factors_factors_double_b;

  /** Precomputed value of diag(data * t(data)) */
  double *diag_data_square;
  /** Precomputed value of sum(diag(data * t(data))) per data type */
  double *dt_var;
  /** Factors of previous iteration */
  double *prev_factors;
  /** Coefficients of previous iteration */
  double *prev_coefficients;
} Workspace;

static Workspace*
ws_alloc(sfamd_Factorization *m)
{
  Workspace *ws;

  debug_print("%s\n", "Allocating Workspace");

  ws = malloc(sizeof(Workspace));
  ws->factors_int = malloc(sizeof(int)*m->n_factors);
  ws->features_double = malloc(sizeof(double)*m->n_features);

  ws->features_factors_double_a = malloc(sizeof(double)*m->n_features*m->n_factors);
  ws->features_factors_double_b = malloc(sizeof(double)*m->n_features*m->n_factors);
  ws->factors_factors_double_a = malloc(sizeof(double)*m->n_factors*m->n_factors);
  ws->factors_factors_double_b = malloc(sizeof(double)*m->n_factors*m->n_factors);

  ws->diag_data_square = malloc(sizeof(double)*m->n_features);
  ws->dt_var = malloc(sizeof(double)*m->n_factors);
  ws->prev_factors = malloc(sizeof(double)*m->n_factors*m->n_samples);
  ws->prev_coefficients = malloc(sizeof(double)*m->n_features*m->n_factors);

  return ws;
}

static void
ws_init(Workspace *ws, sfamd_Factorization *m)
{
  double *d_d;
  size_t dt_i, feature_start, n_features, i;

  debug_print("%s\n", "Initializing Workspace");

  d_d = malloc(sizeof(double)*m->n_features*m->n_features);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      m->n_features, m->n_features, m->n_samples, 1,
      m->data, m->n_samples,
      m->data, m->n_samples,
      0, d_d, m->n_features);

  for (i = 0; i < m->n_features; i++)
  {
    ws->diag_data_square[i] = d_d[i*m->n_features + i];
  }
  feature_start = 0;
  for(dt_i = 0; dt_i < m->n_datatypes; dt_i++)
  {
    ws->dt_var[dt_i] = 0;
    n_features = m->n_features_split[dt_i];
    for(i = feature_start; i < (feature_start + n_features); i++)
    {
      ws->dt_var[dt_i] += ws->diag_data_square[i];
    }
    ws->dt_var[dt_i] /= n_features;
    feature_start = feature_start + n_features;
  }

  free(d_d);
}

static void
ws_free(Workspace *ws)
{
  debug_print("%s\n", "Freeing Workspace");

  free(ws->factors_int);
  free(ws->features_double);
  free(ws->features_factors_double_a);
  free(ws->features_factors_double_b);
  free(ws->factors_factors_double_a);
  free(ws->factors_factors_double_b);

  free(ws->diag_data_square);
  free(ws->dt_var);
  free(ws->prev_factors);
  free(ws->prev_coefficients);

  free(ws);
}

// **************************************
// Utility Functions
// **************************************

static double
max_diff(double *a, double *b, size_t length)
{
  size_t i;
  double md, d;

  md = 0;
  for (i = 0; i < length; i++)
  {
    d = fabs(a[i] - b[i]);
    if (d > md)
    {
      md = d;
    }
  }
  return md;
}

static bool
finite_array(double *a, size_t length)
{
  for (size_t i=0; i<length; i++)
  {
    if (!isfinite(a[i]))
    {
      return false;
    }
  }
  return true;
}

static bool
positive_array(double *a, size_t length)
{
  for (size_t i=0; i<length; i++)
  {
    if (!(a[i] > 0.0))
    {
      return false;
    }
  }
  return true;
}

#define check_finite(a, length) \
    if (DEBUG && !finite_array(a, length)) {\
      debug_print("%s\n", "Error, non-finite values in matrix");}

#define check_positive(a, length) \
    if (DEBUG && !positive_array(a, length)) {\
      debug_print("%s\n", "Error, non-positive values in matrix");}

// **************************************
// Algorithm
// **************************************

static int
expectation_factors(sfamd_Factorization *m, Workspace *ws)
{
  size_t i, j;
  double *r_c, *c_r_c, *c_r_c_i, *o, *c_o;
  double factor_mean, factor_var, factor_sd, deviance;
  int *ipiv;
  lapack_int info;

  debug_print("%s\n", "Computing Expectation and Covariance of Factors");
  check_finite(m->coefficients, m->n_features*m->n_factors);
  check_finite(m->residual_var, m->n_features);
  check_positive(m->residual_var, m->n_features);

  /* Compute $\Psi^{-1} B$
   * r_c = diag(1/residual_var) * coefficients */
  r_c = ws->features_factors_double_a;
  for (i=0; i < m->n_features; i++)
  {
    for (j=0; j < m->n_factors; j++)
    {
      r_c[j * m->n_features + i] =
          m->coefficients[j * m->n_features + i] / m->residual_var[i];
    }
  }

  /* Compute $B^T \Psi^{-1} B + I$
   * c_r_c = t(coefficients) * r_c + I */
  c_r_c = ws->factors_factors_double_a;
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      m->n_factors, m->n_factors, m->n_features, 1,
      m->coefficients, m->n_features,
      r_c, m->n_features,
      0, c_r_c, m->n_factors);
  for (i = 0; i < m->n_factors; i++)
  {
    c_r_c[i*m->n_factors + i] += 1;
  }

  /* Compute $O = \Psi^{-1}B(B^T \Psi^{-1} B + I)^{-1}$
   * o = r_c*solve(c_r_c) */
  c_r_c_i = ws->factors_factors_double_b;
  ipiv = ws->factors_int;
  for (i = 0; i < m->n_factors; i++)
  {
    for (j = 0; j < m->n_factors; j++)
    {
      c_r_c_i[j * m->n_factors + i] = (i == j);
    }
  }

  info = LAPACKE_dgesv(LAPACK_COL_MAJOR, m->n_factors, m->n_factors,
      c_r_c, m->n_factors, ipiv,
      c_r_c_i, m->n_factors);
  if (info != 0) {
    debug_print("Lapack error: %d\n", info);
    return -1;
  }
  c_r_c = NULL;
  o = ws->features_factors_double_b;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
     m->n_features, m->n_factors, m->n_factors, 1,
     r_c, m->n_features,
     c_r_c_i, m->n_factors,
     0, o, m->n_features);

  /* Compute $E[Z|X] = (\Psi^{-1}B(B^T\Psi^{-1}B+I)^{-1})^TX$
   * factors = t(o) * t(X)
   */
  cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans,
      m->n_factors, m->n_samples, m->n_features, 1,
      o, m->n_features,
      m->data, m->n_samples,
      0, m->factors, m->n_factors);

  /* Compute $B^T\Psi^{-1}B(B^T\Psi^{-1}B+I)^{-1}$
   * c_o = t(coefficients) * o
   */
  c_o = ws->factors_factors_double_a;
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      m->n_factors, m->n_factors, m->n_features, 1,
      m->coefficients, m->n_features,
      o, m->n_features,
      0, c_o, m->n_factors);

  /*
   * Compute $E[ZZ^T|X] = I - (B^T\Psi^{-1}B(B^T\Psi^{-1}B+I)^{-1})^T + E[Z|X]E[Z|X]^T$
   * factor_cov = factors * t(factors) + I - c_o
   */
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      m->n_factors, m->n_factors, m->n_samples, 1,
      m->factors, m->n_factors,
      m->factors, m->n_factors,
      0, m->factor_cov, m->n_factors);

  for (i = 0; i < m->n_factors; i++)
  {
    for (j = 0; j < m->n_factors; j++)
    {
      m->factor_cov[j * m->n_factors + i] += (i==j) - c_o[j * m->n_factors + i];
    }
  }

  debug_print("Factor value: %g\n", m->factors[0]);

  // Rescale factors
  for (i = 0; i < m->n_factors; i++)
  {
    factor_mean = 0;
    for (j = 0; j < m->n_samples; j++)
    {
      factor_mean += m->factors[i + j*m->n_factors];
    }
    factor_mean /= m->n_samples;
    debug_print("  Factor mean %zd: %g\n", i, factor_mean);
    factor_var = 0;
    for (j = 0; j < m->n_samples; j++)
    {
      deviance = m->factors[i + j*m->n_factors] - factor_mean;
      factor_var += deviance*deviance;
    }
    factor_var /= m->n_samples;
    debug_print("  Factor variation %zd: %g\n", i, factor_var);
    for (j = 0; j < m->n_factors; j++)
    {
      m->factor_cov[i + j*m->n_factors] /= factor_var;
    }
    factor_sd = sqrt(factor_var);
    for (j = 0; j < m->n_samples; j++)
    {
      m->factors[i + j * m->n_factors] /= factor_sd;
    }
  }

  return 0;
}

static int
maximize(sfamd_Factorization *m, double *lambdas, Workspace *ws)
{
  const size_t n_lambda = 2;
  size_t dt_i, f_i, lambda_i, k;
  double *ZXt, *BZZt;
  double unpenalized_coeff, l1, l2, dt_var;
  size_t n_features, feature_start;

  debug_print("%s\n", "Estimating Coefficients");

  check_finite(m->factor_cov, m->n_factors*m->n_factors);
  check_finite(m->factors, m->n_factors*m->n_samples);

  ZXt = ws->features_factors_double_a;
  BZZt = ws->features_double;
  // Compute E[Z|X]X^T
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      m->n_factors, m->n_features, m->n_samples, 1,
      m->factors, m->n_factors,
      m->data, m->n_samples,
      0, ZXt, m->n_factors);

  // Coordinate Descent over factors
  for(k = 0; k < m->n_factors; k++)
  {
    debug_print("\tCoordinate descent for factor: %zd\n", k);
    // Compute $BE[ZZ^T|X]_(\cdot k)$
    cblas_dgemv(CblasColMajor, CblasNoTrans,
      m->n_features, m->n_factors, 1.0,
      m->coefficients, m->n_features,
      m->factor_cov+(k*m->n_factors), 1,
      0.0, BZZt, 1);

    lambda_i = 0;
    feature_start = 0;
    for(dt_i = 0; dt_i < m->n_datatypes; dt_i++)
    {
      n_features = m->n_features_split[dt_i];
      l1 = lambdas[lambda_i];
      l2 = lambdas[lambda_i+1];
      for(f_i = feature_start; f_i < (feature_start + n_features); f_i++)
      {
        // Compute: $$
        //  \hat B_{(ij)} \leftarrow \frac
        //    {\operatorname{S}(
        //      \hat B_{(ij)} + E[Z|X]_{(j \cdot)} X_{(i \cdot)}^T -
        //      \hat B_{(i\cdot)} \operatorname{E}[ZZ^T|X]_{(\cdot j)},
        //      l_1)}
        //    {1+l_2}
        // $$
        unpenalized_coeff = m->coefficients[f_i + (k * m->n_features)] +
                            ((ZXt[k + (f_i * m->n_factors)] - BZZt[f_i])
                             / m->n_samples);
        m->coefficients[f_i + (k * m->n_features)] =
          copysign(fmax(0.0, fabs(unpenalized_coeff) - l1),
                   unpenalized_coeff) /
          (1 + l2);
      }
      feature_start = feature_start + n_features;
      lambda_i = lambda_i + n_lambda;
    }
  }

  debug_print("%s\n", "Estimating Residual Variance");
 
  // Residual variance per data type
  feature_start = 0;
  for(dt_i = 0; dt_i < m->n_datatypes; dt_i++)
  {
    dt_var = 0;
    n_features = m->n_features_split[dt_i];
    for(f_i = feature_start; f_i < (feature_start + n_features); f_i++)
    {
      for(k = 0; k < m->n_factors; k++)
      {
        dt_var += (ZXt[k + (f_i * m->n_factors)] *
                   m->coefficients[f_i + (k * m->n_features)]);
      }
    }
    dt_var = (ws->dt_var[dt_i] - (dt_var / n_features)) / m->n_samples;
    for(f_i = feature_start; f_i < (feature_start + n_features); f_i++)
    {
      if (dt_var > eps_ev) {
        m->residual_var[f_i] = dt_var;
      } else {
        m->residual_var[f_i] = eps_ev;
      }
    }
    debug_print("\tResidual Variance %zd: %g\n", dt_i, dt_var);
    feature_start = feature_start + n_features;
  }
  return 0;
}

static int
init_svd(sfamd_Factorization *m, Workspace *ws)
{
  lapack_int info;
  size_t i, j;
  double *x, *s, *superb, *vt, *u;
  double scale, f, explained_var, data_var;
  size_t total_n_factors;

  debug_print("%s\n", "Initializing with SVD");

  total_n_factors = min(m->n_features, m->n_samples);

  x = malloc(sizeof(double) * m->n_features * m->n_samples);
  s = malloc(sizeof(double) * total_n_factors);
  superb = malloc(sizeof(double) * (total_n_factors-1));
  vt = malloc(sizeof(double) * total_n_factors * m->n_features);
  u = malloc(sizeof(double) * total_n_factors * m->n_samples);
  if (x == NULL || s == NULL || superb == NULL || vt == NULL)
  {
    free(x);
    free(vt);
    free(s);
    free(u);
    free(superb);
    return -1;
  }

  for (i=0; i < m->n_features * m->n_samples; i++)
  {
    x[i] = m->data[i];
  }

  /* vt = svd(X).V.t */
  debug_print("%s\n", "\tStarting Lapack SVD");
  info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S',
      m->n_samples, m->n_features,
      x, m->n_samples,
      s, u, m->n_samples,
      vt, total_n_factors, superb);
  if (info != 0) {
    debug_print("Lapack error: %d\n", info);
    free(x);
    free(vt);
    free(u);
    free(s);
    free(superb);
    return -1;
  }
  debug_print("%s\n", "\tDone Lapack SVD");

  scale = 0;
  for (i=0; i < m->n_factors; i++)
  {
    for (j=0; j < m->n_samples; j++)
    {
      f = u[j + i*m->n_samples] * s[i];
      m->factors[i + j*m->n_factors] = f;
      scale += f*f;
    }
  }
  scale /= m->n_factors * m->n_samples;
  scale = sqrt(scale);
  for (i=0; i < m->n_factors; i++)
  {
    for (j=0; j < m->n_features; j++)
    {
      m->coefficients[j + i*m->n_features] = vt[i + j*total_n_factors] * scale;
    }
  }
  for (i=0; i < m->n_factors; i++)
  {
    for (j=0; j < m->n_samples; j++)
    {
      m->factors[i + j*m->n_factors] /= scale;
    }
  }

  explained_var = 0;
  for (i=0; i < m->n_factors; i++)
  {
    explained_var += s[i]*s[i];
  }
  explained_var /= m->n_samples * m->n_features;
  data_var = 0;
  for (i=0; i < m->n_features; i++)
  {
    data_var += ws->diag_data_square[i];
  }
  data_var /= m->n_samples * m->n_features;
  debug_print("\tData Var: %g\n", data_var);
  explained_var = data_var - explained_var;
  explained_var = fmax(0, explained_var) + eps_ev;
  for (i = 0; i < m->n_features; i++) {
    m->residual_var[i] = explained_var;
  }
  debug_print("\tResidual Variance: %g\n", explained_var);

  free(x);
  free(vt);
  free(u);
  free(s);
  free(superb);
  return 0;
}

double
calc_reconstruction_error(sfamd_Factorization *m, Workspace *ws)
{
  int i, j;
  double error, e;
  double *prediction;

  error = 0.0;
  for (i = 0; i < m->n_samples; i++)
  {
    prediction = ws->features_double;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        m->n_features, m->n_factors,
        1.0, m->coefficients, m->n_features,
        m->factors + (i*m->n_factors), 1,
        0.0, prediction, 1);
    for (j = 0; j < m->n_features; j++)
    {
      e = m->data[i + (j * m->n_samples)] - prediction[j];
      error += e*e;
    }
  }
  return error;
}

int
sfamd(sfamd_Factorization *m, double eps, int max_iter,
      char regularizations[], double *lambdas, sfamd_Monitor *mon)
{
  int iter = 0;
  int err = 0;
  size_t i;
  double diff = DBL_MAX;
  double diff_f, diff_c;
  double rec_error;
  int const LEN_PREV_REC_ERROR = 10;
  double prev_rec_error[LEN_PREV_REC_ERROR];
  int prev_rec_error_i=0;

  Workspace *ws;

  for (i=0; i<m->n_datatypes; i++)
  {
    if (regularizations[i] != 'e')
    {
      debug_print("%s\n", "Only elastic net regularization is supported");
      return -4;
    }
  }

  ws = ws_alloc(m);
  ws_init(ws, m);

  debug_print("%s\n", "Starting Factorization");

  err = init_svd(m, ws);
  if (err != 0)
  {
    return(1);
  }

  rec_error = calc_reconstruction_error(m, ws);
  if (mon != NULL)
  {
    mon->reconstruction_error[0] = rec_error;
    mon->max_diff_factors[0] = 0;
    mon->max_diff_coefficients[0] = 0;
  }
  for (i=0; i<LEN_PREV_REC_ERROR; i++) {
    prev_rec_error[i] = 0.0;
  }

  /* EM Loop */
  while (iter < max_iter && diff > eps) {
    debug_print("Iteration %d\n", iter);
    memcpy(ws->prev_factors, m->factors,
        sizeof(double)*m->n_factors*m->n_samples);
    memcpy(ws->prev_coefficients, m->coefficients,
        sizeof(double)*m->n_features*m->n_factors);
    err = expectation_factors(m, ws);
    if (err != 0)
    {
      debug_print("Error in computing expectation of factors: %d\n", err);
      ws_free(ws);
      return(2);
    }
    err = maximize(m, lambdas, ws);
    if (err != 0)
    {
      debug_print("Error in maximization: %d\n", err);
      ws_free(ws);
      return(3);
    }

    diff_f = max_diff(m->factors,
        ws->prev_factors, m->n_factors*m->n_samples);
    diff_c = max_diff(m->coefficients, ws->prev_coefficients,
        m->n_features*m->n_factors);
    debug_print("Diff (factor/coefficients): %g; %g\n", diff_f, diff_c);
    prev_rec_error[prev_rec_error_i] = rec_error;
    prev_rec_error_i += 1;
    if (prev_rec_error_i > LEN_PREV_REC_ERROR) {
      prev_rec_error_i = 0;
    }
    rec_error = calc_reconstruction_error(m, ws);
    debug_print("Reconstruction Error: %g\n", rec_error);
    diff = 0.0;
    for (i=0; i<LEN_PREV_REC_ERROR; i++) {
      diff += fabs(prev_rec_error[i] - rec_error);
    }
    diff /= LEN_PREV_REC_ERROR;
    debug_print("Diff (Reconstruction Error): %g\n", diff);

    if (mon != NULL)
    {
      mon->reconstruction_error[iter+1] = rec_error;
      mon->max_diff_factors[iter+1] = diff_f;
      mon->max_diff_coefficients[iter+1] = diff_c;
      mon->n_iter = iter;
    }
    iter = iter + 1;
  }
  ws_free(ws);

  return 0;
}
