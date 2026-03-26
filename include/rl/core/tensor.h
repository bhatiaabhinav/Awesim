#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

// ---- Tensor ----

typedef struct Tensor {
    float* data;
    int rank;
    int n1;
    int n2; // if rank >= 2
    int n3; // if rank >= 3
    int n4; // if rank >= 4
} Tensor;

Tensor* tensor_malloc(int rank, int n1, int n2, int n3, int n4);
Tensor* tensor1d_malloc(int n1);    
Tensor* tensor2d_malloc(int n1, int n2);
Tensor* tensor3d_malloc(int n1, int n2, int n3);
Tensor* tensor4d_malloc(int n1, int n2, int n3, int n4);
void    tensor_assert_shape(Tensor* t, int n1, int n2, int n3, int n4);
void    tensor_free(Tensor* tensor);
void    tensors_free(Tensor* tensors[]);    // frees each tensors in a NULL ended list of Tensor*
float* tensor_at(Tensor* tensor, int i, int j, int k, int l);  // returns pointer to element (i,j,k,l) for rank 4, or (i,j,k) for rank 3, etc.
float* tensor1d_at(Tensor* t, int i);
float* tensor2d_at(Tensor* t, int i, int j);
float* tensor3d_at(Tensor* t, int i, int j, int k);
float* tensor4d_at(Tensor* t, int i, int j, int k, int l);
float  tensor_get_at(Tensor* tensor, int i, int j, int k, int l);  // returns value at element (i,j,k,l) for rank 4, or (i,j,k) for rank 3, etc.
float  tensor1d_get_at(Tensor* t, int i);
float  tensor2d_get_at(Tensor* t, int i, int j);
float  tensor3d_get_at(Tensor* t, int i, int j, int k);
float  tensor4d_get_at(Tensor* t, int i, int j, int k, int l);
void      tensor_set_at(Tensor* tensor, int i, int j, int k, int l, float value);
void      tensor1d_set_at(Tensor* t, int i, float value);
void      tensor2d_set_at(Tensor* t, int i, int j, float value);
void      tensor3d_set_at(Tensor* t, int i, int j, int k, float value);
void      tensor4d_set_at(Tensor* t, int i, int j, int k, int l, float value);
Tensor* tensor_reshaped_view(Tensor* t, int new_rank, int new_n1, int new_n2, int new_n3, int new_n4);  // returns a view of t with the same data but different shape (must be compatible)

// ---- Tensor elementwise operations ----

int  tensor_size(Tensor* t);                        // total element count
void tensor_zero(Tensor* t);                        // zero all elements
void tensor_copy_data(Tensor* dst, Tensor* src);  // copy data (same shape assumed)
void tensor_copy_data_raw(Tensor* dst, float* src);  // copy data from float*
void tensor_add(Tensor* dst, Tensor* src);        // dst += src
void tensor_scale(Tensor* t, float s);              // t *= s
void tensor_scale_add(Tensor* dst, float src_scale, Tensor* src);  // dst += src * src_scale
void tensor_A_plus_cB(Tensor* dst, const Tensor* A, float c, const Tensor* B);  // dst = A + c * B
void tensor_minimum_of(Tensor* out, const Tensor* a, const Tensor* b);  // out[i] = min(a[i], b[i]) (same shape assumed)
void tensor_maximum_of(Tensor* out, const Tensor* a, const Tensor* b);  // out[i] = max(a[i], b[i]) (same shape assumed)

// ===========================================================================
// Matrix operations (row-major float*, explicit dims)
// ===========================================================================

void mat_mul(float* C, const float* A, const float* B, int m, int k, int n);

void mat_mul_bias(float* C, const float* A, const float* B,
                  const float* bias, int m, int k, int n);

void mat_mul_transA_add(float* C, const float* A, const float* B,
                        int m, int k, int n);

void mat_mul_transB(float* C, const float* A, const float* B,
                    int m, int n, int k);

// ===========================================================================
// Vector / activation operations
// ===========================================================================

float vec_dot(const float* x, const float* y, int n);
void vec_relu(float* x, int n);
void vec_relu_backward(float* delta, const float* pre_act, int n);
float vec_dist_squared(const float* x, const float* y, int n) ;
float vec_dist(const float* x, const float* y, int n);
float vec_sum(const float* x, int n);
float vec_mean(const float* x, int n);
float vec_max(const float* x, int n);
float vec_max_abs(const float* x, int n);
// x[i] = (x[i] + shift) * scale    (in-place; works on any contiguous buffer)
void vec_shift_and_scale(float* x, int n, float shift, float scale);

// Population mean and variance of a flat vector.
void vec_mean_and_var(const float* x, int n, float* mean_out, float* var_out);

// Clamp every element to [lo, hi] in-place.
void vec_clamp(float* x, int n, float lo, float hi);


// ===========================================================================
// Aggregation helpers
// ===========================================================================

void mat_col_sum_add(float* out, const float* M, int m, int n);
void mat_row_sum(float* out, const float* M, int m, int n);
void mat_row_weighted_sum(float* out, const float* M, const float* weights, int m, int n, bool same_weights_each_row);
void mat_row_entropy(float* out, const float* probs, int m, int n);  // out[i] = entropy of row i of probs

// Per-column mean and population variance of M[m x n].
void mat_col_mean_and_var(const float* M, int m, int n, float* means, float* vars);

// Per-column in-place: M[i][d] = (M[i][d] + shifts[d]) * scales[d].
void mat_col_shift_and_scale(float* M, int m, int n, const float* shifts, const float* scales);

// Per-row mean and population variance of M[m x n].
void mat_row_mean_and_var(const float* M, int m, int n, float* means, float* vars);

// Per-row in-place: M[i][j] = (M[i][j] + shifts[i]) * scales[i].
void mat_row_shift_and_scale(float* M, int m, int n, const float* shifts, const float* scales);

// ===========================================================================
// Linear system solver
// ===========================================================================

// Solve A*x = b by Gaussian elimination with partial pivoting.
// A (n×n, row-major) and b (n) are modified in place; b gets the solution.
// Returns 0 on success, -1 if the system is (near-)singular.
int mat_solve(float* A, float* b, int n);

// Solve A*X = B for multiple right-hand-side columns (B is n×nrhs, row-major).
// A (n×n) and B (n×nrhs) are modified in place; B gets the solution X.
// Returns 0 on success, -1 if the system is (near-)singular.
int mat_solve_multi(float* A, float* B, int n, int nrhs);
void mat_row_argmax(int* out, const float* M, int m, int n);
void mat_row_max(float* out, const float* M, int m, int n);
void mat_row_softmax(float* out, const float* M, int m, int n, float temperature);
void mat_row_sample(int* out, const float* M, int m, int n);
float* mat_row_slice(const float* M, int row, int n);

// ===========================================================================
// Printing
// ===========================================================================

void tensor_print(Tensor* t, const char* name);

// Concatenation
Tensor* matrix_concat_rows_malloc(Tensor* a, Tensor* b);

// Saving-Loading
void tensor_save(const char* filename, Tensor* t);
Tensor* tensor_load_malloc(const char* filename);

// ===========================================================================
// Utility
// ===========================================================================

// Always double — used by environment dynamics for full precision.
static inline double rand_uniform(double low, double high) {
    return low + ((double)rand() / RAND_MAX) * (high - low);
}

#endif // TENSOR_H
