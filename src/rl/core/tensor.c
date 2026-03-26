#include "rl/core/tensor.h"
#include <stdbool.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

// ===========================================================================
// Tensor
// ===========================================================================

float* tensor1d_at(Tensor* t, int i);
float* tensor2d_at(Tensor* t, int i, int j);
float* tensor3d_at(Tensor* t, int i, int j, int k);
float* tensor4d_at(Tensor* t, int i, int j, int k, int l);

Tensor* tensor_malloc(int rank, int n1, int n2, int n3, int n4  ) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->rank = rank;
    t->n1 = n1;
    t->n2 = n2;
    t->n3 = n3;
    t->n4 = n4;
    int sz = 1;
    if (rank >= 1) sz *= n1;
    if (rank >= 2) sz *= n2;
    if (rank >= 3) sz *= n3;
    if (rank >= 4) sz *= n4;
    t->data = (float*)calloc(sz, sizeof(float));
    return t;
}

Tensor* tensor1d_malloc(int n1) {
    return tensor_malloc(1, n1, 0, 0, 0);
}
Tensor* tensor2d_malloc(int n1, int n2) {
    return tensor_malloc(2, n1, n2, 0, 0);
}
Tensor* tensor3d_malloc(int n1, int n2, int n3) {
    return tensor_malloc(3, n1, n2, n3, 0);
}
Tensor* tensor4d_malloc(int n1, int n2, int n3, int n4) {
    return tensor_malloc(4, n1, n2, n3, n4);
}

void tensor_assert_shape(Tensor* t, int n1, int n2, int n3, int n4) {
    if (!t) {
        fprintf(stderr, "tensor_assert_shape: tensor is NULL\n");
        abort();
    }

    int expected_rank = 0;
    if (n1 > 0) expected_rank = 1;
    if (n2 > 0) expected_rank = 2;
    if (n3 > 0) expected_rank = 3;
    if (n4 > 0) expected_rank = 4;

    if (t->rank != expected_rank || t->n1 != n1 || t->n2 != n2 || t->n3 != n3 || t->n4 != n4) {
        fprintf(stderr,
                "tensor_assert_shape failed: expected rank=%d shape=(%d,%d,%d,%d), got rank=%d shape=(%d,%d,%d,%d)\n",
                expected_rank, n1, n2, n3, n4, t->rank, t->n1, t->n2, t->n3, t->n4);
        abort();
    }
}

void tensor_free(Tensor* t) {
    if (t) { free(t->data); free(t); }
}

void tensors_free(Tensor* tensors[]) {
    if (!tensors) return;
    int i = 0;
    while (tensors[i]) {
        tensor_free(tensors[i++]);
    }
}

float* tensor_at(Tensor* t, int i, int j, int k, int l) {
    if (t->rank == 1) return &t->data[i];
    if (t->rank == 2) return &t->data[i * t->n2 + j];
    if (t->rank == 3) return &t->data[i * (t->n2 * t->n3) + j * t->n3 + k];
    if (t->rank == 4) return &t->data[i * (t->n2 * t->n3 * t->n4) + j * (t->n3 * t->n4) + k * t->n4 + l];
    return NULL;
}

float* tensor1d_at(Tensor* t, int i) {
    return &t->data[i];
}
float* tensor2d_at(Tensor* t, int i, int j) {
    return &t->data[i * t->n2 + j];
}
float* tensor3d_at(Tensor* t, int i, int j, int k) {
    return &t->data[i * (t->n2 * t->n3) + j * t->n3 + k];
}
float* tensor4d_at(Tensor* t, int i, int j, int k, int l) {
    return &t->data[i * (t->n2 * t->n3 * t->n4) + j * (t->n3 * t->n4) + k * t->n4 + l];
}

float tensor_get_at(Tensor* t, int i, int j, int k, int l) {
    return t->data[i * (t->n2 * t->n3 * t->n4) + j * (t->n3 * t->n4) + k * t->n4 + l];
}

float tensor1d_get_at(Tensor* t, int i) {
    return t->data[i];
}

float tensor2d_get_at(Tensor* t, int i, int j) {
    return t->data[i * t->n2 + j];
}

float tensor3d_get_at(Tensor* t, int i, int j, int k) {
    return t->data[i * (t->n2 * t->n3) + j * t->n3 + k];
}

float tensor4d_get_at(Tensor* t, int i, int j, int k, int l) {
    return t->data[i * (t->n2 * t->n3 * t->n4) + j * (t->n3 * t->n4) + k * t->n4 + l];
}

void tensor_set_at(Tensor* t, int i, int j, int k, int l, float val) {
    t->data[i * (t->n2 * t->n3 * t->n4) + j * (t->n3 * t->n4) + k * t->n4 + l] = val;
}

void tensor1d_set_at(Tensor* t, int i, float val) {
    t->data[i] = val;
}

void tensor2d_set_at(Tensor* t, int i, int j, float val) {
    t->data[i * t->n2 + j] = val;
}

void tensor3d_set_at(Tensor* t, int i, int j, int k, float val) {
    t->data[i * (t->n2 * t->n3) + j * t->n3 + k] = val;
}

void tensor4d_set_at(Tensor* t, int i, int j, int k, int l, float val) {
    t->data[i * (t->n2 * t->n3 * t->n4) + j * (t->n3 * t->n4) + k * t->n4 + l] = val;
}

Tensor* tensor_reshaped_view(Tensor* t, int new_rank, int new_n1, int new_n2, int new_n3, int new_n4) {
    int new_sz = 1;
    if (new_rank >= 1) new_sz *= new_n1;
    if (new_rank >= 2) new_sz *= new_n2;
    if (new_rank >= 3) new_sz *= new_n3;
    if (new_rank >= 4) new_sz *= new_n4;

    if (new_sz != tensor_size(t)) {
        fprintf(stderr, "tensor_reshaped_view: incompatible shape change from rank %d (%d,%d,%d,%d) to rank %d (%d,%d,%d,%d)\n",
                t->rank, t->n1, t->n2, t->n3, t->n4,
                new_rank, new_n1, new_n2, new_n3, new_n4);
        abort();
    }

    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    view->data = t->data;  // shared data pointer
    view->rank = new_rank;
    view->n1 = new_n1;
    view->n2 = new_n2;
    view->n3 = new_n3;
    view->n4 = new_n4;
    return view;
}


int tensor_size(Tensor* t) {
    int sz = t->n1;
    if (t->rank >= 2) sz *= t->n2;
    if (t->rank >= 3) sz *= t->n3;
    if (t->rank >= 4) sz *= t->n4;
    return sz;
}

void tensor_zero(Tensor* t) {
    memset(t->data, 0, sizeof(float) * tensor_size(t));
}

void tensor_copy_data(Tensor* dst, Tensor* src) {
    memcpy(dst->data, src->data, sizeof(float) * tensor_size(src));
}

void tensor_copy_data_raw(Tensor* dst, float* src) {
    memcpy(dst->data, src, sizeof(float) * tensor_size(dst));
}

void tensor_add(Tensor* dst, Tensor* src) {
    int n = tensor_size(dst);
    for (int i = 0; i < n; i++) dst->data[i] += src->data[i];
}

void tensor_scale(Tensor* t, float s) {
    int n = tensor_size(t);
    for (int i = 0; i < n; i++) t->data[i] *= s;
}

void tensor_scale_add(Tensor* dst, float src_scale, Tensor* src) {
    int n = tensor_size(dst);
    for (int i = 0; i < n; i++) dst->data[i] += src->data[i] * src_scale;
}

void tensor_A_plus_cB(Tensor* dst, const Tensor* A, float c, const Tensor* B) {
    int n = tensor_size(dst);
    for (int i = 0; i < n; i++) dst->data[i] = A->data[i] + c * B->data[i];
}

void tensor_minimum_of(Tensor* out, const Tensor* a, const Tensor* b) {
    int n = tensor_size(out);
    for (int i = 0; i < n; i++) {
        out->data[i] = fminf(a->data[i], b->data[i]);
    }
}

void tensor_maximum_of(Tensor* out, const Tensor* a, const Tensor* b) {
    int n = tensor_size(out);
    for (int i = 0; i < n; i++) {
        out->data[i] = fmaxf(a->data[i], b->data[i]);
    }
}



Tensor* matrix_concat_rows_malloc(Tensor* a, Tensor* b) {
    Tensor* out = tensor2d_malloc(a->n1 + b->n1, a->n2);
    memcpy(out->data, a->data, a->n1 * a->n2 * sizeof(float));
    memcpy(out->data + a->n1 * a->n2, b->data, b->n1 * b->n2 * sizeof(float));
    return out;
}

// ===========================================================================
// Matrix operations  (all row-major, cache-friendly loop order)
// ===========================================================================

void mat_mul(float* C, const float* A, const float* B, int m, int k, int n) {
    memset(C, 0, sizeof(float) * m * n);
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            const float a = A[i * k + p];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[p * n + j];
        }
}

void mat_mul_bias(float* C, const float* A, const float* B,
                  const float* bias, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        memcpy(&C[i * n], bias, sizeof(float) * n);
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            const float a = A[i * k + p];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[p * n + j];
        }
}

void mat_mul_transA_add(float* C, const float* A, const float* B,
                        int m, int k, int n) {
    for (int p = 0; p < m; p++)
        for (int i = 0; i < k; i++) {
            const float a = A[p * k + i];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[p * n + j];
        }
}

void mat_mul_transB(float* C, const float* A, const float* B,
                    int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) {
            float s = 0;
            for (int p = 0; p < n; p++)
                s += A[i * n + p] * B[j * n + p];
            C[i * k + j] = s;
        }
}

float vec_dot(const float* x, const float* y, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i] * y[i];
    return sum;
}

float vec_dist_squared(const float* x, const float* y, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float d = x[i] - y[i];
        sum += d * d;
    }
    return sum;
}

float vec_dist(const float* x, const float* y, int n) {
    return sqrtf(vec_dist_squared(x, y, n));
}


// ===========================================================================
// Activation / vector ops
// ===========================================================================

void vec_relu(float* x, int n) {
    for (int i = 0; i < n; i++)
        if (x[i] < 0) x[i] = 0;
}

void vec_relu_backward(float* delta, const float* pre_act, int n) {
    for (int i = 0; i < n; i++)
        if (pre_act[i] <= 0) delta[i] = 0;
}

void vec_shift_and_scale(float* x, int n, float shift, float scale) {
    for (int i = 0; i < n; i++)
        x[i] = (x[i] + shift) * scale;
}

float vec_sum(const float* x, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i];
    return sum;
}

float vec_mean(const float* x, int n) {
    return vec_sum(x, n) / n;
}

float vec_max(const float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];
    return max_val;
}

float vec_max_abs(const float* x, int n) {
    float max_val = fabsf(x[0]);
    for (int i = 1; i < n; i++) {
        float abs_val = fabsf(x[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    return max_val;
}

void vec_mean_and_var(const float* x, int n, float* mean_out, float* var_out) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i];
    float mean = sum / n;
    float vsum = 0;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        vsum += d * d;
    }
    *mean_out = mean;
    *var_out  = vsum / n;
}

void vec_clamp(float* x, int n, float lo, float hi) {
    for (int i = 0; i < n; i++) {
        if (x[i] < lo) x[i] = lo;
        if (x[i] > hi) x[i] = hi;
    }
}

// ===========================================================================
// Aggregation
// ===========================================================================

void mat_col_sum_add(float* out, const float* M, int m, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            out[j] += M[i * n + j];
}

void mat_row_sum(float* out, const float* M, int m, int n) {
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++)
            sum += M[i * n + j];
        out[i] = sum;
    }
}

void mat_row_weighted_sum(float* out, const float* M, const float* weights, int m, int n, bool same_weights_each_row) {
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++)
            sum += M[i * n + j] * (same_weights_each_row ? weights[j] : weights[i * n + j]);
        out[i] = sum;
    }
}

void mat_row_entropy(float* out, const float* probs, int m, int n) {
    for (int i = 0; i < m; i++) {
        float entropy = 0;
        for (int j = 0; j < n; j++) {
            float p = probs[i * n + j];
            if (p > 0) entropy -= p * logf(p);
        }
        out[i] = entropy;
    }
}

void mat_col_mean_and_var(const float* M, int m, int n, float* means, float* vars) {
    memset(means, 0, sizeof(float) * n);
    for (int i = 0; i < m; i++)
        for (int d = 0; d < n; d++)
            means[d] += M[i * n + d];
    for (int d = 0; d < n; d++) means[d] /= m;
    memset(vars, 0, sizeof(float) * n);
    for (int i = 0; i < m; i++)
        for (int d = 0; d < n; d++) {
            float diff = M[i * n + d] - means[d];
            vars[d] += diff * diff;
        }
    for (int d = 0; d < n; d++) vars[d] /= m;
}

void mat_col_shift_and_scale(float* M, int m, int n, const float* shifts, const float* scales) {
    for (int i = 0; i < m; i++)
        for (int d = 0; d < n; d++)
            M[i * n + d] = (M[i * n + d] + shifts[d]) * scales[d];
}

void mat_row_mean_and_var(const float* M, int m, int n, float* means, float* vars) {
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) sum += M[i * n + j];
        float mean = sum / n;
        float vsum = 0;
        for (int j = 0; j < n; j++) {
            float d = M[i * n + j] - mean;
            vsum += d * d;
        }
        means[i] = mean;
        vars[i]  = vsum / n;
    }
}

void mat_row_shift_and_scale(float* M, int m, int n, const float* shifts, const float* scales) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            M[i * n + j] = (M[i * n + j] + shifts[i]) * scales[i];
}

void mat_row_argmax(int* out, const float* M, int m, int n) {
    for (int i = 0; i < m; i++) {
        int best = 0;
        float best_val = M[i * n];
        for (int j = 1; j < n; j++) {
            if (M[i * n + j] > best_val) {
                best_val = M[i * n + j];
                best = j;
            }
        }
        out[i] = best;
    }
}

void mat_row_max(float* out, const float* M, int m, int n) {
    for (int i = 0; i < m; i++) {
        float best_val = M[i * n];
        for (int j = 1; j < n; j++) {
            if (M[i * n + j] > best_val) {
                best_val = M[i * n + j];
            }
        }
        out[i] = best_val;
    }
}

void mat_row_softmax(float* out, const float* M, int m, int n, float temperature) {
    for (int i = 0; i < m; i++) {
        const float* row_in  = M   + i * n;
        float*       row_out = out + i * n;

        /* pass 1: find row max for numerical stability */
        float row_max = row_in[0];
        for (int j = 1; j < n; j++)
            row_max = fmaxf(row_max, row_in[j]);

        if (temperature <= 0) {
            /* deterministic argmax: put all probability on the first max element */
            int best = 0;
            for (int j = 1; j < n; j++)
                if (row_in[j] > row_in[best]) best = j;
            for (int j = 0; j < n; j++)
                row_out[j] = (j == best) ? (float)1.0 : (float)0.0;
            continue;
        }

        const float inv_temp = (float)1.0 / temperature;

        /* pass 2: exponentiate with temperature scaling, accumulate sum */
        float sum = 0;
        for (int j = 0; j < n; j++) {
            float e = expf((row_in[j] - row_max) * inv_temp);
            row_out[j] = e;
            sum += e;
        }

        /* pass 3: normalise */
        const float inv_sum = (float)1.0 / sum;
        for (int j = 0; j < n; j++)
            row_out[j] *= inv_sum;
    }
}

void mat_row_sample(int* out, const float* M, int m, int n) {
    for (int i = 0; i < m; i++) {
        const float* row = M + i * n;
        float r = (float)rand() / (float)RAND_MAX;

        /* linear scan through CDF (sequential access, branch-free friendly) */
        float cumsum = 0;
        int chosen = n - 1;            /* fallback to last action */
        for (int j = 0; j < n; j++) {
            cumsum += row[j];
            if (r <= cumsum) {
                chosen = j;
                break;
            }
        }
        out[i] = chosen;
    }
}

float* mat_row_slice(const float* M, int row, int n) {
    return (float*)&M[row * n];
}

// ===========================================================================
// Linear system solver
// ===========================================================================

// Solve A*x = b by Gaussian elimination with partial pivoting.
// A is n×n row-major (modified in place), b is length-n (overwritten with x).
// Returns 0 on success, -1 if the system is (near-)singular.
int mat_solve(float* A, float* b, int n) {
    for (int col = 0; col < n; col++) {
        // Partial pivot: find row with largest absolute value in this column
        int pivot = col;
        float max_val = fabsf(A[col * n + col]);
        for (int row = col + 1; row < n; row++) {
            float val = fabsf(A[row * n + col]);
            if (val > max_val) { max_val = val; pivot = row; }
        }
        if (max_val < (float)1e-12) return -1; // singular

        // Swap rows col <-> pivot
        if (pivot != col) {
            for (int j = 0; j < n; j++) {
                float tmp = A[col * n + j];
                A[col * n + j]   = A[pivot * n + j];
                A[pivot * n + j] = tmp;
            }
            float tmp = b[col]; b[col] = b[pivot]; b[pivot] = tmp;
        }

        // Eliminate below
        float diag_inv = (float)1.0 / A[col * n + col];
        for (int row = col + 1; row < n; row++) {
            float factor = A[row * n + col] * diag_inv;
            A[row * n + col] = 0;
            for (int j = col + 1; j < n; j++)
                A[row * n + j] -= factor * A[col * n + j];
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution
    for (int row = n - 1; row >= 0; row--) {
        float sum = b[row];
        for (int j = row + 1; j < n; j++)
            sum -= A[row * n + j] * b[j];
        b[row] = sum / A[row * n + row];
    }
    return 0;
}

int mat_solve_multi(float* A, float* B, int n, int nrhs) {
    /* Gaussian elimination with partial pivoting; B is n × nrhs, row-major. */
    for (int col = 0; col < n; col++) {
        int pivot = col;
        float max_val = fabsf(A[col * n + col]);
        for (int row = col + 1; row < n; row++) {
            float val = fabsf(A[row * n + col]);
            if (val > max_val) { max_val = val; pivot = row; }
        }
        if (max_val < (float)1e-12) return -1;
        if (pivot != col) {
            for (int j = 0; j < n; j++) {
                float tmp = A[col * n + j]; A[col * n + j] = A[pivot * n + j]; A[pivot * n + j] = tmp;
            }
            for (int j = 0; j < nrhs; j++) {
                float tmp = B[col * nrhs + j]; B[col * nrhs + j] = B[pivot * nrhs + j]; B[pivot * nrhs + j] = tmp;
            }
        }
        float diag_inv = (float)1.0 / A[col * n + col];
        for (int row = col + 1; row < n; row++) {
            float factor = A[row * n + col] * diag_inv;
            A[row * n + col] = 0;
            for (int j = col + 1; j < n; j++)
                A[row * n + j] -= factor * A[col * n + j];
            for (int j = 0; j < nrhs; j++)
                B[row * nrhs + j] -= factor * B[col * nrhs + j];
        }
    }
    for (int row = n - 1; row >= 0; row--) {
        float diag_inv = (float)1.0 / A[row * n + row];
        for (int j = 0; j < nrhs; j++) {
            float sum = B[row * nrhs + j];
            for (int k = row + 1; k < n; k++)
                sum -= A[row * n + k] * B[k * nrhs + j];
            B[row * nrhs + j] = sum * diag_inv;
        }
    }
    return 0;
}

void tensor_print(Tensor* t, const char* name) {
    if (name) printf("%s", name);
    if (t->rank == 1) {
        if (name) printf(" (%d)", t->n1);
        printf("\n");
        printf("[");
        for (int i = 0; i < t->n1; i++) {
            printf("%8.4f", (double)tensor1d_get_at(t, i));
            if (i < t->n1 - 1) printf(", ");
        }
        printf("]\n");
    } else if (t->rank == 2) {
        if (name) printf(" (%d x %d)", t->n1, t->n2);
        printf("\n");
        for (int i = 0; i < t->n1; i++) {
            printf("  [");
            for (int j = 0; j < t->n2; j++) {
                printf("%8.4f", (double)tensor2d_get_at(t, i, j));
                if (j < t->n2 - 1) printf(", ");
            }
            printf("]\n");
        }
    } else if (t->rank == 3) {
        if (name) printf(" (%d x %d x %d)", t->n1, t->n2, t->n3);
        printf("\n");
        for (int i = 0; i < t->n1; i++) {
            printf("  [%d]:\n", i);
            for (int j = 0; j < t->n2; j++) {
                printf("    [");
                for (int k = 0; k < t->n3; k++) {
                    printf("%8.4f", (double)tensor3d_get_at(t, i, j, k));
                    if (k < t->n3 - 1) printf(", ");
                }
                printf("]\n");
            }
        }
    }
}
// ===========================================================================
// Saving / Loading
// ===========================================================================

static void mkdirs(const char* filepath) {
    char tmp[4096];
    size_t len = strlen(filepath);
    if (len >= sizeof(tmp)) return;
    memcpy(tmp, filepath, len + 1);
    for (size_t i = 1; i < len; i++) {
        if (tmp[i] == '/' || tmp[i] == '\\') {
            tmp[i] = '\0';
            MKDIR(tmp);
            tmp[i] = filepath[i];
        }
    }
}

static bool ends_with_csv(const char* filename) {
    size_t len = strlen(filename);
    return len >= 4 && strcmp(filename + len - 4, ".csv") == 0;
}

void tensor_save(const char* filename, Tensor* t) {
    mkdirs(filename);
    if (ends_with_csv(filename)) {
        // CSV format – rank-2 only
        if (t->rank != 2) {
            fprintf(stderr, "tensor_save: CSV format only supported for rank-2 tensors\n");
            return;
        }
        FILE* f = fopen(filename, "w");
        if (!f) { perror("tensor_save"); return; }
        for (int i = 0; i < t->n1; i++) {
            for (int j = 0; j < t->n2; j++) {
                if (j > 0) fputc(',', f);
                fprintf(f, "%.17g", (double)tensor2d_get_at(t, i, j));
            }
            fputc('\n', f);
        }
        fclose(f);
    } else {
        // Binary format: rank, n1, n2, n3, then raw data
        FILE* f = fopen(filename, "wb");
        if (!f) { perror("tensor_save"); return; }
        int header[5] = { t->rank, t->n1, t->n2, t->n3, t->n4 };
        fwrite(header, sizeof(int), 5, f);
        fwrite(t->data, sizeof(float), tensor_size(t), f);
        fclose(f);
    }
}

Tensor* tensor_load_malloc(const char* filename) {
    if (ends_with_csv(filename)) {
        // First pass: count rows and columns
        FILE* f = fopen(filename, "r");
        if (!f) { perror("tensor_load_malloc"); return NULL; }
        int rows = 0, cols = 0;
        char line[1 << 16];
        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '\n' || line[0] == '\0') continue;
            rows++;
            if (rows == 1) {
                cols = 1;
                for (char* p = line; *p; p++)
                    if (*p == ',') cols++;
            }
        }
        rewind(f);

        Tensor* t = tensor2d_malloc(rows, cols);
        int r = 0;
        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '\n' || line[0] == '\0') continue;
            char* tok = line;
            for (int c = 0; c < cols; c++) {
                tensor2d_set_at(t, r, c, (float)strtod(tok, &tok));
                if (*tok == ',') tok++;
            }
            r++;
        }
        fclose(f);
        return t;
    } else {
        FILE* f = fopen(filename, "rb");
        if (!f) { perror("tensor_load_malloc"); return NULL; }
        int header[5];
        if (fread(header, sizeof(int), 5, f) != 5) {
            fclose(f); return NULL;
        }
        Tensor* t = tensor_malloc(header[0], header[1], header[2], header[3], header[4]);
        if (fread(t->data, sizeof(float), tensor_size(t), f) != (size_t)tensor_size(t)) {
            tensor_free(t); fclose(f); return NULL;
        }
        fclose(f);
        return t;
    }
}