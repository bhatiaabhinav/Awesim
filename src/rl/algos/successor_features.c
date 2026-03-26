#include "rl/algos/successor_features.h"

#include <stdio.h>
#include <stdlib.h>

void successor_features_compute(
    Tensor* psi,
    Tensor* probs,
    Tensor* T,
    Tensor* Phi,
    Tensor* d0,
    float gamma,
    Tensor* psi_d0_out)
{
    const int S = probs->n1;
    const int A = probs->n2;
    const int F = Phi->n4;

    tensor_assert_shape(probs, S, A, 0, 0);
    tensor_assert_shape(T, S, A, S, 0);
    tensor_assert_shape(Phi, S, A, S, F);
    tensor_assert_shape(d0, S, 0, 0, 0);
    tensor_assert_shape(psi, S, F, 0, 0);
    if (psi_d0_out) {
        tensor_assert_shape(psi_d0_out, F, 0, 0, 0);
    }

    float* P_pi = (float*)calloc((size_t)S * (size_t)S, sizeof(float));
    float* Phi_bar = (float*)calloc((size_t)S * (size_t)F, sizeof(float));
    float* mat_A = (float*)calloc((size_t)S * (size_t)S, sizeof(float));

    if (!P_pi || !Phi_bar || !mat_A) {
        free(P_pi);
        free(Phi_bar);
        free(mat_A);
        fprintf(stderr, "successor_features_compute: allocation failed\n");
        abort();
    }

    for (int s = 0; s < S; s++) {
        for (int a = 0; a < A; a++) {
            const float pi_sa = tensor2d_get_at(probs, s, a);
            if (pi_sa == (float)0.0) {
                continue;
            }
            for (int sp = 0; sp < S; sp++) {
                const float t = tensor3d_get_at(T, s, a, sp);
                const float pi_t = pi_sa * t;
                P_pi[s * S + sp] += pi_t;
                for (int f = 0; f < F; f++) {
                    Phi_bar[s * F + f] += pi_t * tensor4d_get_at(Phi, s, a, sp, f);
                }
            }
        }
    }

    for (int s = 0; s < S; s++) {
        for (int sp = 0; sp < S; sp++) {
            mat_A[s * S + sp] = (s == sp ? (float)1.0 : (float)0.0) - gamma * P_pi[s * S + sp];
        }
    }

    if (mat_solve_multi(mat_A, Phi_bar, S, F) != 0) {
        memset(Phi_bar, 0, sizeof(float) * (size_t)S * (size_t)F);
    }

    memcpy(psi->data, Phi_bar, sizeof(float) * (size_t)S * (size_t)F);

    if (psi_d0_out) {
        memset(psi_d0_out->data, 0, sizeof(float) * (size_t)F);
        for (int s = 0; s < S; s++) {
            const float d = tensor1d_get_at(d0, s);
            if (d == (float)0.0) {
                continue;
            }
            for (int f = 0; f < F; f++) {
                psi_d0_out->data[f] += d * Phi_bar[s * F + f];
            }
        }
    }

    free(P_pi);
    free(Phi_bar);
    free(mat_A);
}
