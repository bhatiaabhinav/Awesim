#ifndef RL_ALGOS_SUCCESSOR_FEATURES_H
#define RL_ALGOS_SUCCESSOR_FEATURES_H

#include "rl/core/tensor.h"

// Compute successor features for a fixed policy by solving:
//   (I - gamma * P_pi) * psi = phi_bar
//
// Shapes:
//   psi         : (S, F)      output
//   probs       : (S, A)      policy probabilities
//   T           : (S, A, S)   transition probabilities
//   Phi         : (S, A, S, F) one-step feature tensor
//   d0          : (S)         start-state distribution
//   psi_d0_out  : (F)         optional output for sum_s d0[s] * psi[s, :]
void successor_features_compute(
    Tensor* psi,
    Tensor* probs,
    Tensor* T,
    Tensor* Phi,
    Tensor* d0,
    float gamma,
    Tensor* psi_d0_out
);

#endif // RL_ALGOS_SUCCESSOR_FEATURES_H
