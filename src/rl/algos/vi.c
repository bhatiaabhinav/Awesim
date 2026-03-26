#include "rl/core/mdp.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// #define VI_DEBUG

#ifdef VI_DEBUG
#define VI_DEBUG_PRINTF(...) printf(__VA_ARGS__)
#else
#define VI_DEBUG_PRINTF(...) do { if (0) { printf(__VA_ARGS__); } } while(0)
#endif

// Forward declaration (defined below)
float policy_evaluation(Tensor* probs_matrix, Tensor* Q, Tensor* V, Tensor* T, Tensor* R, Tensor* d0, float gamma, float tol, int max_iters);

// ---------------------------------------------------------------------------
// Define PE_USE_LINEAR_ALGEBRA to solve policy evaluation via matrix inversion
// (I - γ Pπ) V = Rπ  — exact, infinite-horizon, O(S³).
// Undefine it to fall back to the iterative Bellman backup.
// ---------------------------------------------------------------------------
#define PE_USE_LINEAR_ALGEBRA

static float bellman_backup_action_values_synchronous(Tensor* Q, Tensor* V, Tensor* T, Tensor* R, float gamma) {
    const int num_states = Q->n1;
    const int num_actions = Q->n2;
    float max_diff = 0;
    for (int s = 0; s < num_states; s++) {
        for (int a = 0; a < num_actions; a++) {
            float q_new = 0;
            for (int s_next = 0; s_next < num_states; s_next++) {
                q_new += tensor3d_get_at(T, s, a, s_next) * (tensor3d_get_at(R, s, a, s_next) + gamma * tensor1d_get_at(V, s_next));
            }
            float diff = fabsf(q_new - tensor2d_get_at(Q, s, a));
            if (diff > max_diff) max_diff = diff;
            tensor2d_set_at(Q, s, a, q_new);
        }
    }
    return max_diff;
}

float value_iteration(Tensor* Q, Tensor* V, Tensor* T, Tensor* R, Tensor* d0, float gamma, float tol, int max_iters) {
    const int num_states = Q->n1;
    const int num_actions = Q->n2;

    // Initialize V and Q to zero
    memset(V->data, 0, sizeof(float) * num_states);
    memset(Q->data, 0, sizeof(float) * num_states * num_actions);

    for (int iter = 0; iter < max_iters; iter++) {
        // Backup action values into Q
        float max_diff = bellman_backup_action_values_synchronous(Q, V, T, R, gamma);
        mat_row_max(V->data, Q->data, num_states, num_actions);
        if (max_diff < tol) break;
    }
    bellman_backup_action_values_synchronous(Q, V, T, R, gamma); // make Q consistent with final V
    float v0 = 0;
    mat_row_weighted_sum(&v0, V->data, d0->data, 1, num_states, true); // v0 = sum_s d0[s] * V[s]
    return v0;
}

float value_iteration_soft(Tensor* pi, Tensor* Q, Tensor* V, Tensor* T, Tensor* R, Tensor* d0, float gamma, float temperature, float tol, int max_iters) {

    float v0 = 0;
    if (temperature <= 0) {
        // Degenerate case: zero temperature → greedy policy
        v0 = value_iteration(Q, V, T, R, d0, gamma, tol, max_iters);
        mat_row_softmax(pi->data, Q->data, Q->n1, Q->n2, 0.0f); // extract deterministic policy
        return v0;
    }

    const int num_states = Q->n1;
    const int num_actions = Q->n2;
    Tensor* v_prev = tensor1d_malloc(num_states);
    Tensor* H = tensor1d_malloc(num_states);

    // Initialize V and Q to zero
    memset(V->data, 0, sizeof(float) * num_states);
    memset(Q->data, 0, sizeof(float) * num_states * num_actions);
    memset(pi->data, 0, sizeof(float) * num_states * num_actions);

    for (int iter = 0; iter < max_iters; iter++) {
        // Backup action values into Q
        bellman_backup_action_values_synchronous(Q, V, T, R, gamma);
        
        // Softmax policy with temperature
        mat_row_softmax(pi->data, Q->data, num_states, num_actions, temperature);

        // Value function = policy-weighted Q + entropy bonus
        mat_row_weighted_sum(V->data, Q->data, pi->data, num_states, num_actions, false);
        mat_row_entropy(H->data, pi->data, num_states, num_actions);
        tensor_scale_add(V, temperature, H); // V += τ H

        // Check convergence in V: max_s |V_new[s] - V_old[s]|
        tensor_scale_add(v_prev, -1.0f, V); // v_prev = V - v_prev
        float max_diff = vec_max_abs(v_prev->data, num_states);
        if (max_diff < tol) break;
        tensor_copy_data(v_prev, V);
    }
    tensor_free(v_prev);
    tensor_free(H);
    v0 = policy_evaluation(pi, Q, V, T, R, d0, gamma, tol, max_iters);
    return v0;
}

// float value_iteration_soft(Tensor* pi, Tensor* Q, Tensor* V, Tensor* T, Tensor* R, Tensor* d0, float gamma, float temperature, float tol, int max_iters) {

//     float v0 = 0;
//     if (temperature <= 0) {
//         // Degenerate case: zero temperature → greedy policy
//         v0 = value_iteration(Q, V, T, R, d0, gamma, tol, max_iters);
//         mat_row_softmax(pi->data, Q->data, Q->n1, Q->n2, 0.0f); // extract deterministic policy
//         return v0;
//     }

//     const int S = Q->n1;
//     const int A = Q->n2;

//     // Adaptive temperature: target average policy entropy = 0.5 * ln(A).
//     // Each iteration, binary-search for the τ that achieves this target
//     // (entropy of Boltzmann(Q/τ) is monotone increasing in τ).
//     const float H_target = 0.9f * logf((float)A);
//     float tau = temperature;  // starting value; adapted below

//     // Initialize V and Q to zero
//     memset(V->data, 0, sizeof(float) * S);
//     memset(Q->data, 0, sizeof(float) * S * A);
//     memset(pi->data, 0, sizeof(float) * S * A);

//     for (int iter = 0; iter < max_iters; iter++) {
//         // Backup action values into Q
//         float max_diff = bellman_backup_action_values_synchronous(Q, V, T, R, gamma);

//         // Binary search for τ giving avg entropy ≈ H_target (in log-space)
//         float log_lo = -18.0f, log_hi = 5.0f;  // τ ∈ [~1.5e-8, ~148]
//         for (int bs = 0; bs < 20; bs++) {
//             float log_mid = 0.5f * (log_lo + log_hi);
//             mat_row_softmax(pi->data, Q->data, S, A, expf(log_mid));
//             float H_avg = 0;
//             for (int s = 0; s < S; s++)
//                 for (int a = 0; a < A; a++) {
//                     float p = pi->data[s * A + a];
//                     if (p > 0) H_avg -= p * logf(p);
//                 }
//             H_avg /= S;
//             if (H_avg < H_target) log_lo = log_mid;
//             else                  log_hi = log_mid;
//         }
//         tau = expf(0.5f * (log_lo + log_hi));

//         mat_row_softmax(pi->data, Q->data, S, A, tau);
//         // Soft Bellman: V(s) = Σ_a π(a|s) Q(s,a) + τ H(π(·|s))
//         mat_row_weighted_sum(V->data, Q->data, pi->data, S, A, false);
//         for (int s = 0; s < S; s++) {
//             float entropy = 0;
//             for (int a = 0; a < A; a++) {
//                 float p = pi->data[s * A + a];
//                 if (p > 0) entropy -= p * logf(p);
//             }
//             V->data[s] += tau * entropy;
//         }
//         if (max_diff < tol) break;
//     }
//     // Extract final soft policy, then get standard (non-entropy) V^π and Q^π
//     // via exact policy evaluation — callers need these for ε bound calculations.
//     mat_row_softmax(pi->data, Q->data, S, A, tau);
//     v0 = policy_evaluation(pi, Q, V, T, R, d0, gamma, tol, max_iters);
//     return v0;
// }

// ---------------------------------------------------------------------------
// Linear-algebra (matrix-inversion) policy evaluation.
//
// Solves the exact infinite-horizon Bellman equation in closed form:
//
//   V = (I - γ Pπ)⁻¹ Rπ
//
// where
//   Rπ[s]    = Σ_{a,s'} π(a|s) T(s,a,s') R(s,a,s')   (expected reward)
//   Pπ[s,s'] = Σ_a  π(a|s) T(s,a,s')                 (induced transition)
//
// After solving for V, Q is computed from the Bellman equation:
//   Q[s,a] = Σ_{s'} T(s,a,s') [ R(s,a,s') + γ V[s'] ]
//
// Complexity: O(S²·A) to build Rπ/Pπ, O(S³) for Gaussian elimination.
// ---------------------------------------------------------------------------
static float policy_evaluation_linalg(
        Tensor* probs_matrix, Tensor* Q, Tensor* V,
        Tensor* T, Tensor* R, Tensor* d0, float gamma) {
    const int S = Q->n1;
    const int A = Q->n2;

    float* R_pi  = (float*)calloc((size_t)S,   sizeof(float)); // Rπ, then V
    float* P_pi  = (float*)calloc((size_t)S*S, sizeof(float)); // Pπ
    float* mat_A = (float*)calloc((size_t)S*S, sizeof(float)); // (I - γ Pπ)

    // Build Rπ and Pπ
    for (int s = 0; s < S; s++) {
        for (int a = 0; a < A; a++) {
            float pi_sa = tensor2d_get_at(probs_matrix, s, a);
            if (pi_sa == (float)0.0) continue;
            for (int sp = 0; sp < S; sp++) {
                float t = tensor3d_get_at(T, s, a, sp);
                R_pi[s]         += pi_sa * t * tensor3d_get_at(R, s, a, sp);
                P_pi[s * S + sp] += pi_sa * t;
            }
        }
    }

    // Build (I - γ Pπ)
    for (int s = 0; s < S; s++)
        for (int sp = 0; sp < S; sp++)
            mat_A[s * S + sp] = (s == sp ? (float)1.0 : (float)0.0)
                                - gamma * P_pi[s * S + sp];

    // Solve (I - γ Pπ) V = Rπ  →  result in R_pi
    if (mat_solve(mat_A, R_pi, S) != 0) {
        // Fallback: zero V on singular system (should not happen for γ < 1)
        memset(R_pi, 0, sizeof(float) * (size_t)S);
    }

    // Copy solution into V
    memcpy(V->data, R_pi, sizeof(float) * (size_t)S);

    // Compute Q[s,a] = Σ_{s'} T(s,a,s') [R(s,a,s') + γ V[s']]
    for (int s = 0; s < S; s++) {
        for (int a = 0; a < A; a++) {
            float q = 0;
            for (int sp = 0; sp < S; sp++)
                q += tensor3d_get_at(T, s, a, sp)
                   * (tensor3d_get_at(R, s, a, sp) + gamma * V->data[sp]);
            tensor2d_set_at(Q, s, a, q);
        }
    }

    free(R_pi); free(P_pi); free(mat_A);

    float v0 = 0;
    mat_row_weighted_sum(&v0, V->data, d0->data, 1, S, true);
    return v0;
}

float policy_evaluation(Tensor* probs_matrix, Tensor* Q, Tensor* V, Tensor* T, Tensor* R, Tensor* d0, float gamma, float tol, int max_iters) {
#ifdef PE_USE_LINEAR_ALGEBRA
    // Exact infinite-horizon solver via (I - γ Pπ)⁻¹ Rπ
    (void)tol; (void)max_iters; // unused by the direct solver
    return policy_evaluation_linalg(probs_matrix, Q, V, T, R, d0, gamma);
#else
    const int num_states = Q->n1;
    const int num_actions = Q->n2;

    // Initialize V and Q to zero
    memset(V->data, 0, sizeof(float) * num_states);
    memset(Q->data, 0, sizeof(float) * num_states * num_actions);

    for (int iter = 0; iter < max_iters; iter++) {
        // Backup action values into Q using the given policy probabilities
        float max_diff = bellman_backup_action_values_synchronous(Q, V, T, R, gamma);
        // sum over actions weighted by policy probabilities to get new V
        mat_row_weighted_sum(V->data, Q->data, probs_matrix->data, num_states, num_actions, false);
        if (max_diff < tol) break;
    }
    float v0 = 0;
    mat_row_weighted_sum(&v0, V->data, d0->data, 1, num_states, true); // v0 = sum_s d0[s] * V[s]
    return v0;
#endif
}

float policy_evaluation_cost(Tensor* probs_matrix, Tensor* Q_cost, Tensor* V_cost, Tensor* T, Tensor* C, Tensor* d0, float cost_gamma, float tol, int max_iters) {
    return policy_evaluation(probs_matrix, Q_cost, V_cost, T, C, d0, cost_gamma, tol, max_iters);
}

float policy_evaluation_secondary_reward(Tensor* probs_matrix, Tensor* Q_sr, Tensor* V_sr, Tensor* T, Tensor* SR, Tensor* d0, float sr_gamma, float tol, int max_iters) {
    return policy_evaluation(probs_matrix, Q_sr, V_sr, T, SR, d0, sr_gamma, tol, max_iters);
}

// ===========================================================================
// Lexicographic Value Iteration (primary reward > secondary reward with slack)
//
// Adapts the LVI algorithm (Wray et al. AAAI 2015) for 2 objectives:
//   objective 0 = primary reward R  (highest priority)
//   objective 1 = secondary reward SR (lower priority)
//
// 1. Run VI on primary reward using all actions.
// 2. Filter acceptable actions: keep a where Q*(s,a) >= maxQ*(s) - eta,
//    with eta = (1 - gamma) * slack   (converts value slack to Q threshold).
// 3. Run VI on secondary reward using only acceptable actions.
// 4. Extract policy: argmax of secondary Q among acceptable actions.
// ===========================================================================

// Bellman backup restricted to acceptable actions.
// Q[s,a] is updated only for acceptable actions; others are set to -1e30.
// V[s] = max over acceptable actions of Q[s,a].
// Returns max |Q_new - Q_old| over all (s, acceptable a).
static float bellman_backup_masked(Tensor* Q, Tensor* V, Tensor* T, Tensor* Reward,
                                      float gamma_val, const bool* acceptable, int S, int A) {
    float max_diff = 0;
    for (int s = 0; s < S; s++) {
        float best_q = -1e30f;
        for (int a = 0; a < A; a++) {
            if (!acceptable[s * A + a]) {
                tensor2d_set_at(Q, s, a, -1e30f);
                continue;
            }
            float q_new = 0;
            for (int sp = 0; sp < S; sp++) {
                q_new += tensor3d_get_at(T, s, a, sp)
                       * (tensor3d_get_at(Reward, s, a, sp) + gamma_val * tensor1d_get_at(V, sp));
            }
            float diff = fabsf(q_new - tensor2d_get_at(Q, s, a));
            if (diff > max_diff) max_diff = diff;
            tensor2d_set_at(Q, s, a, q_new);
            if (q_new > best_q) best_q = q_new;
        }
        tensor1d_set_at(V, s, (best_q > -1e29f) ? best_q : 0.0f);
    }
    return max_diff;
}

float lexicographic_value_iteration(
    Tensor* policy_probs,
    Tensor* Q_primary, Tensor* V_primary,
    Tensor* Q_sr, Tensor* V_sr,
    Tensor* T, Tensor* R, Tensor* SR,
    Tensor* d0,
    float gamma, float sr_gamma, float slack,
    float tol, int max_iters,
    float* out_primary_d0
) {
    const int S = Q_primary->n1;
    const int A = Q_primary->n2;

    // Acceptable action mask (starts with all actions acceptable)
    bool* acceptable = (bool*)malloc((size_t)S * A * sizeof(bool));
    for (int i = 0; i < S * A; i++) acceptable[i] = true;

    // ── Objective 0: VI on primary reward (all actions) ──
    memset(V_primary->data, 0, sizeof(float) * S);
    memset(Q_primary->data, 0, sizeof(float) * S * A);
    (void)value_iteration(Q_primary, V_primary, T, R, d0, gamma, tol, max_iters);

    float eta = (1.0f - gamma) * slack;   // Wray bound (guaranteed feasible, but may be very conservative)

    // ── Final pass with largest feasible eta ──
    for (int s = 0; s < S; s++)
        for (int a = 0; a < A; a++)
            acceptable[s * A + a] = (Q_primary->data[s * A + a] >= V_primary->data[s] - eta - 1e-9f);

    memset(V_sr->data, 0, sizeof(float) * S);
    memset(Q_sr->data, 0, sizeof(float) * S * A);
    for (int iter = 0; iter < max_iters; iter++) {
        float md = bellman_backup_masked(Q_sr, V_sr, T, SR, sr_gamma, acceptable, S, A);
        if (md < tol) break;
    }
    mat_row_softmax(policy_probs->data, Q_sr->data, S, A, 0.0f);

    // ── Evaluate lex policy on primary reward ──
    memset(Q_primary->data, 0, sizeof(float) * S * A);
    memset(V_primary->data, 0, sizeof(float) * S);
    float v0 = policy_evaluation(policy_probs, Q_primary, V_primary, T, R, d0, gamma, tol, max_iters);
    if (out_primary_d0) *out_primary_d0 = v0;

    free(acceptable);

    float sr_d0 = 0;
    mat_row_weighted_sum(&sr_d0, V_sr->data, d0->data, 1, S, true);
    return sr_d0;
}

// ===========================================================================
// Cost-Constrained Value Iteration (CMDP via Lagrangian relaxation)
// ===========================================================================

#include <math.h>
#include <stdio.h>

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Boltzmann (softmax) policy from Q-values with temperature τ
// ─────────────────────────────────────────────────────────────────────────────
void compute_boltzmann_policy(Tensor* probs, Tensor* Q, float temp) {
    const int S = probs->n1;
    const int A = probs->n2;
    for (int s = 0; s < S; s++) {
        float maxq = -1e30f;
        for (int a = 0; a < A; a++)
            if (Q->data[s * A + a] > maxq) maxq = Q->data[s * A + a];

        float sum = 0.0f;
        for (int a = 0; a < A; a++) {
            float v = expf((Q->data[s * A + a] - maxq) / (temp + 1e-8f));
            probs->data[s * A + a] = v;
            sum += v;
        }
        for (int a = 0; a < A; a++)
            probs->data[s * A + a] /= sum;
    }
}

/*
// ─────────────────────────────────────────────────────────────────────────────
// Legacy grid-search CMDP solver — replaced by bisection/gradient versions.
// Delete candidate; kept commented out for reference.
// ─────────────────────────────────────────────────────────────────────────────
*/

#ifdef CMDP_DUAL_OPT_USE_GRADIENTS
// ── Adam gradient-based version: dual gradient ascent on λ with tau annealing ──
static float cmdp_vi_gradient_based(
    Tensor* π,       // output: (S, A) stochastic policy
    Tensor* Q, Tensor* V,
    Tensor* Q_c, Tensor* V_c,
    Tensor* T, Tensor* R, Tensor* C,
    Tensor* d0,
    float γ,
    float γ_c,
    float cost_threshold,
    int max_lambda_iters,
    float vi_tol,
    int max_vi_iters,
    float* out_cost_d0
) {
    int nstates = Q->n1, nactions = Q->n2;
    
    float tau = 1.0f;
    float λ = 0.0f;
    float η_λ = 0.1f;
    float lam_prev = 0.0f;

    // Adam optimizer state for λ
    float adam_m = 0.0f;       // first moment
    float adam_v = 0.0f;       // second moment
    const float adam_beta1 = 0.9f;
    const float adam_beta2 = 0.999f;
    const float adam_eps = 1e-8f;

    Tensor* R_L = tensor3d_malloc(nstates, nactions, nstates);
    const int nelements_rew_fn = nstates * nactions * nstates;
    Tensor* π_best = tensor2d_malloc(nstates, nactions);
    float v_d0_best = -1e30f;
    float v_d0_best_with_prev_tau = -1e30f;
    float tau_best_overall = tau;
    float lam_best_overall = λ;

    float v_d0 = 0.0f, v_c_d0 = 0.0f;
    float v_d0_prev = 0.0f, v_c_d0_prev = 0.0f;
    int iters_since_tau_change = 0;

    for (int k = 0; k < max_lambda_iters; k++) {

        // create combined reward function:
        for (int i = 0; i < nelements_rew_fn; i++) {
            R_L->data[i] = R->data[i] - λ * C->data[i];
        }

        v_d0_prev = v_d0;
        v_c_d0_prev = v_c_d0;
        value_iteration_soft(π, Q, V, T, R_L, d0, γ, tau, vi_tol, max_vi_iters);
        v_d0 = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);
        v_c_d0 = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);

        // Detect extreme oscillations: large V change AND cost swung far from constraint boundary
        if (iters_since_tau_change >= 5) {
            float v_min = fminf(fabsf(v_d0), fabsf(v_d0_prev)) + 1e-8f;
            float v_change = fabsf(v_d0 - v_d0_prev) / v_min;
            float c_dist_prev = fabsf(v_c_d0_prev - cost_threshold);
            float c_dist_now  = fabsf(v_c_d0 - cost_threshold);
            float c_swing_denom = fminf(c_dist_prev, c_dist_now) + 0.01f * (fabsf(cost_threshold) + 1e-8f);
            float c_swing = fmaxf(c_dist_now, c_dist_prev) / c_swing_denom;
            if (v_change > 4.0f && c_swing > 10.0f) {
                VI_DEBUG_PRINTF("Iter %4d | Extreme oscillation detected (dV=%.1fx, cost swing=%.1fx from boundary), stopping.\n", k, v_change, c_swing);
                break;
            }
        }
        iters_since_tau_change++;

        bool is_feasible = v_c_d0 <= cost_threshold;
        float cost_converge_tol = is_feasible ? 0.001f : 0.01f; // tighter tol if we are already feasible (so that we can go higher)
        bool converged_cost = fabsf(v_c_d0 - cost_threshold) / (fabsf(cost_threshold) + 1e-8f) < cost_converge_tol;
        

        if ((is_feasible || converged_cost) && v_d0 > v_d0_best) {
            v_d0_best = v_d0;
            tensor_copy_data(π_best, π);
        }

        if (k % 1 == 0) {
            VI_DEBUG_PRINTF("Iter %4d | V=%.4f | Cost=%.4f | tau=%.4f | lam=%.4f\n", k, v_d0, v_c_d0, tau, λ);
        }

        float grad_λ = v_c_d0 - cost_threshold;

        // Adam update for λ
        adam_m = adam_beta1 * adam_m + (1.0f - adam_beta1) * grad_λ;
        adam_v = adam_beta2 * adam_v + (1.0f - adam_beta2) * grad_λ * grad_λ;
        float m_hat = adam_m / (1.0f - powf(adam_beta1, (float)(k + 1)));
        float v_hat = adam_v / (1.0f - powf(adam_beta2, (float)(k + 1)));

        lam_prev = λ;
        λ += η_λ * m_hat / (sqrtf(v_hat) + adam_eps);
        if (λ < 0.0f) λ = 0.0f;
        bool converged_lam = fabsf(λ - lam_prev) < 1e-4f;

        if (converged_cost || converged_lam) {
            if (converged_cost) {
                VI_DEBUG_PRINTF("Iter %4d | Cost is within 1%% of threshold: %.4f (threshold=%.4f)\n", k, v_c_d0, cost_threshold);
            } else {
                VI_DEBUG_PRINTF("Iter %4d | λ converged: %.4f\n", k, λ);
            }

            if (v_d0_best > v_d0_best_with_prev_tau) {
                tau_best_overall = tau;
                lam_best_overall = λ;
                v_d0_best_with_prev_tau = v_d0_best;
                tau *= 0.5f;
                VI_DEBUG_PRINTF("  New best policy found with V=%.4f, reducing tau to %.4f for refinement.\n", v_d0_best, tau);
                // reset dynamic variables:
                // λ = 0.0f; // hot start from current λ instead of resetting to zero
                adam_m = 0.0f;
                adam_v = 0.0f;
                lam_prev = 0.0f;
                // since we are going to use a more determinisitc policy, expect more osscilations, so reduce the lr to be safe:
                η_λ *= 0.5f;
                η_λ = fmaxf(η_λ, 0.001f);
                iters_since_tau_change = 0;
            } else {
                // stop if we are not improving the reward anymore (but still close to the cost threshold):
                VI_DEBUG_PRINTF("  No improvement in reward, stopping.\n");
                break;
            }
        }
    }

    // final policy & evaluation
    tensor_copy_data(π, π_best);
    v_d0 = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);
    v_c_d0 = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);
    if (out_cost_d0) *out_cost_d0 = v_c_d0;

    VI_DEBUG_PRINTF("Best overall policy found with V=%.4f, Cost=%.4f, tau=%.4f, lambda=%.4f\n",
           v_d0_best, cost_threshold + (v_c_d0 - cost_threshold), tau_best_overall, lam_best_overall);

    tensor_free(R_L);
    tensor_free(π_best);

    return v_d0;
}
#else  // bisection (default)
// ── Binary-search version: bisect λ for each tau level ──
static float cmdp_vi_bisection_based(
    Tensor* π,       // output: (S, A) stochastic policy
    Tensor* Q, Tensor* V,
    Tensor* Q_c, Tensor* V_c,
    Tensor* T, Tensor* R, Tensor* C,
    Tensor* d0,
    float γ,
    float γ_c,
    float cost_threshold,
    int max_lambda_iters,
    float vi_tol,
    int max_vi_iters,
    float* out_cost_d0
) {
    int nstates = Q->n1, nactions = Q->n2;

    Tensor* R_L = tensor3d_malloc(nstates, nactions, nstates);
    const int nelements_rew_fn = nstates * nactions * nstates;
    Tensor* π_best = tensor2d_malloc(nstates, nactions);

    float v_d0_best = -1e30f;
    float v_d0_best_prev_tau = -1e30f;  // best V at the previous tau level
    float tau_best_overall = 0.0f;
    float lam_best_overall = 0.0f;

    // tau schedule: start high (exploratory), halve each round
    float tau = 1.0f;
    const float tau_min = 0.001f;
    int bisect_iters = max_lambda_iters;  // budget per tau level

    while (tau >= tau_min) {

        // ── Find λ_hi such that cost(π(λ_hi)) <= threshold ──
        float lam_lo = 0.0f;
        float lam_hi = 1.0f;

        // Evaluate at λ=0 (unconstrained)
        for (int i = 0; i < nelements_rew_fn; i++)
            R_L->data[i] = R->data[i] - lam_lo * C->data[i];
        value_iteration_soft(π, Q, V, T, R_L, d0, γ, tau, vi_tol, max_vi_iters);
        float cost_lo = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);
        float v_lo = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);

        // If already feasible at λ=0, record and move to next tau
        if (cost_lo <= cost_threshold) {
            VI_DEBUG_PRINTF("tau=%.4f | λ=0 already feasible: V=%.4f, Cost=%.4f\n", tau, v_lo, cost_lo);
            if (v_lo > v_d0_best) {
                v_d0_best = v_lo;
                tensor_copy_data(π_best, π);
                tau_best_overall = tau;
                lam_best_overall = 0.0f;
            }
            tau *= 0.5f;
            continue;
        }

        // Double λ_hi until the policy becomes feasible
        for (int i = 0; i < nelements_rew_fn; i++)
            R_L->data[i] = R->data[i] - lam_hi * C->data[i];
        value_iteration_soft(π, Q, V, T, R_L, d0, γ, tau, vi_tol, max_vi_iters);
        float cost_hi = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);

        while (cost_hi > cost_threshold) {
            lam_hi *= 2.0f;
            if (lam_hi > 1e6f) {
                VI_DEBUG_PRINTF("tau=%.4f | Warning: λ > 1e6, constraint may be infeasible at this tau\n", tau);
                break;
            }
            for (int i = 0; i < nelements_rew_fn; i++)
                R_L->data[i] = R->data[i] - lam_hi * C->data[i];
            value_iteration_soft(π, Q, V, T, R_L, d0, γ, tau, vi_tol, max_vi_iters);
            cost_hi = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);
        }

        if (cost_hi > cost_threshold) {
            // infeasible at this tau, skip to next
            tau *= 0.5f;
            continue;
        }

        // ── Binary search on λ in [lam_lo, lam_hi] ──
        // Invariant: cost(lam_lo) > threshold, cost(lam_hi) <= threshold
        // We want the smallest λ that makes it feasible (maximizes reward).
        float best_v_this_tau = -1e30f;

        for (int b = 0; b < bisect_iters; b++) {
            float lam_mid = (lam_lo + lam_hi) * 0.5f;

            for (int i = 0; i < nelements_rew_fn; i++)
                R_L->data[i] = R->data[i] - lam_mid * C->data[i];
            value_iteration_soft(π, Q, V, T, R_L, d0, γ, tau, vi_tol, max_vi_iters);
            float v_mid = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);
            float cost_mid = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);

                 VI_DEBUG_PRINTF("tau=%.4f | bisect %3d | λ=%.6f | V=%.4f | Cost=%.4f | [%.6f, %.6f]\n",
                   tau, b, lam_mid, v_mid, cost_mid, lam_lo, lam_hi);

            if (cost_mid <= cost_threshold) {
                // feasible: try smaller λ for better reward
                lam_hi = lam_mid;
                if (v_mid > best_v_this_tau) {
                    best_v_this_tau = v_mid;
                    if (v_mid > v_d0_best) {
                        v_d0_best = v_mid;
                        tensor_copy_data(π_best, π);
                        tau_best_overall = tau;
                        lam_best_overall = lam_mid;
                    }
                }
            } else {
                // infeasible: need larger λ
                lam_lo = lam_mid;
            }

            // Converged when interval is tiny
            if (lam_hi - lam_lo < 1e-6f * (lam_hi + 1e-8f)) {
                VI_DEBUG_PRINTF("tau=%.4f | λ converged at %.6f after %d bisections\n", tau, (lam_lo + lam_hi) * 0.5f, b + 1);
                break;
            }
        }

        VI_DEBUG_PRINTF("tau=%.4f | best V this tau=%.4f, overall best V=%.4f\n", tau, best_v_this_tau, v_d0_best);

        // Stop if this tau level didn't improve over the previous one
        if (v_d0_best <= v_d0_best_prev_tau) {
            VI_DEBUG_PRINTF("  No improvement over previous tau, stopping.\n");
            break;
        }
        v_d0_best_prev_tau = v_d0_best;
        tau *= 0.5f;
    }

    // final policy & evaluation
    tensor_copy_data(π, π_best);
    float v_d0 = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);
    float v_c_d0 = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);
    if (out_cost_d0) *out_cost_d0 = v_c_d0;

    VI_DEBUG_PRINTF("Best overall policy found with V=%.4f, Cost=%.4f, tau=%.4f, lambda=%.4f\n",
           v_d0, v_c_d0, tau_best_overall, lam_best_overall);

    tensor_free(R_L);
    tensor_free(π_best);

    return v_d0;
}
#endif  // CMDP_DUAL_OPT_USE_GRADIENTS (single-constraint)

#ifdef CMDP_DUAL_OPT_USE_GRADIENTS
// ── Adam gradient-based two-constraints version: dual gradient ascent on λ1, λ2 ──
static float cmdp_vi_two_constraints_gradient_based(
    Tensor* π,       // output: (S, A) stochastic policy
    Tensor* Q, Tensor* V,
    Tensor* Q_c1, Tensor* V_c1,
    Tensor* Q_c2, Tensor* V_c2,
    Tensor* T, Tensor* R, Tensor* C1, Tensor* C2,
    Tensor* d0,
    float γ,
    float γ_c1,
    float γ_c2,
    float cost_threshold1,
    float cost_threshold2,
    int max_lambda_iters,
    float vi_tol,
    int max_vi_iters,
    float* out_cost1_d0,
    float* out_cost2_d0
) {
    int nstates = Q->n1, nactions = Q->n2;

    float tau = 1.0f;
    float λ1 = 0.0f, λ2 = 0.0f;
    float η_λ1 = 0.1f, η_λ2 = 0.1f;
    float lam1_prev = 0.0f, lam2_prev = 0.0f;

    // Adam optimizer state for λ1
    float adam_m1 = 0.0f, adam_v1 = 0.0f;
    // Adam optimizer state for λ2
    float adam_m2 = 0.0f, adam_v2 = 0.0f;
    const float adam_beta1 = 0.9f;
    const float adam_beta2 = 0.999f;
    const float adam_eps = 1e-8f;

    Tensor* R_L = tensor3d_malloc(nstates, nactions, nstates);
    const int nelements_rew_fn = nstates * nactions * nstates;
    Tensor* π_best = tensor2d_malloc(nstates, nactions);
    float v_d0_best = -1e30f;
    float v_d0_best_with_prev_tau = -1e30f;
    float tau_best_overall = tau;
    float lam1_best_overall = λ1, lam2_best_overall = λ2;

    float v_d0 = 0.0f, v_c1_d0 = 0.0f, v_c2_d0 = 0.0f;
    float v_d0_prev = 0.0f, v_c1_d0_prev = 0.0f, v_c2_d0_prev = 0.0f;
    int iters_since_tau_change = 0;

    for (int k = 0; k < max_lambda_iters; k++) {

        // create combined reward function: R - λ1*C1 - λ2*C2
        for (int i = 0; i < nelements_rew_fn; i++) {
            R_L->data[i] = R->data[i] - λ1 * C1->data[i] - λ2 * C2->data[i];
        }

        v_d0_prev = v_d0;
        v_c1_d0_prev = v_c1_d0;
        v_c2_d0_prev = v_c2_d0;
        value_iteration_soft(π, Q, V, T, R_L, d0, γ, tau, vi_tol, max_vi_iters);
        v_d0 = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);
        v_c1_d0 = policy_evaluation_linalg(π, Q_c1, V_c1, T, C1, d0, γ_c1);
        v_c2_d0 = policy_evaluation_linalg(π, Q_c2, V_c2, T, C2, d0, γ_c2);

        // Detect extreme oscillations: large V change AND any cost swung far from constraint boundary
        if (iters_since_tau_change >= 5) {
            float v_min = fminf(fabsf(v_d0), fabsf(v_d0_prev)) + 1e-8f;
            float v_change = fabsf(v_d0 - v_d0_prev) / v_min;
            float c1_dist_prev = fabsf(v_c1_d0_prev - cost_threshold1);
            float c1_dist_now  = fabsf(v_c1_d0 - cost_threshold1);
            float c1_swing_denom = fminf(c1_dist_prev, c1_dist_now) + 0.01f * (fabsf(cost_threshold1) + 1e-8f);
            float c1_swing = fmaxf(c1_dist_now, c1_dist_prev) / c1_swing_denom;
            float c2_dist_prev = fabsf(v_c2_d0_prev - cost_threshold2);
            float c2_dist_now  = fabsf(v_c2_d0 - cost_threshold2);
            float c2_swing_denom = fminf(c2_dist_prev, c2_dist_now) + 0.01f * (fabsf(cost_threshold2) + 1e-8f);
            float c2_swing = fmaxf(c2_dist_now, c2_dist_prev) / c2_swing_denom;
            if (v_change > 4.0f && (c1_swing > 10.0f || c2_swing > 10.0f)) {
                VI_DEBUG_PRINTF("Iter %4d | Extreme oscillation detected (dV=%.1fx, c1_swing=%.1fx, c2_swing=%.1fx from boundary), stopping.\n", k, v_change, c1_swing, c2_swing);
                break;
            }
        }
        iters_since_tau_change++;

        bool is_feasible1 = v_c1_d0 <= cost_threshold1;
        bool is_feasible2 = v_c2_d0 <= cost_threshold2;
        float cost_tol1 = is_feasible1 ? 0.001f : 0.01f;
        float cost_tol2 = is_feasible2 ? 0.001f : 0.01f;
        bool converged_cost1 = fabsf(v_c1_d0 - cost_threshold1) / (fabsf(cost_threshold1) + 1e-8f) < cost_tol1;
        bool converged_cost2 = fabsf(v_c2_d0 - cost_threshold2) / (fabsf(cost_threshold2) + 1e-8f) < cost_tol2;
        bool converged_cost = converged_cost1 && converged_cost2;
        bool is_feasible = is_feasible1 && is_feasible2;

        if ((is_feasible || converged_cost) && v_d0 > v_d0_best) {
            v_d0_best = v_d0;
            tensor_copy_data(π_best, π);
        }

        if (k % 1 == 0) {
                 VI_DEBUG_PRINTF("Iter %4d | V=%.4f | Cost1=%.4f | Cost2=%.4f | tau=%.4f | lam1=%.4f | lam2=%.4f\n",
                   k, v_d0, v_c1_d0, v_c2_d0, tau, λ1, λ2);
        }

        // Adam update for λ1
        float grad_λ1 = v_c1_d0 - cost_threshold1;
        adam_m1 = adam_beta1 * adam_m1 + (1.0f - adam_beta1) * grad_λ1;
        adam_v1 = adam_beta2 * adam_v1 + (1.0f - adam_beta2) * grad_λ1 * grad_λ1;
        float m1_hat = adam_m1 / (1.0f - powf(adam_beta1, (float)(k + 1)));
        float v1_hat = adam_v1 / (1.0f - powf(adam_beta2, (float)(k + 1)));

        lam1_prev = λ1;
        λ1 += η_λ1 * m1_hat / (sqrtf(v1_hat) + adam_eps);
        if (λ1 < 0.0f) λ1 = 0.0f;

        // Adam update for λ2
        float grad_λ2 = v_c2_d0 - cost_threshold2;
        adam_m2 = adam_beta1 * adam_m2 + (1.0f - adam_beta1) * grad_λ2;
        adam_v2 = adam_beta2 * adam_v2 + (1.0f - adam_beta2) * grad_λ2 * grad_λ2;
        float m2_hat = adam_m2 / (1.0f - powf(adam_beta1, (float)(k + 1)));
        float v2_hat = adam_v2 / (1.0f - powf(adam_beta2, (float)(k + 1)));

        lam2_prev = λ2;
        λ2 += η_λ2 * m2_hat / (sqrtf(v2_hat) + adam_eps);
        if (λ2 < 0.0f) λ2 = 0.0f;

        bool converged_lam = fabsf(λ1 - lam1_prev) < 1e-4f && fabsf(λ2 - lam2_prev) < 1e-4f;

        if (converged_cost || converged_lam) {
            if (converged_cost) {
                  VI_DEBUG_PRINTF("Iter %4d | Costs within tolerance: C1=%.4f (thresh=%.4f), C2=%.4f (thresh=%.4f)\n",
                       k, v_c1_d0, cost_threshold1, v_c2_d0, cost_threshold2);
            } else {
                VI_DEBUG_PRINTF("Iter %4d | λs converged: λ1=%.4f, λ2=%.4f\n", k, λ1, λ2);
            }

            if (v_d0_best > v_d0_best_with_prev_tau) {
                tau_best_overall = tau;
                lam1_best_overall = λ1;
                lam2_best_overall = λ2;
                v_d0_best_with_prev_tau = v_d0_best;
                tau *= 0.5f;
                VI_DEBUG_PRINTF("  New best policy found with V=%.4f, reducing tau to %.4f for refinement.\n", v_d0_best, tau);
                // reset Adam state, hot-start λs
                adam_m1 = 0.0f; adam_v1 = 0.0f;
                adam_m2 = 0.0f; adam_v2 = 0.0f;
                lam1_prev = 0.0f; lam2_prev = 0.0f;
                // reduce lr for more deterministic regime
                η_λ1 *= 0.5f; η_λ1 = fmaxf(η_λ1, 0.0001f);
                η_λ2 *= 0.5f; η_λ2 = fmaxf(η_λ2, 0.0001f);
                iters_since_tau_change = 0;
            } else {
                VI_DEBUG_PRINTF("  No improvement in reward, stopping.\n");
                break;
            }
        }
    }

    // final policy & evaluation
    tensor_copy_data(π, π_best);
    v_d0 = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);
    v_c1_d0 = policy_evaluation_linalg(π, Q_c1, V_c1, T, C1, d0, γ_c1);
    v_c2_d0 = policy_evaluation_linalg(π, Q_c2, V_c2, T, C2, d0, γ_c2);

    if (out_cost1_d0) *out_cost1_d0 = v_c1_d0;
    if (out_cost2_d0) *out_cost2_d0 = v_c2_d0;

    VI_DEBUG_PRINTF("Best overall policy: V=%.4f, C1=%.4f, C2=%.4f, tau=%.4f, lam1=%.4f, lam2=%.4f\n",
           v_d0, v_c1_d0, v_c2_d0, tau_best_overall, lam1_best_overall, lam2_best_overall);

    tensor_free(R_L);
    tensor_free(π_best);

    return v_d0;
}
#else  // bisection (default)
// ── Binary-search version (alternating coordinate bisection on λ1, λ2) ──
static float cmdp_vi_two_constraints_bisection_based(
    Tensor* π,       // output: (S, A) stochastic policy
    Tensor* Q, Tensor* V,
    Tensor* Q_c1, Tensor* V_c1,
    Tensor* Q_c2, Tensor* V_c2,
    Tensor* T, Tensor* R, Tensor* C1, Tensor* C2,
    Tensor* d0,
    float γ,
    float γ_c1,
    float γ_c2,
    float cost_threshold1,
    float cost_threshold2,
    int max_lambda_iters,
    float vi_tol,
    int max_vi_iters,
    float* out_cost1_d0,
    float* out_cost2_d0
) {
    int nstates = Q->n1, nactions = Q->n2;

    Tensor* R_L = tensor3d_malloc(nstates, nactions, nstates);
    const int nelements_rew_fn = nstates * nactions * nstates;
    Tensor* π_best = tensor2d_malloc(nstates, nactions);

    float v_d0_best = -1e30f;
    float v_d0_best_prev_tau = -1e30f;
    float tau_best_overall = 0.0f;
    float lam1_best_overall = 0.0f, lam2_best_overall = 0.0f;

    // Helper: evaluate policy for given (λ1, λ2, tau), return V, C1, C2
    #define EVAL_POLICY(l1, l2, tau_val, out_v, out_c1, out_c2) do { \
        for (int i_ = 0; i_ < nelements_rew_fn; i_++) \
            R_L->data[i_] = R->data[i_] - (l1) * C1->data[i_] - (l2) * C2->data[i_]; \
        value_iteration_soft(π, Q, V, T, R_L, d0, γ, (tau_val), vi_tol, max_vi_iters); \
        (out_v)  = policy_evaluation_linalg(π, Q, V, T, R, d0, γ); \
        (out_c1) = policy_evaluation_linalg(π, Q_c1, V_c1, T, C1, d0, γ_c1); \
        (out_c2) = policy_evaluation_linalg(π, Q_c2, V_c2, T, C2, d0, γ_c2); \
    } while(0)

    float tau = 1.0f;
    const float tau_min = 0.001f;
    int bisect_iters = max_lambda_iters;  // budget per coordinate bisection

    while (tau >= tau_min) {
        float v_d0, c1_d0, c2_d0;

        // Check if λ1=λ2=0 is already feasible
        EVAL_POLICY(0.0f, 0.0f, tau, v_d0, c1_d0, c2_d0);
        if (c1_d0 <= cost_threshold1 && c2_d0 <= cost_threshold2) {
            VI_DEBUG_PRINTF("tau=%.4f | λ1=λ2=0 already feasible: V=%.4f, C1=%.4f, C2=%.4f\n", tau, v_d0, c1_d0, c2_d0);
            if (v_d0 > v_d0_best) {
                v_d0_best = v_d0;
                tensor_copy_data(π_best, π);
                tau_best_overall = tau;
                lam1_best_overall = 0.0f;
                lam2_best_overall = 0.0f;
            }
            tau *= 0.5f;
            continue;
        }

        // ── Alternating coordinate bisection ──
        // Start with λ1=λ2=0, alternately bisect each to satisfy its constraint.
        float λ1 = 0.0f, λ2 = 0.0f;
        const int max_outer_rounds = 20;  // alternation rounds
        float best_v_this_tau = -1e30f;

        for (int round = 0; round < max_outer_rounds; round++) {
            float λ1_prev = λ1, λ2_prev = λ2;

            // ── Bisect λ1 (with λ2 fixed) to satisfy constraint 1 ──
            EVAL_POLICY(0.0f, λ2, tau, v_d0, c1_d0, c2_d0);
            if (c1_d0 > cost_threshold1) {
                float lo1 = 0.0f, hi1 = 1.0f;
                // Find upper bound
                EVAL_POLICY(hi1, λ2, tau, v_d0, c1_d0, c2_d0);
                while (c1_d0 > cost_threshold1) {
                    hi1 *= 2.0f;
                    if (hi1 > 1e6f) break;
                    EVAL_POLICY(hi1, λ2, tau, v_d0, c1_d0, c2_d0);
                }
                if (c1_d0 <= cost_threshold1) {
                    for (int b = 0; b < bisect_iters; b++) {
                        float mid1 = (lo1 + hi1) * 0.5f;
                        EVAL_POLICY(mid1, λ2, tau, v_d0, c1_d0, c2_d0);
                        if (c1_d0 <= cost_threshold1) {
                            hi1 = mid1;
                            // Track best feasible (both constraints)
                            if (c2_d0 <= cost_threshold2 && v_d0 > best_v_this_tau) {
                                best_v_this_tau = v_d0;
                                if (v_d0 > v_d0_best) {
                                    v_d0_best = v_d0;
                                    tensor_copy_data(π_best, π);
                                    tau_best_overall = tau;
                                    lam1_best_overall = mid1;
                                    lam2_best_overall = λ2;
                                }
                            }
                        } else {
                            lo1 = mid1;
                        }
                        if (hi1 - lo1 < 1e-6f * (hi1 + 1e-8f)) break;
                    }
                    λ1 = hi1;  // smallest feasible λ1
                }
            } else {
                λ1 = 0.0f;
                // Already feasible for C1, check if also feasible for C2
                if (c2_d0 <= cost_threshold2 && v_d0 > best_v_this_tau) {
                    best_v_this_tau = v_d0;
                    if (v_d0 > v_d0_best) {
                        v_d0_best = v_d0;
                        tensor_copy_data(π_best, π);
                        tau_best_overall = tau;
                        lam1_best_overall = 0.0f;
                        lam2_best_overall = λ2;
                    }
                }
            }

            // ── Bisect λ2 (with λ1 fixed) to satisfy constraint 2 ──
            EVAL_POLICY(λ1, 0.0f, tau, v_d0, c1_d0, c2_d0);
            if (c2_d0 > cost_threshold2) {
                float lo2 = 0.0f, hi2 = 1.0f;
                // Find upper bound
                EVAL_POLICY(λ1, hi2, tau, v_d0, c1_d0, c2_d0);
                while (c2_d0 > cost_threshold2) {
                    hi2 *= 2.0f;
                    if (hi2 > 1e6f) break;
                    EVAL_POLICY(λ1, hi2, tau, v_d0, c1_d0, c2_d0);
                }
                if (c2_d0 <= cost_threshold2) {
                    for (int b = 0; b < bisect_iters; b++) {
                        float mid2 = (lo2 + hi2) * 0.5f;
                        EVAL_POLICY(λ1, mid2, tau, v_d0, c1_d0, c2_d0);
                        if (c2_d0 <= cost_threshold2) {
                            hi2 = mid2;
                            // Track best feasible (both constraints)
                            if (c1_d0 <= cost_threshold1 && v_d0 > best_v_this_tau) {
                                best_v_this_tau = v_d0;
                                if (v_d0 > v_d0_best) {
                                    v_d0_best = v_d0;
                                    tensor_copy_data(π_best, π);
                                    tau_best_overall = tau;
                                    lam1_best_overall = λ1;
                                    lam2_best_overall = mid2;
                                }
                            }
                        } else {
                            lo2 = mid2;
                        }
                        if (hi2 - lo2 < 1e-6f * (hi2 + 1e-8f)) break;
                    }
                    λ2 = hi2;  // smallest feasible λ2
                }
            } else {
                λ2 = 0.0f;
                // Already feasible for C2, check if also feasible for C1
                if (c1_d0 <= cost_threshold1 && v_d0 > best_v_this_tau) {
                    best_v_this_tau = v_d0;
                    if (v_d0 > v_d0_best) {
                        v_d0_best = v_d0;
                        tensor_copy_data(π_best, π);
                        tau_best_overall = tau;
                        lam1_best_overall = λ1;
                        lam2_best_overall = 0.0f;
                    }
                }
            }

            EVAL_POLICY(λ1, λ2, tau, v_d0, c1_d0, c2_d0);
                 VI_DEBUG_PRINTF("tau=%.4f | round %2d | λ1=%.6f | λ2=%.6f | V=%.4f | C1=%.4f | C2=%.4f | best_V=%.4f\n",
                   tau, round, λ1, λ2, v_d0, c1_d0, c2_d0, best_v_this_tau);

            // Check if both multipliers converged
            if (fabsf(λ1 - λ1_prev) < 1e-6f * (λ1 + 1e-8f) &&
                fabsf(λ2 - λ2_prev) < 1e-6f * (λ2 + 1e-8f)) {
                VI_DEBUG_PRINTF("tau=%.4f | λs converged after %d rounds\n", tau, round + 1);
                break;
            }
        }

        VI_DEBUG_PRINTF("tau=%.4f | best V this tau=%.4f, overall best V=%.4f\n", tau, best_v_this_tau, v_d0_best);

        // Stop if this tau level didn't improve over the previous one
        if (v_d0_best <= v_d0_best_prev_tau) {
            VI_DEBUG_PRINTF("  No improvement over previous tau, stopping.\n");
            break;
        }
        v_d0_best_prev_tau = v_d0_best;
        tau *= 0.5f;
    }

    #undef EVAL_POLICY

    // final policy & evaluation
    tensor_copy_data(π, π_best);
    float v_d0_final = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);
    float c1_final = policy_evaluation_linalg(π, Q_c1, V_c1, T, C1, d0, γ_c1);
    float c2_final = policy_evaluation_linalg(π, Q_c2, V_c2, T, C2, d0, γ_c2);

    if (out_cost1_d0) *out_cost1_d0 = c1_final;
    if (out_cost2_d0) *out_cost2_d0 = c2_final;

    VI_DEBUG_PRINTF("Best overall policy: V=%.4f, C1=%.4f, C2=%.4f, tau=%.4f, lam1=%.4f, lam2=%.4f\n",
           v_d0_final, c1_final, c2_final, tau_best_overall, lam1_best_overall, lam2_best_overall);

    tensor_free(R_L);
    tensor_free(π_best);

    return v_d0_final;
}
#endif  // CMDP_DUAL_OPT_USE_GRADIENTS (two-constraint)

// ===========================================================================
// CMDP with fixed temperature (no tau search) — bisect λ only
// ===========================================================================
float cmdp_value_iteration_soft(
    Tensor* π,
    Tensor* Q, Tensor* V,
    Tensor* Q_c, Tensor* V_c,
    Tensor* T, Tensor* R, Tensor* C,
    Tensor* d0,
    float γ,
    float γ_c,
    float temperature,
    float cost_threshold,
    int max_lambda_iters,
    float vi_tol,
    int max_vi_iters,
    float* out_cost_d0
) {
    int nstates = Q->n1, nactions = Q->n2;

    Tensor* R_L = tensor3d_malloc(nstates, nactions, nstates);
    const int nelements_rew_fn = nstates * nactions * nstates;

    // ── Check if λ=0 (unconstrained) is already feasible ──
    float lam_lo = 0.0f;
    float lam_hi = 1.0f;

    for (int i = 0; i < nelements_rew_fn; i++)
        R_L->data[i] = R->data[i];
    value_iteration_soft(π, Q, V, T, R_L, d0, γ, temperature, vi_tol, max_vi_iters);
    float cost_lo = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);
    float v_best = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);

    if (cost_lo <= cost_threshold) {
        VI_DEBUG_PRINTF("cmdp_soft | λ=0 already feasible: V=%.4f, Cost=%.4f\n", v_best, cost_lo);
        if (out_cost_d0) *out_cost_d0 = cost_lo;
        tensor_free(R_L);
        return v_best;
    }

    // ── Double λ_hi until feasible ──
    for (int i = 0; i < nelements_rew_fn; i++)
        R_L->data[i] = R->data[i] - lam_hi * C->data[i];
    value_iteration_soft(π, Q, V, T, R_L, d0, γ, temperature, vi_tol, max_vi_iters);
    float cost_hi = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);

    while (cost_hi > cost_threshold) {
        lam_hi *= 2.0f;
        if (lam_hi > 1e6f) {
            VI_DEBUG_PRINTF("cmdp_soft | Warning: λ > 1e6, constraint may be infeasible\n");
            break;
        }
        for (int i = 0; i < nelements_rew_fn; i++)
            R_L->data[i] = R->data[i] - lam_hi * C->data[i];
        value_iteration_soft(π, Q, V, T, R_L, d0, γ, temperature, vi_tol, max_vi_iters);
        cost_hi = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);
    }

    if (cost_hi > cost_threshold) {
        // Infeasible — return best-effort (λ_hi policy)
        float v_d0 = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);
        if (out_cost_d0) *out_cost_d0 = cost_hi;
        tensor_free(R_L);
        return v_d0;
    }

    // ── Binary search on λ ──
    Tensor* π_best = tensor2d_malloc(nstates, nactions);
    tensor_copy_data(π_best, π);  // lam_hi is feasible
    v_best = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);

    for (int b = 0; b < max_lambda_iters; b++) {
        float lam_mid = (lam_lo + lam_hi) * 0.5f;

        for (int i = 0; i < nelements_rew_fn; i++)
            R_L->data[i] = R->data[i] - lam_mid * C->data[i];
        value_iteration_soft(π, Q, V, T, R_L, d0, γ, temperature, vi_tol, max_vi_iters);
        float v_mid = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);
        float cost_mid = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);

        VI_DEBUG_PRINTF("cmdp_soft | bisect %3d | λ=%.6f | V=%.4f | Cost=%.4f | [%.6f, %.6f]\n",
                   b, lam_mid, v_mid, cost_mid, lam_lo, lam_hi);

        if (cost_mid <= cost_threshold) {
            lam_hi = lam_mid;
            if (v_mid > v_best) {
                v_best = v_mid;
                tensor_copy_data(π_best, π);
            }
        } else {
            lam_lo = lam_mid;
        }

        if (lam_hi - lam_lo < 1e-6f * (lam_hi + 1e-8f)) {
            VI_DEBUG_PRINTF("cmdp_soft | λ converged at %.6f after %d bisections\n",
                       (lam_lo + lam_hi) * 0.5f, b + 1);
            break;
        }
    }

    // Final policy & evaluation
    tensor_copy_data(π, π_best);
    float v_d0 = policy_evaluation_linalg(π, Q, V, T, R, d0, γ);
    float v_c_d0 = policy_evaluation_linalg(π, Q_c, V_c, T, C, d0, γ_c);
    if(out_cost_d0) *out_cost_d0 = v_c_d0;

    VI_DEBUG_PRINTF("cmdp_soft | Best policy: V=%.4f, Cost=%.4f, tau=%.4f\n",
           v_d0, v_c_d0, temperature);

    tensor_free(R_L);
    tensor_free(π_best);
    return v_d0;
}

// ── Public dispatch wrappers ──

float cmdp_value_iteration(
    Tensor* π,
    Tensor* Q, Tensor* V,
    Tensor* Q_c, Tensor* V_c,
    Tensor* T, Tensor* R, Tensor* C,
    Tensor* d0,
    float γ,
    float γ_c,
    float cost_threshold,
    int max_lambda_iters,
    float vi_tol,
    int max_vi_iters,
    float* out_cost_d0
) {
#ifdef CMDP_DUAL_OPT_USE_GRADIENTS
    return cmdp_vi_gradient_based(
        π, Q, V, Q_c, V_c, T, R, C, d0,
        γ, γ_c, cost_threshold,
        max_lambda_iters, vi_tol, max_vi_iters, out_cost_d0);
#else
    return cmdp_vi_bisection_based(
        π, Q, V, Q_c, V_c, T, R, C, d0,
        γ, γ_c, cost_threshold,
        max_lambda_iters, vi_tol, max_vi_iters, out_cost_d0);
#endif
}

float cmdp_value_iteration_two_constraints(
    Tensor* π,
    Tensor* Q, Tensor* V,
    Tensor* Q_c1, Tensor* V_c1,
    Tensor* Q_c2, Tensor* V_c2,
    Tensor* T, Tensor* R, Tensor* C1, Tensor* C2,
    Tensor* d0,
    float γ,
    float γ_c1,
    float γ_c2,
    float cost_threshold1,
    float cost_threshold2,
    int max_lambda_iters,
    float vi_tol,
    int max_vi_iters,
    float* out_cost1_d0,
    float* out_cost2_d0
) {
#ifdef CMDP_DUAL_OPT_USE_GRADIENTS
    return cmdp_vi_two_constraints_gradient_based(
        π, Q, V, Q_c1, V_c1, Q_c2, V_c2, T, R, C1, C2, d0,
        γ, γ_c1, γ_c2, cost_threshold1, cost_threshold2,
        max_lambda_iters, vi_tol, max_vi_iters, out_cost1_d0, out_cost2_d0);
#else
    return cmdp_vi_two_constraints_bisection_based(
        π, Q, V, Q_c1, V_c1, Q_c2, V_c2, T, R, C1, C2, d0,
        γ, γ_c1, γ_c2, cost_threshold1, cost_threshold2,
        max_lambda_iters, vi_tol, max_vi_iters, out_cost1_d0, out_cost2_d0);
#endif
}
