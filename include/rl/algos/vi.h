#include "rl/core/mdp.h"

float value_iteration(Tensor* Q, Tensor* V, Tensor* T, Tensor* R, Tensor* d0, float gamma, float tol, int max_iters);

float value_iteration_soft(Tensor* pi, Tensor* Q, Tensor* V, Tensor* T, Tensor* R, Tensor* d0, float gamma, float temperature, float tol, int max_iters);

float policy_evaluation(Tensor* probs_matrix, Tensor* Q, Tensor* V, Tensor* T, Tensor* R, Tensor* d0, float gamma, float tol, int max_iters);

// Policy evaluation for costs — same as policy_evaluation but with cost matrix instead of reward matrix.
float policy_evaluation_cost(Tensor* probs_matrix, Tensor* Q_cost, Tensor* V_cost, Tensor* T, Tensor* C, Tensor* d0, float cost_gamma, float tol, int max_iters);

// Policy evaluation for secondary rewards — same as policy_evaluation but with secondary reward matrix.
float policy_evaluation_secondary_reward(Tensor* probs_matrix, Tensor* Q_sr, Tensor* V_sr, Tensor* T, Tensor* SR, Tensor* d0, float sr_gamma, float tol, int max_iters);

// Lexicographic value iteration (Wray et al. AAAI 2015, non-conditional case).
// Two objectives: primary reward R (priority 0) and secondary reward SR (priority 1).
// `slack` is a value-function tolerance (delta): the lex policy's primary return
// is guaranteed >= V*(d0) - delta.  Internally we binary-search for the largest
// per-state Q threshold (eta) such that the actual primary loss stays within slack.
// (The Wray bound eta = (1-gamma)*delta is used as the known-feasible lower bound.)
// Returns V_sr(d0) of the lexicographic policy.
float lexicographic_value_iteration(
    Tensor* policy_probs,                    // output: (S, A) deterministic lex policy
    Tensor* Q_primary, Tensor* V_primary,    // output: Primary Q and V (accounted for slack, so may be slightly suboptimal)
    Tensor* Q_sr, Tensor* V_sr,              // output: lex-constrained secondary Q and V
    Tensor* T, Tensor* R, Tensor* SR,
    Tensor* d0,
    float gamma, float sr_gamma, float slack,
    float tol, int max_iters,
    float* out_primary_d0                 // output (optional): V(d0) of lex policy on primary reward
);

// Cost-constrained value iteration (CMDP via Lagrangian relaxation).
// Returns V(d0). Outputs Q, V (reward), Q_cost, V_cost (cost), and stochastic policy_probs (S, A).
// max_lambda_iters: budget for lambda bisection per tau level.
// vi_tol / max_vi_iters: convergence tolerance and iteration cap for the inner soft VI.
float cmdp_value_iteration(
    Tensor* policy_probs,       // output: (S, A) stochastic policy
    Tensor* Q, Tensor* V,       // output: reward Q-values and state values
    Tensor* Q_cost, Tensor* V_cost, // output: cost Q-values and state values
    Tensor* T,                  // transition matrix (S, A, S')
    Tensor* R,                  // reward matrix (S, A, S')
    Tensor* C,                  // cost matrix (S, A, S')
    Tensor* d0,                 // start state distribution
    float gamma,
    float cost_gamma,
    float cost_threshold,
    int max_lambda_iters,
    float vi_tol,
    int max_vi_iters,
    float* out_cost_d0       // output: expected discounted cost V_cost(d0)
);

// Two-constraint CMDP via Lagrangian relaxation (alternating coordinate bisection).
float cmdp_value_iteration_two_constraints(
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
);

// CMDP with fixed temperature (no tau search) — bisects λ only.
float cmdp_value_iteration_soft(
    Tensor* policy_probs,       // output: (S, A) stochastic policy
    Tensor* Q, Tensor* V,
    Tensor* Q_cost, Tensor* V_cost,
    Tensor* T, Tensor* R, Tensor* C,
    Tensor* d0,
    float gamma,
    float cost_gamma,
    float temperature,
    float cost_threshold,
    int max_lambda_iters,
    float vi_tol,
    int max_vi_iters,
    float* out_cost_d0
);