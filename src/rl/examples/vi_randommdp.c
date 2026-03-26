#include <stdio.h>
#include <time.h>
#include "rl/envs/random_mdp.h"
#include "rl/algos/vi.h"

int main(void) {
    srand((unsigned int)time(NULL));

    int num_states  = 10;
    int num_actions = 5;
    double alpha = 1.0;   // Dirichlet concentration for transitions
    double beta  = 0.0;   // reward spread around mean 1.0
    int max_steps = 500;   // Must be >> 1/(1-gamma) so PE and rollout agree (no absorbing states)
    float gamma = 0.99f;
    float cost_gamma = 0.99f;

    printf("=== RandomDiscreteMDP: %d states, %d actions, alpha=%.2f, beta=%.2f ===\n\n",
           num_states, num_actions, alpha, beta);

    MDP* env = random_mdp_create(num_states, num_actions, alpha, beta, false, max_steps);

    // --- Extract tabular dynamics ---
    Tensor* T  = mdp_malloc_transition_matrix(env);
    Tensor* R  = mdp_malloc_reward_matrix(env);
    Tensor* C  = mdp_malloc_cost_matrix(env);
    Tensor* d0 = mdp_malloc_start_state_probs(env);
    mdp_extract_tabular_dynamics(env, T, R, d0);
    mdp_extract_cost_matrix(env, C);

//     tensor_print(T, "Transition matrix T[s,a,s']");
//     tensor_print(R, "Reward matrix R[s,a,s']");
//     tensor_print(C, "Cost matrix C[s,a,s']");
//     tensor_print(d0, "Start state probabilities d0[s]");

    // --- Value Iteration ---
    Tensor* Q = tensor2d_malloc(num_states, num_actions);
    Tensor* V = tensor1d_malloc(num_states);

    float v_star = value_iteration(Q, V, T, R, d0, gamma, 1e-8f, 10000);
    printf("Value Iteration:\n");
    printf("  V*(d0) = %.6f\n", (double)v_star);

    // Print V* for each state
    tensor_print(V, "Optimal state values V*");
    tensor_print(Q, "Optimal action values Q*");

    // --- Policy Evaluation of Random Policy ---
    Policy* random_pol = random_policy_malloc(env->state_space, env->action_space);
    Tensor* probs = policy_malloc_probs_matrix(random_pol);
    policy_extract_tabular_action_probabilities(random_pol, probs);

    Tensor* Q_rand = tensor2d_malloc(num_states, num_actions);
    Tensor* V_rand = tensor1d_malloc(num_states);

    float v_rand = policy_evaluation(probs, Q_rand, V_rand, T, R, d0, gamma, 1e-8f, 10000);
    printf("\nRandom Policy Evaluation:\n");
    printf("  V_rand(d0) = %.6f\n", (double)v_rand);

    // Cost PE of random policy
    Tensor* Q_rand_c = tensor2d_malloc(num_states, num_actions);
    Tensor* V_rand_c = tensor1d_malloc(num_states);
    float c_rand = policy_evaluation_cost(probs, Q_rand_c, V_rand_c, T, C, d0, cost_gamma, 1e-8f, 10000);
    printf("  C_rand(d0) = %.6f\n", (double)c_rand);

    // --- Policy Evaluation of Greedy (Optimal) Policy ---
    // Use tabular policy with temp=0 (argmax) and Q* as preferences
    Policy* greedy_pol = tabular_policy_malloc(env->state_space, env->action_space);
    tabular_policy_update_from_preferences(greedy_pol, Q, 0.0f);
    Tensor* greedy_probs = tabular_policy_probabilities(greedy_pol);
    
    Tensor* Q_greedy = tensor2d_malloc(num_states, num_actions);
    Tensor* V_greedy = tensor1d_malloc(num_states);
    float v_greedy_pe = policy_evaluation(greedy_probs, Q_greedy, V_greedy, T, R, d0, gamma, 1e-8f, 10000);
    printf("\nGreedy Policy Evaluation:\n");
    printf("  V_greedy_PE(d0) = %.6f\n", (double)v_greedy_pe);
    printf("  V*(d0)          = %.6f\n", (double)v_star);
    printf("  Match? %s (diff=%.2e)\n",
           (fabsf(v_greedy_pe - v_star) < 1e-3f) ? "YES" : "NO",
           (double)fabsf(v_greedy_pe - v_star));

    // Cost PE of greedy policy
    Tensor* Q_greedy_c = tensor2d_malloc(num_states, num_actions);
    Tensor* V_greedy_c = tensor1d_malloc(num_states);
    float c_greedy = policy_evaluation_cost(greedy_probs, Q_greedy_c, V_greedy_c, T, C, d0, cost_gamma, 1e-8f, 10000);
    printf("  C_greedy(d0) = %.6f\n", (double)c_greedy);

    // --- Sanity check: V*(d0) >= V_rand(d0) ---
    printf("\nSanity check: V*(d0) >= V_rand(d0)? %s\n",
           (v_star >= v_rand - 1e-6f) ? "PASS" : "FAIL");

    // --- Rollout the greedy policy from VI ---
    int num_episodes = 1000;
    PlayStats greedy_stats = play(env, greedy_pol, num_episodes, gamma, cost_gamma, 1.0f);
    PlayStats random_stats = play(env, random_pol, num_episodes, gamma, cost_gamma, 1.0f);

    printf("\nRollout results (%d episodes, max_steps=%d):\n", num_episodes, max_steps);
    printf("  Greedy(Q*):  mean_return=%.4f  mean_cost=%.4f  disc_return=%.4f  disc_cost=%.4f\n",
           (double)greedy_stats.mean_return, (double)greedy_stats.mean_cost,
           (double)greedy_stats.mean_discounted_return, (double)greedy_stats.mean_discounted_cost);
    printf("  Random:      mean_return=%.4f  mean_cost=%.4f  disc_return=%.4f  disc_cost=%.4f\n",
           (double)random_stats.mean_return, (double)random_stats.mean_cost,
           (double)random_stats.mean_discounted_return, (double)random_stats.mean_discounted_cost);

    // ===========================================================================
    // CMDP: Cost-constrained Value Iteration
    // ===========================================================================
    float cost_threshold = 25.0f;
    printf("\n=== CMDP Value Iteration (cost_threshold=%.1f) ===\n", (double)cost_threshold);

    Tensor* cmdp_probs  = tensor2d_malloc(num_states, num_actions);
    Tensor* Q_cmdp      = tensor2d_malloc(num_states, num_actions);
    Tensor* V_cmdp      = tensor1d_malloc(num_states);
    Tensor* Q_cmdp_c    = tensor2d_malloc(num_states, num_actions);
    Tensor* V_cmdp_c    = tensor1d_malloc(num_states);
    float cmdp_cost_d0 = 0;

    float cmdp_v0 = cmdp_value_iteration(cmdp_probs, Q_cmdp, V_cmdp, Q_cmdp_c, V_cmdp_c,
                                            T, R, C, d0,
                                            gamma, cost_gamma, cost_threshold,
                                            100000, 1e-3f, 100000,
                                            &cmdp_cost_d0);

    printf("  V_cmdp(d0)  = %.6f\n", (double)cmdp_v0);
    printf("  C_cmdp(d0)  = %.6f  (threshold=%.1f)\n", (double)cmdp_cost_d0, (double)cost_threshold);
    printf("  Feasible?   %s\n", (cmdp_cost_d0 <= cost_threshold + 1e-6f) ? "YES" : "NO");

    // Compare with unconstrained
    printf("  V*(d0)      = %.6f  (unconstrained)\n", (double)v_star);
    printf("  C*(d0)      = %.6f  (unconstrained greedy cost)\n", (double)c_greedy);

    // Check that the CMDP policy is actually stochastic where it differs
    int stochastic_states = 0;
    for (int s = 0; s < num_states; s++) {
        float max_p = 0;
        for (int a = 0; a < num_actions; a++) {
            float p = tensor2d_get_at(cmdp_probs, s, a);
            if (p > max_p) max_p = p;
        }
        if (max_p < 1.0f - 1e-6f) stochastic_states++;
    }
    printf("  Stochastic states: %d / %d\n", stochastic_states, num_states);

    // Rollout the CMDP policy
    Policy* cmdp_pol = tabular_policy_malloc(env->state_space, env->action_space);
    tabular_policy_update(cmdp_pol, cmdp_probs);

    PlayStats cmdp_stats = play(env, cmdp_pol, num_episodes, gamma, cost_gamma, 1.0f);
    printf("\n  CMDP rollout (%d episodes):\n", num_episodes);
    printf("    mean_return=%.4f  mean_cost=%.4f  disc_return=%.4f  disc_cost=%.4f\n",
           (double)cmdp_stats.mean_return, (double)cmdp_stats.mean_cost,
           (double)cmdp_stats.mean_discounted_return, (double)cmdp_stats.mean_discounted_cost);

    // Cleanup
    tensor_free(T);
    tensor_free(R);
    tensor_free(C);
    tensor_free(d0);
    tensor_free(Q);
    tensor_free(V);
    tensor_free(Q_rand);
    tensor_free(V_rand);
    tensor_free(Q_rand_c);
    tensor_free(V_rand_c);
    tensor_free(Q_greedy);
    tensor_free(V_greedy);
    tensor_free(Q_greedy_c);
    tensor_free(V_greedy_c);
    tensor_free(probs);
    tensor_free(cmdp_probs);
    tensor_free(Q_cmdp);
    tensor_free(V_cmdp);
    tensor_free(Q_cmdp_c);
    tensor_free(V_cmdp_c);
    policy_free(greedy_pol);
    policy_free(random_pol);
    policy_free(cmdp_pol);
    mdp_free(env);

    return 0;
}
