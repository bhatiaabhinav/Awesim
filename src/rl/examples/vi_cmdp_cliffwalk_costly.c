#include <stdio.h>
#include <time.h>
#include "rl/envs/gridworld.h"
#include "rl/algos/vi.h"

int main(void) {
    srand((unsigned int)time(NULL));

    MDP* env = cliffwalk_costly_create();
    int num_states  = env->state_space.n;
    int num_actions = env->action_space.n;
    float gamma      = 0.99f;
    float cost_gamma  = 0.99f;

    printf("=== CliffWalk Costly (CMDP) ===\n\n");
    gridworld_print(env);

    // --- Extract tabular dynamics ---
    Tensor* T  = mdp_malloc_transition_matrix(env);
    Tensor* R  = mdp_malloc_reward_matrix(env);
    Tensor* C  = mdp_malloc_cost_matrix(env);
    Tensor* d0 = mdp_malloc_start_state_probs(env);
    mdp_extract_tabular_dynamics(env, T, R, d0);
    mdp_extract_cost_matrix(env, C);

    // ===================================================================
    // 1. Random policy
    // ===================================================================
    printf("\n--- Random Policy ---\n");

    Policy* random_pol = random_policy_malloc(env->state_space, env->action_space);
    Tensor* rand_probs = policy_malloc_probs_matrix(random_pol);
    policy_extract_tabular_action_probabilities(random_pol, rand_probs);

    Tensor* Q_rand   = tensor2d_malloc(num_states, num_actions);
    Tensor* V_rand   = tensor1d_malloc(num_states);
    Tensor* Q_rand_c = tensor2d_malloc(num_states, num_actions);
    Tensor* V_rand_c = tensor1d_malloc(num_states);

    float v_rand = policy_evaluation(rand_probs, Q_rand, V_rand, T, R, d0, gamma, 1e-8f, 10000);
    float c_rand = policy_evaluation_cost(rand_probs, Q_rand_c, V_rand_c, T, C, d0, cost_gamma, 1e-8f, 10000);
    printf("  V_random(d0) = %.4f\n", (double)v_rand);
    printf("  C_random(d0) = %.4f\n", (double)c_rand);

    int num_episodes = 10000;
        PlayStats rand_stats = play(env, random_pol, num_episodes, gamma, cost_gamma, 1.0f);
        printf("  Rollout (%d eps): mean_return=%.4f  disc_return=%.4f  disc_cost=%.4f\n",
            num_episodes, (double)rand_stats.mean_return,
            (double)rand_stats.mean_discounted_return,
            (double)rand_stats.mean_discounted_cost);

    // ===================================================================
    // 2. Greedy unconstrained policy (VI)
    // ===================================================================
    printf("\n--- Greedy Unconstrained Policy (VI) ---\n");

    Tensor* Q_vi = tensor2d_malloc(num_states, num_actions);
    Tensor* V_vi = tensor1d_malloc(num_states);
    float v_star = value_iteration(Q_vi, V_vi, T, R, d0, gamma, 1e-8f, 10000);
    Policy* greedy_pol = tabular_policy_malloc(env->state_space, env->action_space);
    tabular_policy_update_from_preferences(greedy_pol, Q_vi, 0.0f);
    printf("  V*(d0)  = %.4f\n", (double)v_star);

    // Cost of the greedy policy
    Tensor* greedy_probs = tabular_policy_probabilities(greedy_pol);

    Tensor* Q_vi_c = tensor2d_malloc(num_states, num_actions);
    Tensor* V_vi_c = tensor1d_malloc(num_states);
    float c_greedy = policy_evaluation_cost(greedy_probs, Q_vi_c, V_vi_c, T, C, d0, cost_gamma, 1e-8f, 10000);
    printf("  C*(d0)  = %.4f  (unconstrained cost)\n", (double)c_greedy);

    printf("\nValue function (unconstrained):\n");
    gridworld_print_values(env, V_vi);
    printf("\nCost value function (unconstrained):\n");
    gridworld_print_values(env, V_vi_c);
    printf("\nPolicy (unconstrained):\n");
    gridworld_print_policy(env, Q_vi);
    gridworld_render_svg(env, greedy_pol, NULL, "output/viz/examples/cliffwalk_policy_unconstrained.svg");

        PlayStats greedy_stats = play(env, greedy_pol, num_episodes, gamma, cost_gamma, 1.0f);
        printf("\n  Rollout (%d eps): mean_return=%.4f  disc_return=%.4f  disc_cost=%.4f\n",
            num_episodes, (double)greedy_stats.mean_return,
            (double)greedy_stats.mean_discounted_return,
            (double)greedy_stats.mean_discounted_cost);

    // ===================================================================
    // 3. CMDP policy
    // ===================================================================
    float cost_threshold = 1.0f;
    printf("\n--- CMDP Policy (cost_threshold=%.2f) ---\n", (double)cost_threshold);

    Tensor* cmdp_probs = tensor2d_malloc(num_states, num_actions);
    Tensor* Q_cmdp     = tensor2d_malloc(num_states, num_actions);
    Tensor* V_cmdp     = tensor1d_malloc(num_states);
    Tensor* Q_cmdp_c   = tensor2d_malloc(num_states, num_actions);
    Tensor* V_cmdp_c   = tensor1d_malloc(num_states);
    float cmdp_cost_d0 = 0;

    float cmdp_v0 = cmdp_value_iteration(cmdp_probs, Q_cmdp, V_cmdp, Q_cmdp_c, V_cmdp_c,
                                            T, R, C, d0,
                                            gamma, cost_gamma, cost_threshold,
                                            100000, 1e-3f, 100000,
                                            &cmdp_cost_d0);
    // build tabular policy directly from CMDP probs (no softmax)
    Policy* cmdp_pol = tabular_policy_malloc(env->state_space, env->action_space);
    tabular_policy_update(cmdp_pol, cmdp_probs);

    printf("  V_cmdp(d0)  = %.4f\n", (double)cmdp_v0);
    printf("  C_cmdp(d0)  = %.4f  (threshold=%.2f)\n", (double)cmdp_cost_d0, (double)cost_threshold);
    printf("  Feasible?   %s\n", (cmdp_cost_d0 <= cost_threshold + 1e-4f) ? "YES" : "NO");

    // Count stochastic states
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

    printf("\nValue function (CMDP):\n");
    gridworld_print_values(env, V_cmdp);
    printf("\nCost value function (CMDP):\n");
    gridworld_print_values(env, V_cmdp_c);
    printf("\nPolicy (CMDP, dominant action):\n");
    gridworld_print_policy(env, Q_cmdp);
    gridworld_render_svg(env, cmdp_pol, NULL, "output/viz/examples/cliffwalk_policy_cmdp.svg");

    // Rollout:

        PlayStats cmdp_stats = play(env, cmdp_pol, num_episodes, gamma, cost_gamma, 1.0f);
        printf("\n  Rollout (%d eps): mean_return=%.4f  disc_return=%.4f  disc_cost=%.4f\n",
            num_episodes, (double)cmdp_stats.mean_return,
            (double)cmdp_stats.mean_discounted_return,
            (double)cmdp_stats.mean_discounted_cost);

    // ===================================================================
    // Cleanup
    // ===================================================================
    tensor_free(T);
    tensor_free(R);
    tensor_free(C);
    tensor_free(d0);
    tensor_free(rand_probs);
    tensor_free(Q_rand);
    tensor_free(V_rand);
    tensor_free(Q_rand_c);
    tensor_free(V_rand_c);
    tensor_free(Q_vi);
    tensor_free(V_vi);
    tensor_free(Q_vi_c);
    tensor_free(V_vi_c);
    tensor_free(cmdp_probs);
    tensor_free(Q_cmdp);
    tensor_free(V_cmdp);
    tensor_free(Q_cmdp_c);
    tensor_free(V_cmdp_c);
    policy_free(random_pol);
    policy_free(greedy_pol);
    policy_free(cmdp_pol);
    mdp_free(env);
    return 0;
}
