#include <stdio.h>
#include <time.h>
#include "rl/envs/gridworld.h"
#include "rl/algos/vi.h"

int main(void) {
    srand((unsigned int)time(NULL));

    MDP* env = cliffwalk_create();
    int num_states  = env->state_space.n;
    int num_actions = env->action_space.n;
    float gamma = 0.99f;
    float cost_gamma = 0.99f;

    printf("=== CliffWalk GridWorld ===\n\n");
    gridworld_print(env);

    // --- Extract tabular dynamics ---
    Tensor* T  = mdp_malloc_transition_matrix(env);
    Tensor* R  = mdp_malloc_reward_matrix(env);
    Tensor* C  = mdp_malloc_cost_matrix(env);
    Tensor* d0 = mdp_malloc_start_state_probs(env);
    mdp_extract_tabular_dynamics(env, T, R, d0);
    mdp_extract_cost_matrix(env, C);

    // --- Value Iteration ---
    Tensor* Q = tensor2d_malloc(num_states, num_actions);
    Tensor* V = tensor1d_malloc(num_states);

    float v_star = value_iteration(Q, V, T, R, d0, gamma, 1e-8f, 10000);
    printf("\nValue Iteration: V*(d0) = %.4f\n\n", (double)v_star);

    Policy* greedy_pol = tabular_policy_malloc(env->state_space, env->action_space);
    tabular_policy_update_from_preferences(greedy_pol, Q, 0.0f);

    gridworld_print_values(env, V);
    printf("\n");
    gridworld_print_policy(env, Q);
    gridworld_render_svg(env, greedy_pol, NULL, "output/viz/examples/cliffwalk_policy.svg");

    // --- Cost Policy Evaluation ---
    Policy* random_pol = random_policy_malloc(env->state_space, env->action_space);
    Tensor* rand_probs = policy_malloc_probs_matrix(random_pol);
    policy_extract_tabular_action_probabilities(random_pol, rand_probs);
    gridworld_render_svg(env, random_pol, NULL, "output/viz/examples/cliffwalk_random_policy.svg");

    Tensor* greedy_probs = tabular_policy_probabilities(greedy_pol);

    Tensor* Q_gc = tensor2d_malloc(num_states, num_actions);
    Tensor* V_gc = tensor1d_malloc(num_states);
    float c_greedy = policy_evaluation_cost(greedy_probs, Q_gc, V_gc, T, C, d0, cost_gamma, 1e-8f, 10000);

    Tensor* Q_rc = tensor2d_malloc(num_states, num_actions);
    Tensor* V_rc = tensor1d_malloc(num_states);
    float c_random = policy_evaluation_cost(rand_probs, Q_rc, V_rc, T, C, d0, cost_gamma, 1e-8f, 10000);

    printf("\nCost Policy Evaluation:\n");
    printf("  C_greedy(d0) = %.4f\n", (double)c_greedy);
    printf("  C_random(d0) = %.4f\n", (double)c_random);

    // --- Rollouts ---
    int num_episodes = 10000;
    PlayStats greedy_stats = play(env, greedy_pol, num_episodes, gamma, cost_gamma, 1.0f);
    PlayStats random_stats = play(env, random_pol, num_episodes, gamma, cost_gamma, 1.0f);

    printf("\nRollout results (%d episodes):\n", num_episodes);
    printf("  Greedy(Q*): mean_return=%.2f  mean_cost=%.4f  disc_return=%.4f  disc_cost=%.4f\n",
           (double)greedy_stats.mean_return, (double)greedy_stats.mean_cost,
           (double)greedy_stats.mean_discounted_return, (double)greedy_stats.mean_discounted_cost);
    printf("  Random:     mean_return=%.2f  mean_cost=%.4f  disc_return=%.4f  disc_cost=%.4f\n",
           (double)random_stats.mean_return, (double)random_stats.mean_cost,
           (double)random_stats.mean_discounted_return, (double)random_stats.mean_discounted_cost);



    // Cleanup
    tensor_free(T);
    tensor_free(R);
    tensor_free(C);
    tensor_free(d0);
    tensor_free(Q);
    tensor_free(V);
    tensor_free(Q_gc);
    tensor_free(V_gc);
    tensor_free(Q_rc);
    tensor_free(V_rc);
    tensor_free(rand_probs);
    policy_free(greedy_pol);
    policy_free(random_pol);
    mdp_free(env);
    return 0;
}
