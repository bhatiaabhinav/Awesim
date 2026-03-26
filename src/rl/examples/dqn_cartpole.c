#include <time.h>
#include "rl/envs/cartpole.h"
#include "rl/algos/dqn.h"

int main(void) {
    srand((unsigned int)time(NULL));

    MDP* env = cartpole_create(NULL);
    int num_episodes = 100;

    // --- Random policy ---
    Policy* random_pol = random_policy_malloc(env->state_space, env->action_space);
    PlayStats random_stats = play(env, random_pol, num_episodes, 1.0f, 1.0f, 1.0f);
    printf("CartPole-v1 | Random Policy | %d episodes\n", num_episodes);
    printf("  Mean reward: %.2f\n", random_stats.mean_return);
    printf("  Min  reward: %.2f\n", random_stats.min_return);
    printf("  Max  reward: %.2f\n", random_stats.max_return);
    printf("  Total reward: %.2f\n", random_stats.total_reward);
    random_pol = policy_free(random_pol);

    // --- DQN training ---
    int hidden[] = {128, 128};
    Policy* dqn_pol = dqn_policy_malloc(env->state_space, env->action_space, 2, hidden);

    printf("\n--- DQN Training (50k steps) ---\n");
    dqn_train(env, dqn_pol, 10000);
    printf("DQN training completed.\n");

    printf("\n--- Evaluation after training ---\n");
    PlayStats dqn_stats = play(env, dqn_pol, num_episodes, 1.0f, 1.0f, 1.0f);
    printf("CartPole-v1 | DQN Policy (128-128) | %d episodes\n", num_episodes);
    printf("  Mean reward: %.2f\n", dqn_stats.mean_return);
    printf("  Min  reward: %.2f\n", dqn_stats.min_return);
    printf("  Max  reward: %.2f\n", dqn_stats.max_return);
    printf("  Total reward: %.2f\n", dqn_stats.total_reward);

    policy_free(dqn_pol);
    mdp_free(env);
    return 0;
}
