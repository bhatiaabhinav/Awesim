#include <time.h>
#include "rl/envs/gridworld.h"
#include "rl/algos/dqn.h"

int main(void) {
    srand((unsigned int)time(NULL));

    MDP* env = cliffwalk_continuous_create();
    int num_episodes = 100;

    // --- Random baseline ---
    Policy* random_pol = random_policy_malloc(env->state_space, env->action_space);
    PlayStats rs = play(env, random_pol, num_episodes, 1.0f, 1.0f, 1.0f);
    printf("CliffWalk (continuous obs) | Random Policy | %d episodes\n", num_episodes);
    printf("  Mean reward: %.2f\n", rs.mean_return);
    printf("  Min  reward: %.2f\n", rs.min_return);
    printf("  Max  reward: %.2f\n", rs.max_return);
    policy_free(random_pol);

    // --- DQN training ---
    int hidden[] = {64, 64};
    Policy* dqn_pol = dqn_policy_malloc(env->state_space, env->action_space, 2, hidden);

    printf("\n--- DQN Training (100k steps) ---\n");
    dqn_train(env, dqn_pol, 100000);
    printf("DQN training completed.\n");

    printf("\n--- Evaluation after training ---\n");
    PlayStats ds = play(env, dqn_pol, num_episodes, 1.0f, 1.0f, 1.0f);
    printf("CliffWalk (continuous obs) | DQN (64-64) | %d episodes\n", num_episodes);
    printf("  Mean reward: %.2f\n", ds.mean_return);
    printf("  Min  reward: %.2f\n", ds.min_return);
    printf("  Max  reward: %.2f\n", ds.max_return);
    printf("  Mean episode length: %.1f\n", ds.mean_episode_length);

    policy_free(dqn_pol);
    mdp_free(env);
    printf("Done.\n");
    return 0;
}
