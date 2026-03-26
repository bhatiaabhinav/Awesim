#include <time.h>
#include "rl/envs/cartpole.h"
#include "rl/algos/ppo.h"

static MDP* cartpole_create_default(void) {
    return cartpole_create(NULL);
}

int main(void) {
    srand(42);

    MDP* env = cartpole_create(NULL);

    // --- Random baseline ---
    Policy* random_pol = random_policy_malloc(env->state_space, env->action_space);
    PlayStats rs = play(env, random_pol, 100, 1.0f, 1.0f, 1.0f);
    printf("CartPole-v1 | Random Policy | 100 episodes\n");
    printf("  Mean reward: %.2f\n", rs.mean_return);
    printf("  Min  reward: %.2f\n", rs.min_return);
    printf("  Max  reward: %.2f\n", rs.max_return);
    policy_free(random_pol);

    // --- Actor & Critic ---
    int hidden[] = {64, 64};
    int obs_dim = env->state_space.n;
    int n_actions = env->action_space.n;

    // Actor: obs -> logits [n_actions]
    Policy* actor = actor_mlp_policy_malloc(obs_dim, n_actions, hidden, 2, 1.0f, 256 * 8);

    // Critic: obs -> V(s) [1]
    int critic_sizes[] = {obs_dim, 64, 64, 1};
    SequentialNN* critic = seq_nn_make_mlp(4, critic_sizes, false);

    // --- PPO training (vectorised) ---
    int num_envs = 8;
    VecEnv* venv = vecenv_malloc(cartpole_create_default, num_envs, false);
    PPOConfig cfg = ppo_default_config();
    cfg.nsteps = 256;
    cfg.batch_size = 128;
    printf("\n--- PPO Training (100 iters x %d steps x %d envs) ---\n",
           cfg.nsteps > 0 ? cfg.nsteps : 2048 / num_envs, num_envs);
    ppo_train(venv, actor, critic, NULL, 100, cfg, NULL);
    vecenv_free(venv);

    // --- Evaluation (deterministic: argmax of logits) ---
    printf("\n--- Evaluation after training ---\n");
    PlayStats ps = play(env, actor, 100, 1.0f, 1.0f, 1.0f);
    printf("CartPole-v1 | PPO (64-64) | 100 episodes\n");
    printf("  Mean reward: %.2f\n", ps.mean_return);
    printf("  Min  reward: %.2f\n", ps.min_return);
    printf("  Max  reward: %.2f\n", ps.max_return);
    printf("  Total reward: %.2f\n", ps.total_reward);

    policy_free(actor);
    seq_nn_free(critic);
    mdp_free(env);
    printf("Done.\n");
    return 0;
}
