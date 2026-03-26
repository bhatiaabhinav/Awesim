#include <time.h>
#include "rl/envs/gridworld.h"
#include "rl/algos/ppo.h"

static MDP* cliffwalk_continuous_create_default(void) {
    return cliffwalk_continuous_create();
}

int main(void) {
    srand((unsigned int)time(NULL));

    MDP* env = cliffwalk_continuous_create();

    // --- Random baseline ---
    Policy* random_pol = random_policy_malloc(env->state_space, env->action_space);
    PlayStats rs = play(env, random_pol, 100, 1.0f, 1.0f, 1.0f);
    printf("CliffWalk (continuous obs) | Random Policy | 100 episodes\n");
    printf("  Mean reward: %.2f\n", rs.mean_return);
    printf("  Min  reward: %.2f\n", rs.min_return);
    printf("  Max  reward: %.2f\n", rs.max_return);
    policy_free(random_pol);

    // --- Actor & Critic ---
    int hidden[] = {64, 64};
    int obs_dim = env->state_space.n;
    int n_actions = env->action_space.n;

    // Actor: obs(2) -> logits [5 actions]
    Policy* actor = actor_mlp_policy_malloc(obs_dim, n_actions, hidden, 2, 1.0f, 128 * 16);

    // Critic: obs(2) -> V(s) [1]
    int critic_sizes[] = {obs_dim, 64, 64, 1};
    SequentialNN* critic = seq_nn_make_mlp(4, critic_sizes, false);

    // --- PPO training (vectorised) ---
    int num_envs = 16;
    VecEnv* venv = vecenv_malloc(cliffwalk_continuous_create_default, num_envs, false);
    PPOConfig cfg = ppo_default_config();
    cfg.nsteps = 128;
    cfg.batch_size = 128;
    cfg.gamma = 0.99f;
    cfg.lam = 0.95f;
    cfg.actor_lr = 1e-4f;
    cfg.critic_lr = 1e-4f;
    cfg.decay_lr = false;
    cfg.entropy_coef = 0.1f;
    cfg.decay_entropy_coef = true;
    cfg.reward_norm = false;
    cfg.target_kl = 0.01f;
    printf("\n--- PPO Training (1000 iters x %d steps x %d envs) ---\n",
           cfg.nsteps, num_envs);
    ppo_train(venv, actor, critic, NULL, 1000, cfg, "output/logs/examples/ppo_cliffwalk_continuous.csv");
    vecenv_free(venv);

    // --- Evaluation ---
    printf("\n--- Evaluation after training ---\n");
    ActorModel* a_m = (ActorModel*)actor->model;
    a_m->deterministic = true; // use deterministic policy for evaluation (argmax action selection)
    PlayStats ps = play(env, actor, 100, 1.0f, 1.0f, 1.0f);
    printf("CliffWalk (continuous obs) | PPO (64-64) | 100 episodes\n");
    printf("  Mean reward: %.2f\n", ps.mean_return);
    printf("  Min  reward: %.2f\n", ps.min_return);
    printf("  Max  reward: %.2f\n", ps.max_return);
    printf("  Mean episode length: %.1f\n", ps.mean_episode_length);

    policy_free(actor);
    seq_nn_free(critic);
    mdp_free(env);
    printf("Done.\n");
    return 0;
}
