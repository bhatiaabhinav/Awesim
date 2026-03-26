#include <time.h>
#include "rl/envs/gridworld.h"
#include "rl/algos/ppo.h"

static MDP* cliffwalk_costly_continuous_create_default(void) {
    return cliffwalk_costly_continuous_create();
}

int main(void) {
    srand((unsigned int)time(NULL));

    float gamma      = 0.99f;
    float cost_gamma  = 0.99f;
    float cost_threshold = 1.0f;

    MDP* env = cliffwalk_costly_continuous_create();

    // --- Random baseline ---
    Policy* random_pol = random_policy_malloc(env->state_space, env->action_space);
    PlayStats rs = play(env, random_pol, 100, gamma, cost_gamma, 1.0f);
    printf("CliffWalk Costly (continuous obs) | Random Policy | 100 episodes\n");
    printf("  Mean disc. return: %.4f\n", rs.mean_discounted_return);
    printf("  Mean disc. cost:   %.4f\n", rs.mean_discounted_cost);
    policy_free(random_pol);

    // --- Actor, Critic, Cost Critic ---
    int hidden[] = {64, 64};
    int obs_dim = env->state_space.n;
    int n_actions = env->action_space.n;

    Policy* actor  = actor_mlp_policy_malloc(obs_dim, n_actions, hidden, 2, 1.0f, 256 * 8);
    int critic_sizes[] = {obs_dim, 64, 64, 1};
    SequentialNN* critic = seq_nn_make_mlp(4, critic_sizes, false);
    SequentialNN* cost_critic = seq_nn_make_mlp(4, critic_sizes, false);

    // --- PPO-Lagrangian training (CMDP) ---
    int num_envs = 8;
    VecEnv* venv = vecenv_malloc(cliffwalk_costly_continuous_create_default, num_envs, false);
    PPOConfig cfg = ppo_default_config();
    cfg.nsteps = 256;
    cfg.batch_size = 128;
    cfg.gamma = gamma;
    cfg.lam = 0.95f;
    cfg.actor_lr = 1e-4f;
    cfg.critic_lr = 1e-4f;
    cfg.decay_lr = false;
    cfg.entropy_coef = 0.1f;
    cfg.decay_entropy_coef = false;
    cfg.reward_norm = false;
    cfg.target_kl = 0.01f;
    cfg.cmdp = true;
    cfg.cost_gamma = cost_gamma;
    cfg.cost_threshold = cost_threshold;
    cfg.stats_window = 100;
    cfg.lagrange_lr = 0.01f;
    cfg.moving_cost_alpha = 0.01f;
    cfg.lagrange_init = 0.0f;
    cfg.lagrange_max = 100.0f;

    printf("\n--- PPO-Lagrangian Training (2000 iters x %d steps x %d envs, cost_thresh=%.2f) ---\n",
           cfg.nsteps, num_envs, (double)cost_threshold);
    ppo_train(venv, actor, critic, cost_critic, 2000, cfg, "output/logs/examples/ppo_cliffwalk_costly_continuous.csv");
    vecenv_free(venv);

    // --- Evaluation ---
    printf("\n--- Evaluation after training ---\n");
    ActorModel* a_m = (ActorModel*)actor->model;
    a_m->deterministic = false;  // cmdp policies are stochastic
    int eval_episodes = 1000;
    PlayStats ps = play(env, actor, eval_episodes, gamma, cost_gamma, 1.0f);
    printf("CliffWalk Costly (continuous obs) | PPO-Lagrangian (64-64) | %d episodes\n", eval_episodes);
    printf("  Mean disc. return: %.4f\n", ps.mean_discounted_return);
    printf("  Mean disc. cost:   %.4f  (threshold=%.2f)\n", ps.mean_discounted_cost, (double)cost_threshold);
    printf("  Mean reward:       %.2f\n", ps.mean_return);
    printf("  Mean ep length:    %.1f\n", ps.mean_episode_length);
    printf("  Feasible?          %s\n", ps.mean_discounted_cost <= cost_threshold + 1e-3 ? "YES" : "NO");

    policy_free(actor);
    seq_nn_free(critic);
    seq_nn_free(cost_critic);
    mdp_free(env);
    printf("Done.\n");
    return 0;
}
