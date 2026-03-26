#ifndef PPO_H
#define PPO_H

#include "rl/core/mdp.h"
#include "rl/algos/actor.h"

// ===========================================================================
// PPO hyperparameter config  (all tunables in one struct)
// ===========================================================================

typedef struct {
    float gamma;           // discount factor
    float lam;             // GAE lambda
    float actor_lr;        // actor learning rate
    float critic_lr;       // critic learning rate
    int      nsteps;          // rollout steps per env per iteration (0 = auto: 2048/N)
    int      nepochs;         // SGD epochs per iteration
    int      batch_size;      // minibatch size
    float clip_ratio;      // PPO clipping epsilon
    float entropy_coef;    // entropy bonus coefficient
    float max_grad_norm;   // gradient clipping max norm
    float beta1;           // Adam beta1
    float beta2;           // Adam beta2
    float adam_eps;        // Adam epsilon
    int      stats_window;    // sliding window size for mean reward logging
    // Target KL for early stopping (0 = disabled)
    float target_kl;
    // Linear decay flags and min values
    bool     decay_lr;        // linearly anneal actor_lr, critic_lr
    float min_lr;          // floor for learning rate decay
    bool     decay_clip_ratio; // linearly anneal clip_ratio
    float min_clip_ratio;  // floor for clip ratio decay
    bool     decay_entropy_coef; // linearly anneal entropy_coef
    float min_entropy_coef;   // floor for entropy coef decay
    // Entropy targeting: auto-tune entropy_coef to match a target entropy
    // Set entropy_target > 0 to enable. Default: 0 (disabled).
    // A good default when enabled is 0.5 * log(n_actions).
    float entropy_target;       // target entropy (0 = disabled)
    float entropy_target_lr;    // step size for entropy coef adjustment
    bool     reward_norm;        // normalise rewards by std of discounted returns (default false)
    // CMDP (Lagrangian penalty) — requires cost_critic passed to ppo_train
    bool     cmdp;               // enable CMDP mode
    float cost_gamma;         // discount factor for costs (1.0 = undiscounted)
    float cost_threshold;     // per-episode cost constraint
    float lagrange_lr;        // Lagrange multiplier learning rate
    float lagrange_init;      // initial Lagrange multiplier
    float lagrange_max;       // max Lagrange multiplier
    float moving_cost_alpha;  // EMA step size for per-episode cost estimate
    // Secondary reward shaping: reward += secondary_reward_coef * secondary_reward.
    // Default 0 (disabled). When > 0, secondary reward is folded into the optimised objective.
    float secondary_reward_coef;
    float primary_reward_coef;    // coefficient for primary reward (default 1.0, can be set to 0 to optimise purely for secondary reward)
} PPOConfig;

// Returns a PPOConfig with sensible defaults.
static inline PPOConfig ppo_default_config(void) {
    return (PPOConfig){
        .gamma          = 0.99,
        .lam            = 0.95,
        .actor_lr       = 3e-4,
        .critic_lr      = 3e-4,
        .nsteps         = 0,      // 0 = auto (2048 / num_envs)
        .nepochs        = 10,
        .batch_size     = 64,
        .clip_ratio     = 0.2,
        .entropy_coef   = 0.01,
        .max_grad_norm  = 0.5,
        .beta1          = 0.9,
        .beta2          = 0.999,
        .adam_eps        = 1e-7,
        .stats_window   = 100,
        .target_kl      = 0.0,    // 0 = disabled
        .decay_lr       = false,
        .min_lr         = 1.25e-5,
        .decay_clip_ratio = false,
        .min_clip_ratio = 0.01,
        .decay_entropy_coef = false,
        .min_entropy_coef  = 0.0,
        .entropy_target    = 0.0,    // 0 = disabled
        .entropy_target_lr = 3e-4,
        .reward_norm       = false,
        // CMDP defaults (disabled)
        .cmdp              = false,
        .cost_gamma        = 1.0,
        .cost_threshold    = 0.0,
        .lagrange_lr       = 0.01,
        .lagrange_init     = 0.0,
        .lagrange_max      = 1000.0,
        .moving_cost_alpha = 0.01,
        .secondary_reward_coef = 0.0,
        .primary_reward_coef = 1.0,
    };
}

// PPO training for discrete action spaces (vectorized environment).
// actor:       Policy* with ActorModel (created via actor_mlp_policy_malloc).
// critic:      SequentialNN* mapping obs -> scalar value (obs_dim -> 1).
// cost_critic: optional SequentialNN* for CMDP cost value (NULL when cfg.cmdp is false).
// log_path:    optional CSV log file path (NULL to disable).
void ppo_train(VecEnv* venv, Policy* actor, SequentialNN* critic,
               SequentialNN* cost_critic, int num_iters, PPOConfig cfg,
               const char* log_path);

#endif // PPO_H
