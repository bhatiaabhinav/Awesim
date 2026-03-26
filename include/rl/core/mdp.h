#ifndef MDP_H
#define MDP_H

#include <stdbool.h>
#include <stdio.h>
#include <pthread.h>
#include "nn.h"

// ---- Spaces ----

typedef enum { SPACE_TYPE_DISCRETE, SPACE_TYPE_BOX } SpaceType;

typedef struct Space {
    SpaceType type;
    int n;          // discrete: number of actions; box: dimensionality
    double* lows;   // only for box (always double — physical env ranges)
    double* highs;  // only for box
} Space;


// ===========================================================================
// MDP
// ===========================================================================

typedef struct MDP MDP;
typedef struct Policy Policy;

struct MDP {
    const char* name;
    Space state_space;
    Space action_space;
    void* config;
    void* state;
    void* observation;
    float reward;
    float cost;
    float secondary_reward;
    bool terminated;
    bool truncated;
    int step_count;
    void (*reset)(MDP* env);
    void (*step)(MDP* env, void* action);
    void (*free_config)(MDP* env);
    void (*free_state)(MDP* env);
    float (*transition_probability)(MDP* env, int s, int a, int s_next);
    float (*reward_function)(MDP* env, int s, int a, int s_next);
    float (*cost_function)(MDP* env, int s, int a, int s_next);
    float (*secondary_reward_function)(MDP* env, int s, int a, int s_next);
    float (*start_state_probability)(MDP* env, int s);
    // Linear feature function for secondary rewards: dot(features_out, w) = secondary_reward(s, a, s_next).
    // features_out must have space for secondary_reward_linear_features_dim floats.
    int secondary_reward_linear_features_dim;
    void (*secondary_reward_linear_features)(MDP* env, int s, int a, int s_next, float* features_out);
};

MDP* mdp_malloc(const char* name, Space state_space, Space action_space, void* config, void* state,
                void (*reset)(MDP*), void (*step)(MDP*, void*), void (*free_config)(MDP*), void (*free_state)(MDP*));
void mdp_free(MDP* env);

static inline void mdp_reset(MDP* env) { env->reset(env); }
static inline void mdp_step(MDP* env, void* action) { env->step(env, action); }

Tensor* mdp_malloc_transition_matrix(MDP* env);
Tensor*  mdp_malloc_reward_matrix(MDP* env);
Tensor*   mdp_malloc_start_state_probs(MDP* env);
Tensor*  mdp_malloc_cost_matrix(MDP* env);
Tensor*  mdp_malloc_secondary_reward_matrix(MDP* env);
void mdp_extract_tabular_dynamics(MDP* env, Tensor* transition_matrix, Tensor* reward_matrix, Tensor* start_state_probs);
void mdp_extract_cost_matrix(MDP* env, Tensor* cost_matrix);
void mdp_extract_secondary_reward_matrix(MDP* env, Tensor* secondary_reward_matrix);
void mdp_extract_secondary_reward_linear_features(MDP* env, Tensor* secondary_reward_linear_features);

// ===========================================================================
// Policy
// ===========================================================================

struct Policy {
    const char* name;
    Space observation_space;
    Space action_space;
    void* action;
    void* model;
    void (*act)(Policy* policy, void* observation);
    void (*free_model)(Policy* policy);
    float (*action_probability)(Policy* policy, int state, int action);
};

Policy* policy_malloc(const char* name, Space observation_space, Space action_space, void* model, void (*act)(Policy*, void*), void (*free_model)(Policy*));
Policy* policy_free(Policy* policy);
static inline void policy_act(Policy* policy, void* observation) { policy->act(policy, observation); }

Tensor* policy_malloc_probs_matrix(Policy* policy);
void policy_extract_tabular_action_probabilities(Policy* policy, Tensor* probs_matrix);

// ---- Random policy ----

Policy* random_policy_malloc(Space observation_space, Space action_space);

// ---- Tabular policy ----
typedef struct TabularPolicyModel {
    Tensor* probabilities;
} TabularPolicyModel;

Policy* tabular_policy_malloc(Space state_space, Space action_space);
Policy* tabular_policy_update(Policy* policy, Tensor* probabilities);
Policy* tabular_policy_update_from_preferences(Policy* policy, Tensor* preferences, float temperature);
Tensor* tabular_policy_probabilities(Policy* policy);

// ---- Play / rollout ----

typedef struct {
    float gamma;
    float cost_gamma;
    float secondary_reward_gamma;

    int num_episodes;
    float total_reward;
    float min_return;
    float max_return;
    float mean_episode_length;
    float mean_return;
    float mean_cost;
    float mean_secondary_reward;
    float mean_discounted_return;
    float mean_discounted_cost;
    float mean_discounted_secondary_return;
    float var_discounted_return;            // variance of discounted returns
    float var_discounted_cost;              // variance of discounted costs
    float var_discounted_secondary_return;  // variance of discounted secondary returns
    float m2_discounted_return;             // Welford M2 accumulator (infinite window only)
    float m2_discounted_cost;
    float m2_discounted_secondary_return;

    int cur_episode_length;
    float gamma_t;
    float cost_gamma_t;
    float secondary_reward_gamma_t;
    float cur_episode_reward;
    float cur_episode_cost;
    float cur_episode_secondary_reward;
    float cur_episode_disc_return;
    float cur_episode_disc_cost;
    float cur_episode_disc_secondary_return;

    // sliding window (0 = infinite / use all episodes)
    int stats_window;
    int buf_count;     // min(num_episodes, stats_window)
    int buf_idx;       // next write position in circular buffer
    float* buf_return;
    float* buf_cost;
    float* buf_secondary_reward;
    float* buf_disc_return;
    float* buf_disc_cost;
    float* buf_disc_secondary_return;
    float* buf_episode_length;
} PlayStats;

PlayStats play_stats_init(float gamma, float cost_gamma, float secondary_reward_gamma, int stats_window);

void play_stats_push(PlayStats* stats, float reward, float cost, float secondary_reward, bool done);

void play_stats_merge(PlayStats* dest, const PlayStats* src);  // merge src into dest (for combining parallel stats)

void play_stats_reset(PlayStats* stats);

void play_stats_free_buffers(PlayStats* stats);

PlayStats play(MDP* env, Policy* policy, int num_episodes, float gamma, float cost_gamma, float secondary_reward_gamma);

// Sample an action index from a discrete probability distribution over num_actions actions.
int sample_action(const float* probs, int num_actions);

// policy entropy
float policy_entropy(const Tensor* probs_matrix);

// ===========================================================================
// Vectorized Environment  (async: fixed thread pool, auto-resets on done)
// ===========================================================================

typedef struct VecEnv VecEnv;

typedef struct {
    int     thread_idx;   // index of this worker thread
    int     num_envs;     // how many envs this thread handles
    int*    env_indices;  // array of env indices assigned to this thread
    VecEnv* venv;
} VecEnvWorker;

struct VecEnv {
    int       num_envs;
    int       num_threads;   // number of worker threads (<= num_envs)
    bool      async;
    MDP**     envs;
    // Step results (captured by workers before auto-reset)
    float* step_rewards;
    float* step_costs;
    float* step_secondary_rewards;
    bool*     step_terminated;
    bool*     step_truncated;
    // Thread pool
    pthread_t*        threads;
    VecEnvWorker*     workers;
    pthread_mutex_t   mutex;
    pthread_cond_t    work_cond;
    pthread_cond_t    done_cond;
    void**            step_actions;
    int               generation;
    int*              worker_gen;  // per-thread generation
    int               done_count;  // threads finished this step
    bool              running;
};

// num_threads=0 means auto-detect CPU cores (capped at num_envs)
VecEnv* vecenv_malloc(MDP* (*constructor)(void), int num_envs, bool async);
void    vecenv_step(VecEnv* venv, void** actions);
void    vecenv_reset(VecEnv* venv);
void    vecenv_free(VecEnv* venv);

#endif // CGYM_MDP_H
