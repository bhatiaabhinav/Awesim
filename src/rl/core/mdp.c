#include <string.h>
#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif
#include "rl/core/mdp.h"
#include <limits.h> 

// ===========================================================================
// MDP
// ===========================================================================

MDP* mdp_malloc(const char* name, Space state_space, Space action_space, void* config, void* state,
                void (*reset)(MDP*), void (*step)(MDP*, void*), void (*free_config)(MDP*), void (*free_state)(MDP*)) {
    MDP* env = (MDP*)malloc(sizeof(MDP));
    env->name = name;
    env->state_space = state_space;
    env->action_space = action_space;
    env->config = config;
    env->state = state;
    if (state_space.type == SPACE_TYPE_BOX) {
        env->observation = malloc(sizeof(float) * state_space.n);
    } else {
        env->observation = malloc(sizeof(int));
    }
    env->reward = 0;
    env->cost = 0;
    env->secondary_reward = 0;
    env->terminated = false;
    env->truncated = false;
    env->step_count = 0;
    env->reset = reset;
    env->step = step;
    env->free_config = free_config;
    env->free_state = free_state;
    env->transition_probability = NULL;
    env->reward_function = NULL;
    env->cost_function = NULL;
    env->secondary_reward_function = NULL;
    env->start_state_probability = NULL;
    env->secondary_reward_linear_features_dim = 0;
    env->secondary_reward_linear_features = NULL;
    return env;
}

void mdp_free(MDP* env) {
    if (env) {
        if (env->free_config) env->free_config(env);
        if (env->free_state)  env->free_state(env);
        if (env->config) free(env->config); // free env-specific config if not already freed
        if (env->state)  free(env->state);  // free env-specific state if not already freed
        free(env->observation);
        env->observation = NULL;
        if (env->state_space.type == SPACE_TYPE_BOX) {
            free(env->state_space.lows);
            env->state_space.lows = NULL;
            free(env->state_space.highs);
            env->state_space.highs = NULL;
        }
        if (env->action_space.type == SPACE_TYPE_BOX) {
            free(env->action_space.lows);
            env->action_space.lows = NULL;
            free(env->action_space.highs);
            env->action_space.highs = NULL;
        }
        free(env);
    }
}

Tensor* mdp_malloc_transition_matrix(MDP* env) {
    if (env->state_space.type != SPACE_TYPE_DISCRETE || env->action_space.type != SPACE_TYPE_DISCRETE) {
        fprintf(stderr, "Error: Transition matrix can only be allocated for discrete state and action spaces.\n");
        return NULL;
    }
    return tensor3d_malloc(env->state_space.n, env->action_space.n, env->state_space.n);
}

Tensor*  mdp_malloc_reward_matrix(MDP* env) {
    if (env->state_space.type != SPACE_TYPE_DISCRETE || env->action_space.type != SPACE_TYPE_DISCRETE) {
        fprintf(stderr, "Error: Reward matrix can only be allocated for discrete state and action spaces.\n");
        return NULL;
    }
    return tensor3d_malloc(env->state_space.n, env->action_space.n, env->state_space.n);
}
Tensor*   mdp_malloc_start_state_probs(MDP* env) {
    if (env->state_space.type != SPACE_TYPE_DISCRETE) {
        fprintf(stderr, "Error: Start state probabilities can only be allocated for discrete state spaces.\n");
        return NULL;
    }
    return tensor1d_malloc(env->state_space.n);
}

Tensor* mdp_malloc_cost_matrix(MDP* env) {
    if (env->state_space.type != SPACE_TYPE_DISCRETE || env->action_space.type != SPACE_TYPE_DISCRETE) {
        fprintf(stderr, "Error: Cost matrix can only be allocated for discrete state and action spaces.\n");
        return NULL;
    }
    return tensor3d_malloc(env->state_space.n, env->action_space.n, env->state_space.n);
}

Tensor* mdp_malloc_secondary_reward_matrix(MDP* env) {
    if (env->state_space.type != SPACE_TYPE_DISCRETE || env->action_space.type != SPACE_TYPE_DISCRETE) {
        fprintf(stderr, "Error: Secondary reward matrix can only be allocated for discrete state and action spaces.\n");
        return NULL;
    }
    return tensor3d_malloc(env->state_space.n, env->action_space.n, env->state_space.n);
}

void mdp_extract_tabular_dynamics(MDP* env, Tensor* transition_matrix, Tensor* reward_matrix, Tensor* start_state_probs) {
    if (!env->transition_probability || !env->reward_function || !env->start_state_probability) {
        fprintf(stderr, "Error: MDP does not have the necessary functions to extract tabular dynamics.\n");
        return;
    }
    Tensor* T = transition_matrix;
    Tensor* R = reward_matrix;
    Tensor* d0 = start_state_probs;
    tensor_assert_shape(T, env->state_space.n, env->action_space.n, env->state_space.n, 0);
    tensor_assert_shape(R, env->state_space.n, env->action_space.n, env->state_space.n, 0);
    tensor_assert_shape(d0, env->state_space.n, 0, 0, 0);
    for (int s = 0; s < env->state_space.n; s++) {
        for (int a = 0; a < env->action_space.n; a++) {
            for (int s_next = 0; s_next < env->state_space.n; s_next++) {
               tensor3d_set_at(T, s, a, s_next, env->transition_probability(env, s, a, s_next));
               tensor3d_set_at(R, s, a, s_next, env->reward_function(env, s, a, s_next));
            }
        }
        tensor1d_set_at(d0, s, env->start_state_probability(env, s));
    }
}

void mdp_extract_cost_matrix(MDP* env, Tensor* cost_matrix) {
    if (!env->cost_function) {
        fprintf(stderr, "Error: MDP does not have a cost_function to extract cost matrix.\n");
        return;
    }
    tensor_assert_shape(cost_matrix, env->state_space.n, env->action_space.n, env->state_space.n, 0);
    for (int s = 0; s < env->state_space.n; s++) {
        for (int a = 0; a < env->action_space.n; a++) {
            for (int s_next = 0; s_next < env->state_space.n; s_next++) {
                tensor3d_set_at(cost_matrix, s, a, s_next, env->cost_function(env, s, a, s_next));
            }
        }
    }
}

void mdp_extract_secondary_reward_matrix(MDP* env, Tensor* secondary_reward_matrix) {
    if (!env->secondary_reward_function) {
        fprintf(stderr, "Error: MDP does not have a secondary_reward_function to extract secondary reward matrix.\n");
        return;
    }
    // Use tensor dimensions (discrete state count) rather than state_space.n,
    // which may be the observation dimension in continuous-state environments.
    int S = secondary_reward_matrix->n1;
    int A = env->action_space.n;
    tensor_assert_shape(secondary_reward_matrix, S, A, S, 0);
    for (int s = 0; s < S; s++) {
        for (int a = 0; a < A; a++) {
            for (int s_next = 0; s_next < S; s_next++) {
                tensor3d_set_at(secondary_reward_matrix, s, a, s_next, env->secondary_reward_function(env, s, a, s_next));
            }
        }
    }
}

void mdp_extract_secondary_reward_linear_features(MDP* env, Tensor* secondary_reward_linear_features) {
    if (!env->secondary_reward_linear_features) {
        fprintf(stderr, "Error: MDP does not have a secondary_reward_linear_features function to extract features.\n");
        return;
    }
    int S = env->state_space.n;
    int A = env->action_space.n;
    int F = env->secondary_reward_linear_features_dim;
    tensor_assert_shape(secondary_reward_linear_features, S, A, S, F);
    
    for (int s = 0; s < S; s++) {
        for (int a = 0; a < A; a++) {
            for (int s_next = 0; s_next < S; s_next++) {
                float* features_out = tensor4d_at(secondary_reward_linear_features, s, a, s_next, 0);
                env->secondary_reward_linear_features(env, s, a, s_next, features_out);
            }
        }
    }
}

// ===========================================================================
// Policy
// ===========================================================================

Policy* policy_malloc(const char* name, Space observation_space, Space action_space, void* model, void (*act)(Policy*, void*), void (*free_model)(Policy*)) {
    Policy* p = (Policy*)malloc(sizeof(Policy));
    p->name = name;
    p->observation_space = observation_space;
    p->action_space = action_space;
    p->model = model;
    
    if (action_space.type == SPACE_TYPE_DISCRETE) {
        p->action = malloc(sizeof(int));
    } else {
        p->action = malloc(sizeof(float) * action_space.n);
    }
    p->act = act;
    p->free_model = free_model;
    p->action_probability = NULL;
    return p;
}

Policy* policy_free(Policy* p) {
    if (p) {
        if (p->free_model) {
            p->free_model(p);
        }
        if (p->model) {
            free(p->model); // free policy-specific model if not already freed by free_model
            p->model = NULL;
        }
        free(p->action);
        p->action = NULL;
        free(p);
    }
    return NULL;
}

Tensor* policy_malloc_probs_matrix(Policy* policy) {
    if (policy->action_space.type != SPACE_TYPE_DISCRETE || policy->observation_space.type != SPACE_TYPE_DISCRETE) {
        fprintf(stderr, "Error: Action probabilities can only be allocated for policies with discrete observation and action spaces.\n");
        return NULL;
    }
    return tensor2d_malloc(policy->observation_space.n, policy->action_space.n);
}

void policy_extract_tabular_action_probabilities(Policy* policy, Tensor* probs_matrix) {
    if (policy->action_probability == NULL) {
        fprintf(stderr, "Error: Policy does not have an action_probability function.\n");
        return;
    }
    tensor_assert_shape(probs_matrix, policy->observation_space.n, policy->action_space.n, 0, 0);
    for (int s = 0; s < policy->observation_space.n; s++) {
        for (int a = 0; a < policy->action_space.n; a++) {
            tensor2d_set_at(probs_matrix, s, a, policy->action_probability(policy, s, a));
        }
    }
}

// ---- Random policy ----

static void random_policy_act(Policy* p, void* observation) {
    if (p->action_space.type == SPACE_TYPE_DISCRETE) {
        *(int*)p->action = rand() % p->action_space.n;
    } else if (p->action_space.type == SPACE_TYPE_BOX) {
        float* action_vec = (float*)p->action;
        for (int i = 0; i < p->action_space.n; i++) {
            action_vec[i] = (float)rand_uniform(p->action_space.lows[i],
                                                   p->action_space.highs[i]);
        }
    }
}

static float random_policy_action_probability(Policy* p, int state, int action) {
    (void)state; (void)action;
    return (float)1.0 / p->action_space.n;
}

Policy* random_policy_malloc(Space observation_space, Space action_space) {
    Policy* pol = policy_malloc("random", observation_space, action_space, NULL, random_policy_act, NULL);
    pol->action_probability = random_policy_action_probability;
    return pol;
}

// ---- Tabular policy ----

static void tabular_policy_act(Policy* p, void* observation) {
    int s = *(int*)observation;
    TabularPolicyModel* model = (TabularPolicyModel*)p->model;
    int n = p->action_space.n;
    float* probs_s = mat_row_slice(model->probabilities->data, s, n);
    mat_row_sample((int*)p->action, probs_s, 1, n);
}

void tabular_policy_free_model(Policy* p) {
    TabularPolicyModel* model = (TabularPolicyModel*)p->model;
    if (model) {
        tensor_free(model->probabilities);
        free(model);
        p->model = NULL;
    }
}

static float tabular_policy_action_probability(Policy* p, int state, int action) {
    TabularPolicyModel* model = (TabularPolicyModel*)p->model;
    int n = p->action_space.n;
    float* probs_s = mat_row_slice(model->probabilities->data, state, n);
    return probs_s[action];
}

Policy* tabular_policy_malloc(Space state_space, Space action_space) {
    TabularPolicyModel* model = (TabularPolicyModel*)malloc(sizeof(TabularPolicyModel));
    model->probabilities = tensor2d_malloc(state_space.n, action_space.n);

    Policy* pol = policy_malloc("tabular", state_space, action_space, model, tabular_policy_act, tabular_policy_free_model);
    pol->action_probability = tabular_policy_action_probability;
    return pol;
}

Policy* tabular_policy_update(Policy* policy, Tensor* probabilities) {
    if (policy->model == NULL) {
        fprintf(stderr, "Error: Policy does not have a model to update.\n");
        return policy;
    }
    TabularPolicyModel* model = (TabularPolicyModel*)policy->model;
    tensor_assert_shape(probabilities, model->probabilities->n1, model->probabilities->n2, 0, 0);
    tensor_copy_data(model->probabilities, probabilities);
    return policy;
}

Policy* tabular_policy_update_from_preferences(Policy* policy, Tensor* preferences, float temperature) {
    if (policy->model == NULL) {
        fprintf(stderr, "Error: Policy does not have a model to update.\n");
        return policy;
    }
    TabularPolicyModel* model = (TabularPolicyModel*)policy->model;
    tensor_assert_shape(preferences, model->probabilities->n1, model->probabilities->n2, 0, 0);
    mat_row_softmax(model->probabilities->data, preferences->data, model->probabilities->n1, model->probabilities->n2, temperature);
    return policy;
}

Tensor* tabular_policy_probabilities(Policy* policy) {
    if (policy->model == NULL) {
        fprintf(stderr, "Error: Policy does not have a model to extract probabilities from.\n");
        return NULL;
    }
    TabularPolicyModel* model = (TabularPolicyModel*)policy->model;
    return model->probabilities;
}


// ===========================================================================
// Sampling
// ===========================================================================

int sample_action(const float* probs, int num_actions) {
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0;
    for (int a = 0; a < num_actions; a++) {
        cumsum += probs[a];
        if (r <= cumsum) return a;
    }
    return num_actions - 1;
}

float policy_entropy(const Tensor* probs_matrix) {
    float* entropies = (float*)malloc(sizeof(float) * probs_matrix->n1);
    mat_row_entropy(entropies, probs_matrix->data, probs_matrix->n1, probs_matrix->n2);
    float mean_entropy = vec_mean(entropies, probs_matrix->n1);
    free(entropies);
    return mean_entropy;
}

// ===========================================================================
// Play / rollout
// ===========================================================================

PlayStats play_stats_init(float gamma, float cost_gamma, float secondary_reward_gamma, int stats_window) {
    PlayStats s = {0};
    s.gamma = gamma;
    s.cost_gamma = cost_gamma;
    s.secondary_reward_gamma = secondary_reward_gamma;
    s.stats_window = stats_window;
    if (stats_window > 0) {
        s.buf_return              = (float*)calloc(stats_window, sizeof(float));
        s.buf_cost                = (float*)calloc(stats_window, sizeof(float));
        s.buf_secondary_reward    = (float*)calloc(stats_window, sizeof(float));
        s.buf_disc_return         = (float*)calloc(stats_window, sizeof(float));
        s.buf_disc_cost           = (float*)calloc(stats_window, sizeof(float));
        s.buf_disc_secondary_return = (float*)calloc(stats_window, sizeof(float));
        s.buf_episode_length      = (float*)calloc(stats_window, sizeof(float));
    }
    play_stats_reset(&s);
    return s;
}

void play_stats_push(PlayStats* stats, float reward, float cost, float secondary_reward, bool done) {
    stats->cur_episode_reward += reward;
    stats->cur_episode_cost += cost;
    stats->cur_episode_secondary_reward += secondary_reward;
    stats->cur_episode_disc_return += stats->gamma_t * reward;
    stats->cur_episode_disc_cost += stats->cost_gamma_t * cost;
    stats->cur_episode_disc_secondary_return += stats->secondary_reward_gamma_t * secondary_reward;
    stats->gamma_t *= stats->gamma;
    stats->cost_gamma_t *= stats->cost_gamma;
    stats->secondary_reward_gamma_t *= stats->secondary_reward_gamma;
    stats->cur_episode_length++;
    if (done) {
        const float ep_r   = stats->cur_episode_reward;
        const float ep_c   = stats->cur_episode_cost;
        const float ep_sr  = stats->cur_episode_secondary_reward;
        const float ep_dr  = stats->cur_episode_disc_return;
        const float ep_dc  = stats->cur_episode_disc_cost;
        const float ep_dsr = stats->cur_episode_disc_secondary_return;
        const float ep_len = (float)stats->cur_episode_length;

        stats->total_reward += ep_r;

        if (stats->num_episodes == 0) {
            stats->min_return = ep_r;
            stats->max_return = ep_r;
        } else {
            if (ep_r < stats->min_return) stats->min_return = ep_r;
            if (ep_r > stats->max_return) stats->max_return = ep_r;
        }

        stats->num_episodes += 1;

        if (stats->stats_window > 0) {
            // sliding window: write to circular buffer, recompute means
            int idx = stats->buf_idx;
            stats->buf_return[idx]              = ep_r;
            stats->buf_cost[idx]                = ep_c;
            stats->buf_secondary_reward[idx]    = ep_sr;
            stats->buf_disc_return[idx]         = ep_dr;
            stats->buf_disc_cost[idx]           = ep_dc;
            stats->buf_disc_secondary_return[idx] = ep_dsr;
            stats->buf_episode_length[idx]      = ep_len;
            stats->buf_idx = (idx + 1) % stats->stats_window;
            if (stats->buf_count < stats->stats_window) stats->buf_count++;

            float sum_r = 0, sum_c = 0, sum_sr = 0, sum_dr = 0, sum_dc = 0, sum_dsr = 0, sum_len = 0;
            for (int i = 0; i < stats->buf_count; i++) {
                sum_r   += stats->buf_return[i];
                sum_c   += stats->buf_cost[i];
                sum_sr  += stats->buf_secondary_reward[i];
                sum_dr  += stats->buf_disc_return[i];
                sum_dc  += stats->buf_disc_cost[i];
                sum_dsr += stats->buf_disc_secondary_return[i];
                sum_len += stats->buf_episode_length[i];
            }
            float inv_n = 1.0f / (float)stats->buf_count;
            stats->mean_return                      = sum_r   * inv_n;
            stats->mean_cost                        = sum_c   * inv_n;
            stats->mean_secondary_reward            = sum_sr  * inv_n;
            stats->mean_discounted_return           = sum_dr  * inv_n;
            stats->mean_discounted_cost             = sum_dc  * inv_n;
            stats->mean_discounted_secondary_return = sum_dsr * inv_n;
            stats->mean_episode_length              = sum_len * inv_n;
            // sliding window variance for discounted quantities
            if (stats->buf_count > 1) {
                float ss_dr = 0, ss_dc = 0, ss_dsr = 0;
                for (int i = 0; i < stats->buf_count; i++) {
                    float d_dr  = stats->buf_disc_return[i]           - stats->mean_discounted_return;
                    float d_dc  = stats->buf_disc_cost[i]             - stats->mean_discounted_cost;
                    float d_dsr = stats->buf_disc_secondary_return[i] - stats->mean_discounted_secondary_return;
                    ss_dr  += d_dr  * d_dr;
                    ss_dc  += d_dc  * d_dc;
                    ss_dsr += d_dsr * d_dsr;
                }
                float inv_nm1 = 1.0f / (float)(stats->buf_count - 1);
                stats->var_discounted_return            = ss_dr  * inv_nm1;
                stats->var_discounted_cost              = ss_dc  * inv_nm1;
                stats->var_discounted_secondary_return  = ss_dsr * inv_nm1;
            } else {
                stats->var_discounted_return            = 0.0f;
                stats->var_discounted_cost              = 0.0f;
                stats->var_discounted_secondary_return  = 0.0f;
            }
        } else {
            // infinite window: Welford's online algorithm
            const float inv_n = 1.0f / (float)stats->num_episodes;
            const float old_mean_dr  = stats->mean_discounted_return;
            const float old_mean_dc  = stats->mean_discounted_cost;
            const float old_mean_dsr = stats->mean_discounted_secondary_return;
            stats->mean_return                      += (ep_r  - stats->mean_return) * inv_n;
            stats->mean_cost                        += (ep_c  - stats->mean_cost) * inv_n;
            stats->mean_secondary_reward            += (ep_sr - stats->mean_secondary_reward) * inv_n;
            stats->mean_discounted_return           += (ep_dr - stats->mean_discounted_return) * inv_n;
            stats->mean_discounted_cost             += (ep_dc - stats->mean_discounted_cost) * inv_n;
            stats->mean_discounted_secondary_return += (ep_dsr- stats->mean_discounted_secondary_return) * inv_n;
            stats->mean_episode_length              += (ep_len- stats->mean_episode_length) * inv_n;
            // Welford M2 update: M2 += (x - old_mean) * (x - new_mean)
            stats->m2_discounted_return            += (ep_dr  - old_mean_dr)  * (ep_dr  - stats->mean_discounted_return);
            stats->m2_discounted_cost              += (ep_dc  - old_mean_dc)  * (ep_dc  - stats->mean_discounted_cost);
            stats->m2_discounted_secondary_return  += (ep_dsr - old_mean_dsr) * (ep_dsr - stats->mean_discounted_secondary_return);
            if (stats->num_episodes > 1) {
                float inv_nm1 = 1.0f / (float)(stats->num_episodes - 1);
                stats->var_discounted_return            = stats->m2_discounted_return           * inv_nm1;
                stats->var_discounted_cost              = stats->m2_discounted_cost             * inv_nm1;
                stats->var_discounted_secondary_return  = stats->m2_discounted_secondary_return * inv_nm1;
            }
        }

        // Reset per-episode accumulators
        stats->cur_episode_length = 0;
        stats->gamma_t = 1.0f;
        stats->cost_gamma_t = 1.0f;
        stats->secondary_reward_gamma_t = 1.0f;
        stats->cur_episode_reward = 0.0f;
        stats->cur_episode_cost = 0.0f;
        stats->cur_episode_secondary_reward = 0.0f;
        stats->cur_episode_disc_return = 0.0f;
        stats->cur_episode_disc_cost = 0.0f;
        stats->cur_episode_disc_secondary_return = 0.0f;
    }
}

void play_stats_reset(PlayStats* stats) {
    stats->num_episodes = 0;
    stats->total_reward = 0.0f;
    stats->min_return = -1e30f;
    stats->max_return = 1e30f;
    stats->mean_episode_length = 0.0f;
    stats->mean_return = 0.0f;
    stats->mean_cost = 0.0f;
    stats->mean_secondary_reward = 0.0f;
    stats->mean_discounted_return = 0.0f;
    stats->mean_discounted_cost = 0.0f;
    stats->mean_discounted_secondary_return = 0.0f;
    stats->var_discounted_return = 0.0f;
    stats->var_discounted_cost = 0.0f;
    stats->var_discounted_secondary_return = 0.0f;
    stats->m2_discounted_return = 0.0f;
    stats->m2_discounted_cost = 0.0f;
    stats->m2_discounted_secondary_return = 0.0f;
    stats->cur_episode_length = 0;
    stats->gamma_t = 1.0f;
    stats->cost_gamma_t = 1.0f;
    stats->secondary_reward_gamma_t = 1.0f;
    stats->cur_episode_reward = 0.0f;
    stats->cur_episode_cost = 0.0f;
    stats->cur_episode_secondary_reward = 0.0f;
    stats->cur_episode_disc_return = 0.0f;
    stats->cur_episode_disc_cost = 0.0f;
    stats->cur_episode_disc_secondary_return = 0.0f;
    stats->buf_count = 0;
    stats->buf_idx = 0;
    // buffers (and their capacity stats_window) are preserved, just zeroed
    if (stats->stats_window > 0) {
        memset(stats->buf_return,              0, stats->stats_window * sizeof(float));
        memset(stats->buf_cost,                0, stats->stats_window * sizeof(float));
        memset(stats->buf_secondary_reward,    0, stats->stats_window * sizeof(float));
        memset(stats->buf_disc_return,         0, stats->stats_window * sizeof(float));
        memset(stats->buf_disc_cost,           0, stats->stats_window * sizeof(float));
        memset(stats->buf_disc_secondary_return, 0, stats->stats_window * sizeof(float));
        memset(stats->buf_episode_length,      0, stats->stats_window * sizeof(float));
    }
}

void play_stats_free_buffers(PlayStats* stats) {
    free(stats->buf_return);
    free(stats->buf_cost);
    free(stats->buf_secondary_reward);
    free(stats->buf_disc_return);
    free(stats->buf_disc_cost);
    free(stats->buf_disc_secondary_return);
    free(stats->buf_episode_length);
    stats->buf_return = NULL;
    stats->buf_cost = NULL;
    stats->buf_secondary_reward = NULL;
    stats->buf_disc_return = NULL;
    stats->buf_disc_cost = NULL;
    stats->buf_disc_secondary_return = NULL;
    stats->buf_episode_length = NULL;
    stats->buf_count = 0;
    stats->buf_idx = 0;
}

void play_stats_merge(PlayStats* dest, const PlayStats* src) {
    // Merge completed-episode aggregates from src into dest.
    // dest must use stats_window == 0 (infinite window); src may have a sliding window.
    const int n_d = dest->num_episodes;
    const int n_s = src->num_episodes;
    if (n_s == 0) return;

    dest->total_reward += src->total_reward;

    if (n_d == 0) {
        dest->min_return = src->min_return;
        dest->max_return = src->max_return;
    } else if (n_s > 0) {
        if (src->min_return < dest->min_return) dest->min_return = src->min_return;
        if (src->max_return > dest->max_return) dest->max_return = src->max_return;
    }

    // src->mean_* covers buf_count episodes when windowed, all num_episodes otherwise
    const int eff_s = (src->stats_window > 0) ? src->buf_count : n_s;
    const int eff   = n_d + eff_s;
    if (eff > 0) {
        const float fd = (float)n_d   / (float)eff;
        const float fs = (float)eff_s / (float)eff;

        const float old_mean_dr  = dest->mean_discounted_return;
        const float old_mean_dc  = dest->mean_discounted_cost;
        const float old_mean_dsr = dest->mean_discounted_secondary_return;

        dest->mean_return                      = fd * dest->mean_return                      + fs * src->mean_return;
        dest->mean_cost                        = fd * dest->mean_cost                        + fs * src->mean_cost;
        dest->mean_secondary_reward            = fd * dest->mean_secondary_reward            + fs * src->mean_secondary_reward;
        dest->mean_discounted_return           = fd * dest->mean_discounted_return           + fs * src->mean_discounted_return;
        dest->mean_discounted_cost             = fd * dest->mean_discounted_cost             + fs * src->mean_discounted_cost;
        dest->mean_discounted_secondary_return = fd * dest->mean_discounted_secondary_return + fs * src->mean_discounted_secondary_return;
        dest->mean_episode_length              = fd * dest->mean_episode_length              + fs * src->mean_episode_length;

        // Parallel variance merge: Var_combined = (n_d*Var_d + eff_s*Var_s + n_d*delta_d^2 + eff_s*delta_s^2) / (eff - 1)
        // where delta_d = old_mean_d - new_combined_mean, delta_s = src_mean - new_combined_mean
        // Equivalently use M2: M2_combined = M2_d + M2_s + delta^2 * n_d * eff_s / eff
        const float src_var_dr  = (src->stats_window > 0) ? src->var_discounted_return           : src->var_discounted_return;
        const float src_var_dc  = (src->stats_window > 0) ? src->var_discounted_cost             : src->var_discounted_cost;
        const float src_var_dsr = (src->stats_window > 0) ? src->var_discounted_secondary_return : src->var_discounted_secondary_return;
        const float delta_dr  = old_mean_dr  - src->mean_discounted_return;
        const float delta_dc  = old_mean_dc  - src->mean_discounted_cost;
        const float delta_dsr = old_mean_dsr - src->mean_discounted_secondary_return;
        const float cross = (float)n_d * (float)eff_s / (float)eff;
        // M2_d = var_d * (n_d - 1), M2_s = var_s * (eff_s - 1)
        const float m2_d_dr  = (n_d   > 1) ? dest->var_discounted_return           * (float)(n_d - 1)   : 0.0f;
        const float m2_d_dc  = (n_d   > 1) ? dest->var_discounted_cost             * (float)(n_d - 1)   : 0.0f;
        const float m2_d_dsr = (n_d   > 1) ? dest->var_discounted_secondary_return * (float)(n_d - 1)   : 0.0f;
        const float m2_s_dr  = (eff_s > 1) ? src_var_dr  * (float)(eff_s - 1) : 0.0f;
        const float m2_s_dc  = (eff_s > 1) ? src_var_dc  * (float)(eff_s - 1) : 0.0f;
        const float m2_s_dsr = (eff_s > 1) ? src_var_dsr * (float)(eff_s - 1) : 0.0f;
        const float m2_dr  = m2_d_dr  + m2_s_dr  + delta_dr  * delta_dr  * cross;
        const float m2_dc  = m2_d_dc  + m2_s_dc  + delta_dc  * delta_dc  * cross;
        const float m2_dsr = m2_d_dsr + m2_s_dsr + delta_dsr * delta_dsr * cross;
        if (eff > 1) {
            const float inv_effm1 = 1.0f / (float)(eff - 1);
            dest->var_discounted_return            = m2_dr  * inv_effm1;
            dest->var_discounted_cost              = m2_dc  * inv_effm1;
            dest->var_discounted_secondary_return  = m2_dsr * inv_effm1;
        }
        // Update M2 accumulators for dest (infinite window)
        dest->m2_discounted_return            = m2_dr;
        dest->m2_discounted_cost              = m2_dc;
        dest->m2_discounted_secondary_return  = m2_dsr;
    }

    dest->num_episodes = n_d + eff_s;
}

PlayStats play(MDP* env, Policy* policy, int num_episodes, float gamma, float cost_gamma, float secondary_reward_gamma) {
    PlayStats s = play_stats_init(gamma, cost_gamma, secondary_reward_gamma, 0);

    const bool has_cost = (env->cost_function != NULL);
    const bool has_secondary_reward = (env->secondary_reward_function != NULL);

    for (int ep = 0; ep < num_episodes; ep++) {
        env->reset(env);
        while (!env->terminated && !env->truncated) {
            policy->act(policy, env->observation);
            env->step(env, policy->action);
            const float r  = env->reward;
            const float c  = has_cost ? env->cost : 0.0f;
            const float sr = has_secondary_reward ? env->secondary_reward : 0.0f;
            const bool done = (env->terminated || env->truncated);
            play_stats_push(&s, r, c, sr, done);
        }
    }
    // Ensure reported count matches requested episodes (should already match)
    // and return streaming statistics accumulated by play_stats_push.
    return s;
}

// ===========================================================================
// Vectorized Environment  (thread-per-env)
// ===========================================================================

static void* vecenv_worker_fn(void* arg) {
    VecEnvWorker* w = (VecEnvWorker*)arg;
    VecEnv* venv = w->venv;
    int tid = w->thread_idx;

    while (1) {
        pthread_mutex_lock(&venv->mutex);
        while (venv->worker_gen[tid] == venv->generation && venv->running)
            pthread_cond_wait(&venv->work_cond, &venv->mutex);
        if (!venv->running) {
            pthread_mutex_unlock(&venv->mutex);
            break;
        }
        venv->worker_gen[tid] = venv->generation;
        pthread_mutex_unlock(&venv->mutex);

        // Step all envs assigned to this thread
        for (int i = 0; i < w->num_envs; i++) {
            int e = w->env_indices[i];
            MDP* env = venv->envs[e];

            env->step(env, &venv->step_actions[e]);

            venv->step_rewards[e]    = env->reward;
            venv->step_costs[e]      = env->cost;
            venv->step_secondary_rewards[e] = env->secondary_reward;
            venv->step_terminated[e] = env->terminated;
            venv->step_truncated[e]  = env->truncated;

            if (env->terminated || env->truncated)
                env->reset(env);
        }

        pthread_mutex_lock(&venv->mutex);
        venv->done_count++;
        if (venv->done_count == venv->num_threads)
            pthread_cond_signal(&venv->done_cond);
        pthread_mutex_unlock(&venv->mutex);
    }
    return NULL;
}

// Detect number of logical CPU cores (platform-specific)
static int detect_cpu_cores(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#elif defined(__APPLE__)
    int count = 0;
    size_t sz = sizeof(count);
    if (sysctlbyname("hw.logicalcpu", &count, &sz, NULL, 0) == 0 && count > 0)
        return count;
    return 4;  // fallback
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 4;
#endif
}

VecEnv* vecenv_malloc(MDP* (*constructor)(void), int num_envs, bool async) {
    VecEnv* venv = (VecEnv*)malloc(sizeof(VecEnv));
    venv->num_envs = num_envs;
    venv->async    = async;
    venv->envs = (MDP**)malloc(sizeof(MDP*) * num_envs);
    for (int i = 0; i < num_envs; i++)
        venv->envs[i] = constructor();

    venv->step_rewards    = (float*)malloc(sizeof(float) * num_envs);
    venv->step_costs       = (float*)malloc(sizeof(float) * num_envs);
    venv->step_secondary_rewards = (float*)malloc(sizeof(float) * num_envs);
    venv->step_terminated = (bool*)  malloc(sizeof(bool)   * num_envs);
    venv->step_truncated  = (bool*)  malloc(sizeof(bool)   * num_envs);
    venv->step_actions    = (void**)malloc(sizeof(void*) * num_envs);

    venv->threads     = NULL;
    venv->workers     = NULL;
    venv->worker_gen  = NULL;
    venv->num_threads = 0;
    if (async) {
        // Determine thread count: min(cpu_cores, num_envs)
        int ncpu = detect_cpu_cores();
        int nthreads = ncpu < num_envs ? ncpu : num_envs;
        venv->num_threads = nthreads;

        venv->generation = 0;
        venv->done_count = 0;
        venv->running    = true;
        venv->worker_gen = (int*)calloc(nthreads, sizeof(int));

        pthread_mutex_init(&venv->mutex, NULL);
        pthread_cond_init(&venv->work_cond, NULL);
        pthread_cond_init(&venv->done_cond, NULL);

        // Round-robin assignment: thread t gets envs t, t+nthreads, t+2*nthreads ...
        venv->workers = (VecEnvWorker*)malloc(sizeof(VecEnvWorker) * nthreads);
        venv->threads = (pthread_t*)   malloc(sizeof(pthread_t)    * nthreads);
        for (int t = 0; t < nthreads; t++) {
            // Count envs for this thread
            int count = 0;
            for (int e = t; e < num_envs; e += nthreads)
                count++;
            venv->workers[t].thread_idx  = t;
            venv->workers[t].num_envs    = count;
            venv->workers[t].env_indices = (int*)malloc(sizeof(int) * count);
            venv->workers[t].venv        = venv;
            int idx = 0;
            for (int e = t; e < num_envs; e += nthreads)
                venv->workers[t].env_indices[idx++] = e;

            pthread_create(&venv->threads[t], NULL,
                           vecenv_worker_fn, &venv->workers[t]);
        }
        printf("  VecEnv: %d envs, %d threads (detected %d CPU cores)\n",
               num_envs, nthreads, ncpu);
    }
    return venv;
}

void vecenv_step(VecEnv* venv, void** actions) {
    if (venv->async) {
        pthread_mutex_lock(&venv->mutex);
        memcpy(venv->step_actions, actions, sizeof(void*) * venv->num_envs);
        venv->done_count = 0;
        venv->generation++;
        pthread_cond_broadcast(&venv->work_cond);
        while (venv->done_count < venv->num_threads)
            pthread_cond_wait(&venv->done_cond, &venv->mutex);
        pthread_mutex_unlock(&venv->mutex);
    } else {
        for (int e = 0; e < venv->num_envs; e++) {
            MDP* env = venv->envs[e];
            void* action = actions[e];
            env->step(env, action);
            venv->step_rewards[e]    = env->reward;
            venv->step_costs[e]      = env->cost;
            venv->step_secondary_rewards[e] = env->secondary_reward;
            venv->step_terminated[e] = env->terminated;
            venv->step_truncated[e]  = env->truncated;
            if (env->terminated || env->truncated)
                env->reset(env);
        }
    }
}

void vecenv_reset(VecEnv* venv) {
    for (int i = 0; i < venv->num_envs; i++)
        venv->envs[i]->reset(venv->envs[i]);
}

void vecenv_free(VecEnv* venv) {
    if (!venv) return;

    if (venv->async) {
        pthread_mutex_lock(&venv->mutex);
        venv->running = false;
        pthread_cond_broadcast(&venv->work_cond);
        pthread_mutex_unlock(&venv->mutex);

        for (int i = 0; i < venv->num_threads; i++) {
            pthread_join(venv->threads[i], NULL);
            free(venv->workers[i].env_indices);
        }

        pthread_mutex_destroy(&venv->mutex);
        pthread_cond_destroy(&venv->work_cond);
        pthread_cond_destroy(&venv->done_cond);
    }

    for (int i = 0; i < venv->num_envs; i++)
        mdp_free(venv->envs[i]);
    free(venv->envs);
    free(venv->step_rewards);
    free(venv->step_costs);
    free(venv->step_secondary_rewards);
    free(venv->step_terminated);
    free(venv->step_truncated);
    free(venv->step_actions);
    free(venv->worker_gen);
    free(venv->workers);
    free(venv->threads);
    free(venv);
}
