#include "rl/algos/dqn.h"
#include <string.h>

// ===========================================================================
// DQN Policy  (MLP → argmax, wraps SequentialNN)
// ===========================================================================

static void dqn_policy_act(Policy* policy, void* observation) {
    DQNModel* m = (DQNModel*)policy->model;
    int n_actions = policy->action_space.n;
    float* q = (float*)malloc(sizeof(float) * n_actions);
    seq_nn_forward(m->nn, (float*)observation, q, 1);
    int best;
    mat_row_argmax(&best, q, 1, n_actions);
    *(int*)policy->action = best;
    free(q);
}

static void dqn_free_model(Policy* policy) {
    DQNModel* m = (DQNModel*)policy->model;
    if (m) {
        seq_nn_free(m->nn);
        m->nn = NULL;
    }
    // policy_free will free(policy->model) itself
}

Policy* dqn_policy_malloc(Space observation_space, Space action_space,
                          int num_hidden, int* hidden_sizes) {
    int num_sizes = num_hidden + 2;
    int* sizes = (int*)malloc(sizeof(int) * num_sizes);
    sizes[0] = observation_space.n;
    for (int i = 0; i < num_hidden; i++)
        sizes[i + 1] = hidden_sizes[i];
    sizes[num_sizes - 1] = action_space.n;

    DQNModel* m = (DQNModel*)malloc(sizeof(DQNModel));
    m->nn = seq_nn_make_mlp(num_sizes, sizes, false);
    m->obs_dim = observation_space.n;
    free(sizes);

    return policy_malloc("dqn", observation_space, action_space,
                         m, dqn_policy_act, dqn_free_model);
}

// ===========================================================================
// Replay buffer
// ===========================================================================

ReplayBuffer* replay_buffer_malloc(int capacity, int obs_dim) {
    ReplayBuffer* rb = (ReplayBuffer*)malloc(sizeof(ReplayBuffer));
    rb->capacity = capacity;
    rb->size = 0;
    rb->head = 0;
    rb->obs_dim = obs_dim;
    rb->states      = (float*)malloc(sizeof(float) * capacity * obs_dim);
    rb->actions     = (int*)malloc(sizeof(int) * capacity);
    rb->rewards     = (float*)malloc(sizeof(float) * capacity);
    rb->next_states = (float*)malloc(sizeof(float) * capacity * obs_dim);
    rb->dones       = (bool*)malloc(sizeof(bool) * capacity);
    return rb;
}

void replay_buffer_free(ReplayBuffer* rb) {
    if (rb) {
        free(rb->states);
        free(rb->actions);
        free(rb->rewards);
        free(rb->next_states);
        free(rb->dones);
        free(rb);
    }
}

void replay_buffer_push(ReplayBuffer* rb, float* state, int action,
                         float reward, float* next_state, bool done) {
    int idx = rb->head;
    memcpy(&rb->states[idx * rb->obs_dim], state, sizeof(float) * rb->obs_dim);
    rb->actions[idx] = action;
    rb->rewards[idx] = reward;
    memcpy(&rb->next_states[idx * rb->obs_dim], next_state, sizeof(float) * rb->obs_dim);
    rb->dones[idx] = done;
    rb->head = (rb->head + 1) % rb->capacity;
    if (rb->size < rb->capacity) rb->size++;
}

// ===========================================================================
// DQN Training  (batched forward / backward, Double DQN)
// ===========================================================================

void dqn_train(MDP* env, Policy* policy, int num_samples) {
    // ---- Hyperparameters ----
    float gamma          = 0.99f;
    float lr             = 1e-3f;
    float eps_start      = 1.0f;
    float eps_end        = 0.01f;
    float eps_decay_frac = 0.5f;
    int    batch_size     = 64;
    int    replay_capacity = 50000;
    int    learn_start    = 1000;
    int    target_update  = 500;
    int    log_interval   = 1000;
    float beta1 = 0.9f, beta2 = 0.999f, adam_eps = 1e-8f;

    DQNModel* m = (DQNModel*)policy->model;
    SequentialNN* nn = m->nn;
    int obs_dim   = m->obs_dim;
    int n_actions = policy->action_space.n;

    // ---- Replay buffer ----
    ReplayBuffer* rb = replay_buffer_malloc(replay_capacity, obs_dim);

    // ---- Target network (clone of policy nn, no backprop) ----
    SequentialNN* target_nn = seq_nn_clone(nn, false);

    // ---- Optimizer ----
    AdamState* adam = adam_malloc_seq(nn, lr, beta1, beta2, adam_eps);

    // ---- Pre-allocate batched buffers ----
    float* batch_states      = (float*)malloc(sizeof(float) * batch_size * obs_dim);
    int*    batch_actions      = (int*)malloc(sizeof(int)    * batch_size);
    float* batch_rewards      = (float*)malloc(sizeof(float) * batch_size);
    float* batch_next_states  = (float*)malloc(sizeof(float) * batch_size * obs_dim);
    bool*   batch_dones        = (bool*)malloc(sizeof(bool)   * batch_size);

    float* batch_q              = (float*)malloc(sizeof(float) * batch_size * n_actions);
    float* batch_q_next          = (float*)malloc(sizeof(float) * batch_size * n_actions);
    float* batch_q_next_online   = (float*)malloc(sizeof(float) * batch_size * n_actions);
    float* batch_delta            = (float*)malloc(sizeof(float) * batch_size * n_actions);

    // Single-sample Q buffer for epsilon-greedy
    float* q_vals   = (float*)malloc(sizeof(float) * n_actions);
    float* state_buf = (float*)malloc(sizeof(float) * obs_dim);

    // ---- Episode tracking ----
    float episode_reward    = 0.0f;
    int    episodes_done     = 0;
    float reward_sum_recent = 0.0f;
    int    episodes_recent   = 0;

    env->reset(env);
    memcpy(state_buf, (float*)env->observation, sizeof(float) * obs_dim);

    // ---- Main loop ----
    for (int step = 0; step < num_samples; step++) {

        // Epsilon-greedy action
        float epsilon = eps_end + (eps_start - eps_end) *
            fmaxf(0.0f, 1.0f - (float)step / (eps_decay_frac * num_samples));

        int action;
        if ((double)rand() / RAND_MAX < epsilon) {
            action = rand() % n_actions;
        } else {
            seq_nn_forward(nn, state_buf, q_vals, 1);
            mat_row_argmax(&action, q_vals, 1, n_actions);
        }

        // Environment step
        env->step(env, &action);

        replay_buffer_push(rb, state_buf, action, env->reward,
                           (float*)env->observation, env->terminated);
        episode_reward += env->reward;

        if (env->terminated || env->truncated) {
            episodes_done++;
            reward_sum_recent += episode_reward;
            episodes_recent++;
            episode_reward = 0.0f;
            env->reset(env);
        }
        memcpy(state_buf, (float*)env->observation, sizeof(float) * obs_dim);

        // ---- Learning step (batched) ----
        if (rb->size >= learn_start) {

            // 1. Sample batch from replay buffer
            for (int b = 0; b < batch_size; b++) {
                int idx = rand() % rb->size;
                memcpy(&batch_states[b * obs_dim],
                       &rb->states[idx * obs_dim], sizeof(float) * obs_dim);
                batch_actions[b] = rb->actions[idx];
                batch_rewards[b] = rb->rewards[idx];
                memcpy(&batch_next_states[b * obs_dim],
                       &rb->next_states[idx * obs_dim], sizeof(float) * obs_dim);
                batch_dones[b] = rb->dones[idx];
            }

            // 2. Batched forward: policy Q-values (with backprop)
            seq_nn_set_backprop(nn, true);
            seq_nn_forward(nn, batch_states, batch_q, batch_size);

            // 3. Batched forward: target Q-values (no backprop)
            seq_nn_forward(target_nn, batch_next_states, batch_q_next,
                           batch_size);

            // 4. Double DQN: select action with online net, evaluate with target
            seq_nn_set_backprop(nn, false);
            seq_nn_forward(nn, batch_next_states, batch_q_next_online,
                           batch_size);

            // 5. Build delta matrix [batch x n_actions]
            memset(batch_delta, 0, sizeof(float) * batch_size * n_actions);
            for (int b = 0; b < batch_size; b++) {
                int   a = batch_actions[b];
                float target_val;
                if (batch_dones[b]) {
                    target_val = batch_rewards[b];
                } else {
                    // Double DQN: a* = argmax_a' Q_online(s', a')
                    int best_a = 0;
                    float best_q = batch_q_next_online[b * n_actions];
                    for (int a2 = 1; a2 < n_actions; a2++) {
                        float v = batch_q_next_online[b * n_actions + a2];
                        if (v > best_q) { best_q = v; best_a = a2; }
                    }
                    target_val = batch_rewards[b] + gamma * batch_q_next[b * n_actions + best_a];
                }
                float td = batch_q[b * n_actions + a] - target_val;
                float dL = 2.0f * td / batch_size;
                // Clip gradient for stability
                if (dL >  1.0f) dL =  1.0f;
                if (dL < -1.0f) dL = -1.0f;
                batch_delta[b * n_actions + a] = dL;
            }

            // 6. Batched backward + Adam update
            seq_nn_zero_grad(nn);
            seq_nn_backward(nn, batch_delta, NULL, batch_size);
            adam_step_seq(adam, nn);
        }

        // Target network sync
        if ((step + 1) % target_update == 0)
            seq_nn_copy_params(target_nn, nn);

        // Logging
        if ((step + 1) % log_interval == 0) {
            double avg = episodes_recent > 0
                ? (double)reward_sum_recent / episodes_recent : 0.0;
            printf("  step %6d/%d | eps %.3f | episodes %d | avg reward %.1f\n",
                   step + 1, num_samples, epsilon, episodes_done, avg);
            reward_sum_recent = 0.0f;
            episodes_recent = 0;
        }
    }

    // ---- Cleanup ----
    free(q_vals);
    free(state_buf);
    free(batch_states);
    free(batch_actions);
    free(batch_rewards);
    free(batch_next_states);
    free(batch_dones);
    free(batch_q);
    free(batch_q_next);
    free(batch_q_next_online);
    free(batch_delta);
    adam_free(adam);
    seq_nn_free(target_nn);
    replay_buffer_free(rb);
}
