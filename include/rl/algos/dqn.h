#ifndef CGYM_DQN_H
#define CGYM_DQN_H

#include "rl/core/mdp.h"

// ---- Replay buffer ----

typedef struct {
    int capacity;
    int size;
    int head;
    int obs_dim;
    float* states;
    int*      actions;
    float* rewards;
    float* next_states;
    bool*     dones;
} ReplayBuffer;

ReplayBuffer* replay_buffer_malloc(int capacity, int obs_dim);
void          replay_buffer_free(ReplayBuffer* rb);
void          replay_buffer_push(ReplayBuffer* rb, float* state, int action,
                                 float reward, float* next_state, bool done);

// ---- DQN policy (MLP → argmax) ----

typedef struct {
    SequentialNN* nn;
    int           obs_dim;
} DQNModel;

Policy* dqn_policy_malloc(Space observation_space, Space action_space,
                          int num_hidden, int* hidden_sizes);

// ---- DQN training ----

void dqn_train(MDP* env, Policy* policy, int num_samples);

#endif // CGYM_DQN_H
