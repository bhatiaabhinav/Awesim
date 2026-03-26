#pragma once

#include "rl/core/tensor.h"
#include "rl/core/mdp.h"
#include "rl/core/rl_utils.h"

typedef struct ActorModel {
    SequentialNN* nn;
    float temperature;
    float* logits_buffer;
    float* probs_buffer;
    int max_batch_size;
    // observation normalisation (optional; used during inference)
    bool           norm_obs;
    bool           deterministic;
    ObsNormalizer  obs_norm;
} ActorModel;

void actor_act(Policy* policy, void* observation);
void actor_model_free(Policy* policy);
Policy* actor_mlp_policy_malloc(int input_dim, int output_dim, const int* hidden_sizes, int num_hidden, float temperature, int max_batch_size);