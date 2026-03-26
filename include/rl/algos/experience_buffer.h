#pragma once
#include "rl/core/tensor.h"
#include "rl/core/mdp.h"


typedef struct ExperienceBuffer {
    Tensor* states;
    Tensor* actions;        // should be cast to integers when using
    Tensor* rewards;        // primary/env reward
    Tensor* secondary_rewards; // secondary reward component
    Tensor* costs;              // cost component
    Tensor* states_next;
    Tensor* terminateds;
    Tensor* is_start;       // 1.0 if this state was an episode start state, 0.0 otherwise
    int cur_index;
    int count;
    int start_count;        // number of start states stored
} ExperienceBuffer;


ExperienceBuffer* experience_buffer_malloc(Space 𝒮, Space 𝒜, int capacity);

void experience_buffer_push(ExperienceBuffer* buffer, Tensor* state, int action,
                            float reward, float secondary_reward, float cost,
                            Tensor* state_next, bool terminated);

void experience_buffer_push_start(ExperienceBuffer* buffer);

void experience_buffer_clear(ExperienceBuffer* buffer);

void experience_buffer_sample(ExperienceBuffer* buffer, ExperienceBuffer* buffer_out);

// Sample start states uniformly from the buffer into states_out (2D: n_samples x state_dim)
void experience_buffer_sample_start_states(ExperienceBuffer* buffer, Tensor* states_out);

void experience_buffer_free(ExperienceBuffer* buffer);
