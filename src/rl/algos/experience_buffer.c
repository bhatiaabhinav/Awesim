#include "rl/algos/experience_buffer.h"


ExperienceBuffer* experience_buffer_malloc(Space 𝒮, Space 𝒜, int capacity) {
    ExperienceBuffer* buffer = (ExperienceBuffer*) malloc(sizeof(ExperienceBuffer));
    buffer->states = tensor2d_malloc(capacity, 𝒮.n);
    buffer->actions = tensor1d_malloc(capacity);
    buffer->rewards = tensor1d_malloc(capacity);
    buffer->secondary_rewards = tensor1d_malloc(capacity);
    buffer->costs = tensor1d_malloc(capacity);
    buffer->states_next = tensor2d_malloc(capacity, 𝒮.n);
    buffer->terminateds = tensor1d_malloc(capacity);
    buffer->is_start = tensor1d_malloc(capacity);
    buffer->cur_index = 0;
    buffer->count = 0;
    buffer->start_count = 0;
    return buffer;
}

void experience_buffer_push(ExperienceBuffer* buffer, Tensor* state, int action,
                            float reward, float secondary_reward, float cost,
                            Tensor* state_next, bool terminated) {
    int capacity  = buffer->states->n1;
    int state_dim = buffer->states->n2;
    int idx = buffer->cur_index;
    // if overwriting a start state, decrement start_count
    if (buffer->count == capacity && buffer->is_start->data[idx] == 1.0f)
        buffer->start_count--;
    memcpy(&buffer->states->data[idx * state_dim], state->data, sizeof(float) * state_dim);
    buffer->actions->data[idx]     = (float)action;
    buffer->rewards->data[idx]     = reward;
    buffer->secondary_rewards->data[idx] = secondary_reward;
    buffer->costs->data[idx]       = cost;
    memcpy(&buffer->states_next->data[idx * state_dim], state_next->data, sizeof(float) * state_dim);
    buffer->terminateds->data[idx] = terminated ? 1.0f : 0.0f;
    buffer->is_start->data[idx]    = 0.0f;
    buffer->cur_index = (buffer->cur_index + 1) % capacity;
    if (buffer->count < capacity) buffer->count++;
}

void experience_buffer_push_start(ExperienceBuffer* buffer) {
    // Mark the most recently pushed entry as a start state.
    // Must be called right after experience_buffer_push for the start-of-episode transition.
    int capacity = buffer->states->n1;
    int idx = (buffer->cur_index - 1 + capacity) % capacity;
    buffer->is_start->data[idx] = 1.0f;
    buffer->start_count++;
}

void experience_buffer_clear(ExperienceBuffer* buffer) {
    buffer->cur_index = 0;
    buffer->count     = 0;
    buffer->start_count = 0;
}

void experience_buffer_sample(ExperienceBuffer* buffer, ExperienceBuffer* buffer_out) {
    int n_samples = buffer_out->states->n1;
    int state_dim = buffer->states->n2;
    experience_buffer_clear(buffer_out);
    for (int i = 0; i < n_samples; i++) {
        int idx     = rand() % buffer->count;
        int out_idx = buffer_out->cur_index;
        memcpy(&buffer_out->states->data[out_idx * state_dim],
               &buffer->states->data[idx * state_dim], sizeof(float) * state_dim);
        buffer_out->actions->data[out_idx]     = buffer->actions->data[idx];
        buffer_out->rewards->data[out_idx]     = buffer->rewards->data[idx];
        buffer_out->secondary_rewards->data[out_idx] = buffer->secondary_rewards->data[idx];
        buffer_out->costs->data[out_idx]       = buffer->costs->data[idx];
        memcpy(&buffer_out->states_next->data[out_idx * state_dim],
               &buffer->states_next->data[idx * state_dim], sizeof(float) * state_dim);
        buffer_out->terminateds->data[out_idx] = buffer->terminateds->data[idx];
        buffer_out->cur_index++;
        buffer_out->count++;
    }
}

void experience_buffer_sample_start_states(ExperienceBuffer* buffer, Tensor* states_out) {
    int n_samples = states_out->n1;
    int state_dim = buffer->states->n2;
    // Build index list of start states
    int* start_indices = (int*)malloc(sizeof(int) * buffer->start_count);
    int k = 0;
    for (int i = 0; i < buffer->count; i++)
        if (buffer->is_start->data[i] == 1.0f)
            start_indices[k++] = i;
    // Sample uniformly from start states
    for (int i = 0; i < n_samples; i++) {
        int idx = start_indices[rand() % k];
        memcpy(&states_out->data[i * state_dim],
               &buffer->states->data[idx * state_dim], sizeof(float) * state_dim);
    }
    free(start_indices);
}

void experience_buffer_free(ExperienceBuffer* buffer) {
    if (!buffer) return;
    tensor_free(buffer->states);
    tensor_free(buffer->actions);
    tensor_free(buffer->rewards);
    tensor_free(buffer->secondary_rewards);
    tensor_free(buffer->costs);
    tensor_free(buffer->states_next);
    tensor_free(buffer->terminateds);
    tensor_free(buffer->is_start);
    free(buffer);
}
