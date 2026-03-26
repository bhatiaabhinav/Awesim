#include "rl/algos/actor.h"
#include "rl/core/mdp.h"
#include <string.h>

void actor_act(Policy* policy, void* observation) {
    ActorModel* model = (ActorModel*) policy->model;
    int n_actions = policy->action_space.n;
    float* obs = (float*) observation;

    // optional observation normalisation (copy to logits_buffer as scratch)
    if (model->norm_obs && model->obs_norm.count > 0) {
        memcpy(model->logits_buffer, obs, sizeof(float) * model->obs_norm.dim);
        obs_normalizer_apply(&model->obs_norm, model->logits_buffer, 1);
        obs = model->logits_buffer;
    }

    seq_nn_forward(model->nn, obs, model->logits_buffer, 1);

    int chosen;
    if (model->deterministic || model->temperature == 0) {
        mat_row_argmax(&chosen, model->logits_buffer, 1, n_actions);
    } else {
        mat_row_softmax(model->probs_buffer, model->logits_buffer, 1, n_actions, model->temperature);
        chosen = sample_action(model->probs_buffer, n_actions);
    }
    *((int*)policy->action) = chosen;
}


Policy* actor_mlp_policy_malloc(int input_dim, int output_dim, const int* hidden_sizes, int num_hidden, float temperature, int max_batch_size) {
    int num_sizes = num_hidden + 2;
    int* sizes = (int*)malloc(num_sizes * sizeof(int));
    sizes[0] = input_dim;
    for (int i = 0; i < num_hidden; i++) {
        sizes[i + 1] = hidden_sizes[i];
    }
    sizes[num_sizes - 1] = output_dim;

    ActorModel* model = (ActorModel*)malloc(sizeof(ActorModel));
    model->nn = seq_nn_make_mlp(num_sizes, sizes, false);
    model->temperature = temperature;
    model->logits_buffer = (float*)malloc(max_batch_size * output_dim * sizeof(float));
    model->probs_buffer  = (float*)malloc(max_batch_size * output_dim * sizeof(float));
    model->max_batch_size = max_batch_size;
    model->norm_obs      = false;
    model->deterministic = false;
    model->obs_norm      = obs_normalizer_make(input_dim);

    free(sizes);

    Space obs_space = {SPACE_TYPE_BOX, input_dim, NULL, NULL};
    Space act_space = {SPACE_TYPE_DISCRETE, output_dim, NULL, NULL};

    return policy_malloc("actor_mlp", obs_space, act_space, model, actor_act, actor_model_free);
}

void actor_model_free(Policy* policy) {
    if (!policy || !policy->model) return;
    ActorModel* model = (ActorModel*)policy->model;
    if (model->nn) {
        seq_nn_free(model->nn);
    }
    if (model->logits_buffer) {
        free(model->logits_buffer);
    }
    if (model->probs_buffer) {
        free(model->probs_buffer);
    }
    obs_normalizer_free(&model->obs_norm);
    free(model);
    policy->model = NULL;
}


