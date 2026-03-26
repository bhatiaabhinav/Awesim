#include "rl/algos/ppo.h"
#include "rl/core/logging.h"
#include "rl/core/rl_utils.h"
#include <string.h>
#include <time.h>

// ===========================================================================
// PPO Training  (discrete action space, vectorised environments)
// ===========================================================================

void ppo_train(VecEnv* venv, Policy* actor, SequentialNN* critic,
               SequentialNN* cost_critic, int num_iters, PPOConfig cfg,
               const char* log_path) {
    int N = venv->num_envs;

    // ---- Hyperparameters from config ----
    float gamma          = cfg.gamma;
    float lam            = cfg.lam;
    float actor_lr_init  = cfg.actor_lr;
    float critic_lr_init = cfg.critic_lr;
    float actor_lr       = actor_lr_init;
    float critic_lr      = critic_lr_init;
    int    M              = cfg.nsteps > 0 ? cfg.nsteps
                          : (2048 > N ? 2048 / N : 1);
    int    nepochs        = cfg.nepochs;
    int    batch_size     = cfg.batch_size;
    float clip_ratio_init = cfg.clip_ratio;
    float clip_ratio      = clip_ratio_init;
    float entropy_coef_init = cfg.entropy_coef;
    float entropy_coef      = entropy_coef_init;
    float max_grad_norm  = cfg.max_grad_norm;
    float beta1 = cfg.beta1, beta2 = cfg.beta2, adam_eps = cfg.adam_eps;
    int    stats_window   = cfg.stats_window;
    float target_kl      = cfg.target_kl;
    bool   decay_lr       = cfg.decay_lr;
    float min_lr         = cfg.min_lr;
    bool   decay_clip     = cfg.decay_clip_ratio;
    float min_clip       = cfg.min_clip_ratio;
    bool   decay_ent      = cfg.decay_entropy_coef;
    float min_ent_coef   = cfg.min_entropy_coef;
    float entropy_target    = cfg.entropy_target;
    float entropy_target_lr = cfg.entropy_target_lr;
    float log_entropy_coef  = logf(entropy_coef);

    // ---- CMDP (optional) ----
    bool     cmdp           = cfg.cmdp;
    float cost_gamma     = cfg.cost_gamma;
    float cost_threshold = cfg.cost_threshold;
    float lagrange_lr    = cfg.lagrange_lr;
    float lagrange_max   = cfg.lagrange_max;
    float log_lagrange   = logf(cfg.lagrange_init + 1e-9f);
    float lagrange       = cfg.lagrange_init;
    const float lagrange_lr_decay = 0.9995f;
    const float lagrange_osc_decay = 0.9f;
    float delta_cost_prev = 0.0f;

    // ---- Secondary reward shaping ----
    float primary_reward_coef = cfg.primary_reward_coef;
    float secondary_reward_coef = cfg.secondary_reward_coef;

    // ---- Extract models ----
    ActorModel* a_m = (ActorModel*)actor->model;
    SequentialNN* a_nn = a_m->nn;
    SequentialNN* c_nn = critic;
    SequentialNN* cc_nn = cmdp ? cost_critic : NULL;
    int obs_dim   = a_m->obs_norm.dim;
    bool norm_obs = a_m->norm_obs;
    int n_actions = actor->action_space.n;
    int total     = M * N;
    int n_minibatches = total / batch_size;

    // ---- Optimisers ----
    AdamState* actor_adam  = adam_malloc_seq(a_nn, actor_lr, beta1, beta2, adam_eps);
    AdamState* critic_adam = adam_malloc_seq(c_nn, critic_lr, beta1, beta2, adam_eps);
    AdamState* cc_adam     = cmdp ? adam_malloc_seq(cc_nn, critic_lr, beta1, beta2, adam_eps) : NULL;
    Tensor* d_critic = tensor1d_malloc(batch_size);
    float critic_loss = 0.0f;
    Tensor* d_actor = tensor2d_malloc(batch_size, n_actions);
    float actor_loss = 0.0f;
    float H_bar = 0.0f;

    // ---- Rollout buffers (Tensor-based) ----
    Tensor* states      = tensor3d_malloc(M + 1, N, obs_dim);
    Tensor* actions     = tensor2d_malloc(M, N);
    Tensor* rewards     = tensor2d_malloc(M, N);
    Tensor* costs       = tensor2d_malloc(M, N);
    Tensor* terminateds = tensor2d_malloc(M, N);
    Tensor* truncateds  = tensor2d_malloc(M, N);
    Tensor* logits_t    = tensor3d_malloc(M, N, n_actions);
    Tensor* probs_t     = tensor3d_malloc(M, N, n_actions);
    Tensor* V           = tensor2d_malloc(M + 1, N);
    Tensor* A           = tensor2d_malloc(M, N);
    Tensor* Vc          = tensor2d_malloc(M + 1, N);
    Tensor* Ac          = tensor2d_malloc(M, N);
    int* actions_t      = (int*)malloc(sizeof(int) * N);
    int* indices        = (int*)malloc(sizeof(int) * total);
    for (int i = 0; i < total; i++) indices[i] = i;

    // scratch for obs normalization during collection
    float* norm_scratch = norm_obs ? (float*)malloc(sizeof(float) * N * obs_dim) : NULL;

    // ---- Minibatch buffers ----
    Tensor* mb_states     = tensor2d_malloc(batch_size, obs_dim);
    Tensor* mb_actions    = tensor1d_malloc(batch_size);
    Tensor* mb_probs      = tensor2d_malloc(batch_size, n_actions);
    Tensor* mb_V_rew      = tensor1d_malloc(batch_size);
    Tensor* mb_A_rew      = tensor1d_malloc(batch_size);
    Tensor* mb_V_cost     = tensor1d_malloc(batch_size);
    Tensor* mb_A_cost     = tensor1d_malloc(batch_size);
    Tensor* mb_A_combined = tensor1d_malloc(batch_size);
    Tensor* mb_V_out      = tensor1d_malloc(batch_size);

    // indexed copy pairs for minibatch (matching srpp pattern)
    Tensor* mb_dst_src_pairs[][2] = {
        {mb_actions, actions}, {mb_V_rew, V}, {mb_A_rew, A},
        {mb_V_cost, Vc}, {mb_A_cost, Ac}
    };
    const int num_mb_pairs = sizeof(mb_dst_src_pairs) / sizeof(mb_dst_src_pairs[0]);

    // ---- Stats tracking (PlayStats) ----
    PlayStats stats = play_stats_init(gamma, cost_gamma, gamma, 0);
    PlayStats iter_stats = play_stats_init(gamma, cost_gamma, gamma, 0);
    PlayStats* vec_stats = malloc(sizeof(PlayStats) * N);
    for (int i = 0; i < N; i++)
        vec_stats[i] = play_stats_init(gamma, cost_gamma, gamma, stats_window / N);
    int total_steps = 0;

    vecenv_reset(venv);

    // Actor parameter backup for target_kl revert
    SequentialNN* actor_old = NULL;
    if (target_kl > 0.0f)
        actor_old = seq_nn_clone(a_nn, false);

    // Print config
    printf("  PPO config: gamma=%.3f lam=%.3f actor_lr=%.1e critic_lr=%.1e "
           "nsteps=%d nepochs=%d batch=%d clip=%.2f ent_coef=%.3f\n",
           gamma, lam, actor_lr, critic_lr, M, nepochs,
           batch_size, clip_ratio, entropy_coef);
    if (target_kl > 0.0f)
        printf("  target_kl=%.4f (early stop at 1.5x, revert at 50x)\n",
               target_kl);
    if (decay_lr)
        printf("  decay_lr: %.1e -> %.1e over %d iters\n",
               actor_lr_init, min_lr, num_iters);
    if (decay_clip)
        printf("  decay_clip: %.3f -> %.3f over %d iters\n",
               clip_ratio_init, min_clip, num_iters);
    if (decay_ent)
        printf("  decay_entropy_coef: %.4f -> %.4f over %d iters\n",
               entropy_coef_init, min_ent_coef, num_iters);
    if (entropy_target > 0.0f)
        printf("  entropy_target=%.4f  entropy_target_lr=%.1e\n",
               entropy_target, entropy_target_lr);
    if (primary_reward_coef != 1.0f)
        printf("  primary_reward_coef=%.4f\n", (double)primary_reward_coef);
    if (secondary_reward_coef != 0.0f)
        printf("  secondary_reward_coef=%.4f\n", (double)secondary_reward_coef);
    if (cmdp)
        printf("  CMDP: cost_gamma=%.3f cost_threshold=%.3f lagrange_lr=%.4f "
               "lagrange_init=%.4f lagrange_max=%.1f\n",
               cost_gamma, cost_threshold, lagrange_lr, lagrange, lagrange_max);

    clock_t iter_start;

    // CSV logger
    enum {
        LOG_ITER, LOG_TOTAL_STEPS, LOG_EPISODES,
        LOG_MEAN_REWARD, LOG_MEAN_DISC_SECONDARY_RETURN,
        LOG_MEAN_DISC_COST, LOG_MEAN_EPISODE_LENGTH, LOG_LAGRANGE,
        LOG_PRIMARY_REWARD_COEF, LOG_SECONDARY_REWARD_COEF,
        LOG_PI_LOSS, LOG_V_LOSS, LOG_ENTROPY,
        LOG_ENTROPY_COEF, LOG_APPROX_KL,
        LOG_CLIP_FRAC, LOG_EXPLAINED_VAR, LOG_MEAN_VALUE, LOG_SPS,
        LOG_NUM_COLS
    };
    const char* log_cols[LOG_NUM_COLS] = {
        "iter", "total_steps", "episodes",
        "mean_return", "mean_disc_secondary_return",
        "mean_disc_cost", "mean_episode_length", "lagrange", "primary_reward_coef",
        "secondary_reward_coef",
        "pi_loss", "v_loss", "entropy",
        "entropy_coef", "approx_kl",
        "clip_frac", "explained_var", "mean_value", "sps"
    };
    CSVLogger  _logger;
    CSVLogger* logger = NULL;
    if (log_path) {
        _logger = csv_logger(log_path, 10, LOG_NUM_COLS, log_cols);
        logger = &_logger;
    }

    // ==================================================================
    // Main loop
    // ==================================================================
    for (int iter = 0; iter < num_iters; iter++) {
        iter_start = clock();

        // ---- Per-iteration annealing ----
        float progress = (float)iter / (float)num_iters;
        if (decay_lr) {
            float lr_scale = 1.0f - progress;
            actor_lr  = actor_lr_init * lr_scale;
            critic_lr = critic_lr_init * lr_scale;
            if (actor_lr  < min_lr) actor_lr  = min_lr;
            if (critic_lr < min_lr) critic_lr = min_lr;
            actor_adam->lr  = actor_lr;
            critic_adam->lr = critic_lr;
            if (cc_adam) cc_adam->lr = critic_lr;
        }
        if (decay_clip) {
            clip_ratio = clip_ratio_init * (1.0f - progress);
            if (clip_ratio < min_clip) clip_ratio = min_clip;
        }
        if (decay_ent) {
            entropy_coef = entropy_coef_init * (1.0f - progress);
            if (entropy_coef < min_ent_coef) entropy_coef = min_ent_coef;
            log_entropy_coef = logf(entropy_coef);
        }

        // ============================================================
        // Collect M steps of experience from N envs
        // ============================================================
        for (int t = 0; t < M; t++) {
            // copy state
            for (int e = 0; e < N; e++) memcpy(tensor3d_at(states, t, e, 0), venv->envs[e]->observation, obs_dim * sizeof(float));

            // forward through actor (with obs normalization if needed)
            float* obs_for_forward = tensor3d_at(states, t, 0, 0);
            if (norm_obs && a_m->obs_norm.count > 0) {
                memcpy(norm_scratch, obs_for_forward, N * obs_dim * sizeof(float));
                obs_normalizer_apply(&a_m->obs_norm, norm_scratch, N);
                obs_for_forward = norm_scratch;
            }
            seq_nn_forward(a_nn, obs_for_forward, tensor3d_at(logits_t, t, 0, 0), N);
            mat_row_softmax(tensor3d_at(probs_t, t, 0, 0), tensor3d_at(logits_t, t, 0, 0), N, n_actions, 1.0f);
            mat_row_sample(actions_t, tensor3d_at(probs_t, t, 0, 0), N, n_actions);

            // step envs
            for (int e = 0; e < N; e++) {
                venv->step_actions[e] = &actions_t[e];
                tensor2d_set_at(actions, t, e, (float)actions_t[e]);
            }
            vecenv_step(venv, venv->step_actions);

            // store results
            memcpy(tensor2d_at(costs, t, 0), venv->step_costs, N * sizeof(float));
            for (int e = 0; e < N; e++) {
                float r  = venv->step_rewards[e];
                float sr = venv->step_secondary_rewards[e];
                tensor2d_set_at(rewards, t, e, primary_reward_coef * r + secondary_reward_coef * sr);
                tensor2d_set_at(terminateds, t, e, (float)venv->step_terminated[e]);
                tensor2d_set_at(truncateds, t, e, (float)venv->step_truncated[e]);
                play_stats_push(&vec_stats[e], r, venv->step_costs[e], sr,
                                venv->step_terminated[e] || venv->step_truncated[e]);
                total_steps++;
            }
        }
        // copy last state
        for (int e = 0; e < N; e++) memcpy(tensor3d_at(states, M, e, 0), venv->envs[e]->observation, obs_dim * sizeof(float));

        // consolidate stats across envs
        iter_stats = play_stats_init(gamma, cost_gamma, gamma, 0);
        for (int e = 0; e < N; e++) {
            play_stats_merge(&iter_stats, &vec_stats[e]);
            play_stats_merge(&stats, &vec_stats[e]);
        }

        // ============================================================
        // Observation normalisation: update stats, then normalise
        // ============================================================
        if (norm_obs) {
            obs_normalizer_update(&a_m->obs_norm, states->data, total);
            obs_normalizer_apply(&a_m->obs_norm, states->data, (M + 1) * N);
        }

        // ============================================================
        // Reward normalisation (scale by std of discounted returns)
        // ============================================================
        if (cfg.reward_norm && stats.num_episodes > 100)
            tensor_scale(rewards, 1.0f / (sqrtf(stats.var_discounted_return) + 1e-8f));

        // ============================================================
        // Values and GAE advantages
        // ============================================================
        seq_nn_forward(c_nn, states->data, V->data, (M + 1) * N);
        float* _v = V->data;
        float* v_ = tensor2d_at(V, 1, 0);
        for (int i = 0; i < total; i++) {
            A->data[i] = rewards->data[i] + (1.0f - terminateds->data[i]) * gamma * v_[i] - _v[i];
            A->data[i] *= (1.0f - truncateds->data[i]);
        }
        for (int i = total - 1; i >= 0; i--) {
            float mask = (terminateds->data[i] || truncateds->data[i]) ? 0.0f : 1.0f;
            float adv_next = (i >= (M - 1) * N) ? 0.0f : A->data[i + N];
            A->data[i] += gamma * lam * mask * adv_next;
        }

        // Cost critic values + cost GAE (CMDP)
        if (cmdp) {
            seq_nn_forward(cc_nn, states->data, Vc->data, (M + 1) * N);
            float* _cv = Vc->data;
            float* cv_ = tensor2d_at(Vc, 1, 0);
            for (int i = 0; i < total; i++) {
                Ac->data[i] = costs->data[i] + (1.0f - terminateds->data[i]) * cost_gamma * cv_[i] - _cv[i];
                Ac->data[i] *= (1.0f - truncateds->data[i]);
            }
            for (int i = total - 1; i >= 0; i--) {
                float mask = (terminateds->data[i] || truncateds->data[i]) ? 0.0f : 1.0f;
                float adv_next = (i >= (M - 1) * N) ? 0.0f : Ac->data[i + N];
                Ac->data[i] += cost_gamma * lam * mask * adv_next;
            }
        }

        // ============================================================
        // PPO epochs
        // ============================================================
        float sum_approx_kl = 0.0f;
        float sum_clip_frac = 0.0f;
        int   n_updates     = 0;
        bool  stop_actor    = false;
        float epoch_kl      = 0.0f;
        float epoch_kl_old  = 0.0f;

        if (target_kl > 0.0f)
            seq_nn_copy_params(actor_old, a_nn);

        for (int epoch = 0; epoch < nepochs; epoch++) {
            if (target_kl > 0.0f && !stop_actor) {
                epoch_kl_old = epoch_kl;
                seq_nn_copy_params(actor_old, a_nn);
            }

            shuffle(indices, total);
            critic_loss = 0.0f;
            actor_loss = 0.0f;
            H_bar = 0.0f;

            for (int mb = 0; mb < n_minibatches; mb++) {
                // prepare minibatch (matching srpp pattern)
                for (int i = 0; i < batch_size; i++) {
                    int idx = indices[mb * batch_size + i];
                    memcpy(&mb_states->data[i * obs_dim], &states->data[idx * obs_dim], obs_dim * sizeof(float));
                    memcpy(&mb_probs->data[i * n_actions], &probs_t->data[idx * n_actions], n_actions * sizeof(float));
                    for (int j = 0; j < num_mb_pairs; j++) {
                        Tensor* dst = mb_dst_src_pairs[j][0];
                        Tensor* src = mb_dst_src_pairs[j][1];
                        dst->data[i] = src->data[idx];
                    }
                }

                // ---- Critic update ----
                seq_nn_set_backprop(c_nn, true);
                seq_nn_forward(c_nn, mb_states->data, mb_V_out->data, batch_size);
                for (int i = 0; i < batch_size; i++) {
                    float error = mb_V_out->data[i] - mb_V_rew->data[i] - mb_A_rew->data[i];
                    d_critic->data[i] = 2.0f * error / batch_size;
                    critic_loss += error * error / total;
                }
                seq_nn_zero_grad(c_nn);
                seq_nn_backward(c_nn, d_critic->data, NULL, batch_size);
                grad_clip_norm(c_nn->grad_views, c_nn->num_param_tensors, max_grad_norm);
                adam_step_seq(critic_adam, c_nn);
                seq_nn_set_backprop(c_nn, false);

                // ---- Cost critic update (CMDP) ----
                if (cmdp) {
                    seq_nn_set_backprop(cc_nn, true);
                    seq_nn_forward(cc_nn, mb_states->data, mb_V_out->data, batch_size);
                    for (int i = 0; i < batch_size; i++) {
                        float error = mb_V_out->data[i] - mb_V_cost->data[i] - mb_A_cost->data[i];
                        d_critic->data[i] = 2.0f * error / batch_size;
                    }
                    seq_nn_zero_grad(cc_nn);
                    seq_nn_backward(cc_nn, d_critic->data, NULL, batch_size);
                    grad_clip_norm(cc_nn->grad_views, cc_nn->num_param_tensors, max_grad_norm);
                    adam_step_seq(cc_adam, cc_nn);
                    seq_nn_set_backprop(cc_nn, false);
                }

                // ---- Combined advantage: combine, normalise ----
                for (int i = 0; i < batch_size; i++)
                    mb_A_combined->data[i] = mb_A_rew->data[i];
                if (cmdp) {
                    for (int i = 0; i < batch_size; i++)
                        mb_A_combined->data[i] -= lagrange * mb_A_cost->data[i];
                }
                float a_mean, a_var;
                vec_mean_and_var(mb_A_combined->data, batch_size, &a_mean, &a_var);
                vec_shift_and_scale(mb_A_combined->data, batch_size, -a_mean, 1.0f / (sqrtf(a_var) + 1e-8f));

                // ---- Actor update (PPO clipped, softmax Jacobian) ----
                // PPO clipped surrogate objective (maximize):
                //   L(θ) = (1/n) Σᵢ min(rᵢ·Âᵢ, clip(rᵢ, 1-ε, 1+ε)·Âᵢ) + c·H(πθ)
                //   where rᵢ = πθ(aᵢ|sᵢ) / π_old(aᵢ|sᵢ), c = entropy_coef
                //
                // dL_clip/dπ(aᵢ) = Âᵢ / π_old(aᵢ)  when ratio NOT clipped, else 0
                // Entropy: ∂H/∂πₐ = -(log πₐ + 1)
                // Combined: gₐ = ∂L_clip/∂πₐ - c·(log πₐ + 1)
                // Softmax Jacobian with baseline ḡ = Σ_b πb·gb:
                //   ∂L/∂zₐ = πₐ·(gₐ - ḡ), negate for descent.
                tensor_zero(d_actor);
                seq_nn_set_backprop(a_nn, true);
                seq_nn_forward(a_nn, mb_states->data, a_m->logits_buffer, batch_size);
                mat_row_softmax(a_m->probs_buffer, a_m->logits_buffer, batch_size, n_actions, 1.0f);

                for (int i = 0; i < batch_size; i++) {
                    int a_i = (int)mb_actions->data[i];
                    float pi_old_ai = mb_probs->data[i * n_actions + a_i];
                    float pi_new_ai = a_m->probs_buffer[i * n_actions + a_i];
                    float ratio = pi_new_ai / (pi_old_ai + 1e-10f);
                    float A_i = mb_A_combined->data[i];

                    float surr1 = ratio * A_i;
                    float clipped_ratio = fmaxf(fminf(ratio, 1.0f + clip_ratio), 1.0f - clip_ratio);
                    float surr2 = clipped_ratio * A_i;

                    // dL_clip/dπ(aᵢ): gradient flows through surr1 when it's the min, else clipped (0)
                    float dL_clip = (surr1 <= surr2) ? A_i / (pi_old_ai + 1e-10f) : 0.0f;

                    // approx KL and clip fraction
                    float log_ratio = logf(ratio + 1e-10f);
                    sum_approx_kl += (ratio - 1.0f - log_ratio) / total;
                    if (ratio < 1.0f - clip_ratio || ratio > 1.0f + clip_ratio)
                        sum_clip_frac += 1.0f / total;

                    // Pass 1: compute gₐ = ∂L/∂πₐ and baseline ḡ = Σ_b πb·gb
                    float g_bar = 0.0f;
                    float H_i = 0.0f;
                    for (int a = 0; a < n_actions; a++) {
                        float pi_a = a_m->probs_buffer[i * n_actions + a];
                        float log_pi_a = logf(pi_a + 1e-10f);
                        float g_a = (a == a_i ? dL_clip : 0.0f) - entropy_coef * (log_pi_a + 1.0f);
                        g_bar += pi_a * g_a;
                        H_i -= pi_a * log_pi_a;
                    }
                    H_bar += H_i / total;

                    // Pass 2: ∂L/∂zₐ = πₐ·(gₐ - ḡ), negate for descent
                    for (int a = 0; a < n_actions; a++) {
                        float pi_a = a_m->probs_buffer[i * n_actions + a];
                        float log_pi_a = logf(pi_a + 1e-10f);
                        float g_a = (a == a_i ? dL_clip : 0.0f) - entropy_coef * (log_pi_a + 1.0f);
                        d_actor->data[i * n_actions + a] = -pi_a * (g_a - g_bar) / batch_size;
                    }

                    // loss for logging (negative of objective)
                    actor_loss -= fminf(surr1, surr2) / total;
                }

                if (!stop_actor) {
                    seq_nn_zero_grad(a_nn);
                    seq_nn_backward(a_nn, d_actor->data, NULL, batch_size);
                    grad_clip_norm(a_nn->grad_views, a_nn->num_param_tensors, max_grad_norm);
                    adam_step_seq(actor_adam, a_nn);
                }
                seq_nn_set_backprop(a_nn, false);

                n_updates++;
            }

            // Adapt entropy coefficient toward target entropy (SAC-style, in log space)
            if (entropy_target > 0.0f) {
                log_entropy_coef -= entropy_target_lr * (H_bar - entropy_target);
                log_entropy_coef = fmaxf(fminf(log_entropy_coef, 1.0f), -4.0f);
                entropy_coef = expf(log_entropy_coef);
            }

            // ---- Target KL check after each epoch ----
            if (target_kl > 0.0f && !stop_actor) {
                float full_kl = 0.0f;
                float* kl_logits = (float*)malloc(sizeof(float) * total * n_actions);
                float* kl_probs  = (float*)malloc(sizeof(float) * total * n_actions);
                seq_nn_forward(a_nn, states->data, kl_logits, total);
                mat_row_softmax(kl_probs, kl_logits, total, n_actions, 1.0f);
                for (int i = 0; i < total; i++) {
                    int a = (int)actions->data[i];
                    float old_p = probs_t->data[i * n_actions + a];
                    float new_p = kl_probs[i * n_actions + a];
                    float ratio = new_p / (old_p + 1e-10f);
                    float log_r = logf(ratio + 1e-10f);
                    full_kl += ratio - 1.0f - log_r;
                }
                full_kl /= total;
                epoch_kl = full_kl;
                free(kl_logits);
                free(kl_probs);

                if (epoch_kl > 1.5f * target_kl) {
                    stop_actor = true;
                    if (epoch_kl > 50.0f * target_kl) {
                        seq_nn_copy_params(a_nn, actor_old);
                        epoch_kl = epoch_kl_old;
                    }
                }
            }
        }

        // ============================================================
        // Lagrange multiplier update (CMDP) — SGD in log-λ space
        // ============================================================
        if (cmdp) {
            float delta_cost = iter_stats.mean_discounted_cost - cost_threshold;
            log_lagrange += lagrange_lr * delta_cost;
            log_lagrange = fmaxf(fminf(log_lagrange, logf(lagrange_max + 1e-9f)), -9.0f);
            lagrange = expf(log_lagrange);
            lagrange_lr *= lagrange_lr_decay;
            if (iter >= 100) {
                if (delta_cost_prev > 0.0f && delta_cost < 0.0f)
                    lagrange_lr *= lagrange_osc_decay;
            }
            delta_cost_prev = delta_cost;
        }

        // ============================================================
        // Compute explained variance
        // ============================================================
        float val_mean = 0.0f;
        float ret_mean = 0.0f;
        for (int i = 0; i < total; i++) {
            val_mean += V->data[i];
            ret_mean += V->data[i] + A->data[i];
        }
        val_mean /= total;
        ret_mean /= total;
        float var_ret = 0.0f, var_residual = 0.0f;
        for (int i = 0; i < total; i++) {
            float r = V->data[i] + A->data[i] - ret_mean;
            var_ret += r * r;
            var_residual += A->data[i] * A->data[i];
        }
        var_ret /= total;
        var_residual /= total;
        float explained_var = (var_ret > 1e-8f)
            ? 1.0f - var_residual / var_ret : 0.0f;

        // ============================================================
        // Logging
        // ============================================================
        double iter_secs = (double)(clock() - iter_start) / CLOCKS_PER_SEC;
        int    sps = (iter_secs > 0.0) ? (int)(total / iter_secs) : 0;

        float mean_r       = iter_stats.mean_return;
        float mean_cost_ep = iter_stats.mean_discounted_cost;
        float mean_ep_len  = iter_stats.mean_episode_length;
        float mean_sr_ep   = iter_stats.mean_discounted_secondary_return;

        float avg_approx_kl  = n_updates > 0 ? sum_approx_kl  / n_updates : 0.0f;
        float avg_clip_frac  = n_updates > 0 ? sum_clip_frac  / n_updates : 0.0f;

        if (cmdp) {
            printf("  iter %3d/%d | steps %7d | eps %4d "
                   "| mean_%d %.1f | disc_sr %.2f | len %.1f",
                   iter + 1, num_iters, total_steps, stats.num_episodes,
                   stats_window, (double)mean_r,
                   (double)mean_sr_ep, (double)mean_ep_len);
            if (mean_cost_ep > 0.0f)
                printf(" | disc_cost %.3f (thresh %.2f)", (double)mean_cost_ep, (double)cost_threshold);
            printf(" | lam %.4f "
                   "| pi %.4f | v %.4f "
                   "| ent %.4f (c=%.3f) | kl %.5f | clip %.3f "
                   "| ev %.3f | val %.3f | SPS %d\n",
                   (double)lagrange,
                   (double)actor_loss, (double)critic_loss,
                   (double)H_bar, (double)entropy_coef, (double)avg_approx_kl,
                   (double)avg_clip_frac,
                   (double)explained_var, (double)val_mean, sps);
        } else {
            printf("  iter %3d/%d | steps %7d | episodes %4d "
                   "| mean_%d %.1f | disc_sr %.2f | len %.1f",
                   iter + 1, num_iters, total_steps, stats.num_episodes,
                   stats_window, (double)mean_r,
                   (double)mean_sr_ep, (double)mean_ep_len);
            if (mean_cost_ep > 0.0f)
                printf(" | disc_cost %.3f", (double)mean_cost_ep);
            printf(" | pi_loss %.4f | v_loss %.4f "
                   "| entropy %.4f (c=%.3f) | kl %.5f | clip %.3f "
                   "| expl_var %.3f | val %.3f | SPS %d\n",
                   (double)actor_loss, (double)critic_loss,
                   (double)H_bar, (double)entropy_coef, (double)avg_approx_kl,
                   (double)avg_clip_frac,
                   (double)explained_var, (double)val_mean, sps);
        }

        // CSV log row
        if (logger) {
            csv_logger_log_int(logger, LOG_ITER, iter + 1);
            csv_logger_log_int(logger, LOG_TOTAL_STEPS, total_steps);
            csv_logger_log_int(logger, LOG_EPISODES, stats.num_episodes);
            csv_logger_log(logger, LOG_MEAN_REWARD, (double)mean_r);
            csv_logger_log(logger, LOG_MEAN_DISC_SECONDARY_RETURN, (double)mean_sr_ep);
            csv_logger_log(logger, LOG_MEAN_DISC_COST, (double)mean_cost_ep);
            csv_logger_log(logger, LOG_MEAN_EPISODE_LENGTH, (double)mean_ep_len);
            csv_logger_log(logger, LOG_LAGRANGE, (double)lagrange);
            csv_logger_log(logger, LOG_PI_LOSS, (double)actor_loss);
            csv_logger_log(logger, LOG_V_LOSS, (double)critic_loss);
            csv_logger_log(logger, LOG_ENTROPY, (double)H_bar);
            csv_logger_log(logger, LOG_ENTROPY_COEF, (double)entropy_coef);
            csv_logger_log(logger, LOG_APPROX_KL, (double)avg_approx_kl);
            csv_logger_log(logger, LOG_CLIP_FRAC, (double)avg_clip_frac);
            csv_logger_log(logger, LOG_EXPLAINED_VAR, (double)explained_var);
            csv_logger_log(logger, LOG_MEAN_VALUE, (double)val_mean);
            csv_logger_log(logger, LOG_PRIMARY_REWARD_COEF, (double)primary_reward_coef);
            csv_logger_log(logger, LOG_SECONDARY_REWARD_COEF, (double)secondary_reward_coef);
            csv_logger_log_int(logger, LOG_SPS, sps);
            csv_logger_commit_row(logger);
        }
    }

    // ---- Cleanup ----
    tensors_free((Tensor*[]){states, actions, rewards, costs, terminateds, truncateds,
                              logits_t, probs_t, V, A, Vc, Ac,
                              mb_states, mb_actions, mb_probs,
                              mb_V_rew, mb_A_rew, mb_V_cost, mb_A_cost,
                              mb_A_combined, mb_V_out, d_critic, d_actor, NULL});
    free(actions_t); free(indices); free(norm_scratch);
    for (int i = 0; i < N; i++) play_stats_free_buffers(&vec_stats[i]);
    free(vec_stats);
    adam_free(actor_adam);
    adam_free(critic_adam);
    if (cc_adam) adam_free(cc_adam);
    if (actor_old) seq_nn_free(actor_old);
    if (logger) csv_logger_close(logger);
}
