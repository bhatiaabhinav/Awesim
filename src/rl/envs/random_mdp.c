#include "rl/envs/random_mdp.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- Dirichlet sampling via Gamma(alpha, 1) ----------

// Sample Gamma(alpha, 1) using Marsaglia & Tsang's method (alpha >= 1)
// For alpha < 1, use the relation: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
static double rand_gamma(double alpha) {
    if (alpha < 1.0) {
        double u = (double)rand() / RAND_MAX;
        if (u < 1e-15) u = 1e-15;
        return rand_gamma(alpha + 1.0) * pow(u, 1.0 / alpha);
    }
    double d = alpha - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    while (1) {
        double x, v;
        do {
            double u1 = (double)rand() / RAND_MAX;
            double u2 = (double)rand() / RAND_MAX;
            if (u1 < 1e-15) u1 = 1e-15;
            x = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        double u = (double)rand() / RAND_MAX;
        if (u < 1e-15) u = 1e-15;
        if (u < 1.0 - 0.0331 * (x * x) * (x * x))
            return d * v;
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
            return d * v;
    }
}

// Sample a Dirichlet(alpha, ..., alpha) vector of length n into out[]
static void rand_dirichlet(double* out, int n, double alpha) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        out[i] = rand_gamma(alpha);
        if (out[i] < 1e-15) out[i] = 1e-15;
        sum += out[i];
    }
    for (int i = 0; i < n; i++)
        out[i] /= sum;
}

// Standard normal via Box-Muller
static double rand_standard_normal(void) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// ---------- MDP callbacks ----------

static float random_mdp_transition_prob(MDP* env, int s, int a, int s_next) {
    RandomMDPConfig* cfg = (RandomMDPConfig*)env->config;
    return tensor3d_get_at(cfg->T, s, a, s_next);
}

// Mean reward R(s,a) — does not depend on s_next
static float random_mdp_reward_fn(MDP* env, int s, int a, int s_next) {
    (void)s_next;
    RandomMDPConfig* cfg = (RandomMDPConfig*)env->config;
    return tensor2d_get_at(cfg->R, s, a);
}

// Mean cost C(s,a) — does not depend on s_next
static float random_mdp_cost_fn(MDP* env, int s, int a, int s_next) {
    (void)s_next;
    RandomMDPConfig* cfg = (RandomMDPConfig*)env->config;
    return tensor2d_get_at(cfg->C, s, a);
}

// Mean secondary reward SR(s,a) — does not depend on s_next
static float random_mdp_secondary_reward_fn(MDP* env, int s, int a, int s_next) {
    (void)s_next;
    RandomMDPConfig* cfg = (RandomMDPConfig*)env->config;
    return tensor2d_get_at(cfg->SR, s, a);
}

static float random_mdp_start_prob(MDP* env, int s) {
    RandomMDPConfig* cfg = (RandomMDPConfig*)env->config;
    return tensor1d_get_at(cfg->d0, s);
}

static void random_mdp_reset(MDP* env) {
    RandomMDPConfig* cfg = (RandomMDPConfig*)env->config;
    RandomMDPState* state = (RandomMDPState*)env->state;

    // Sample start state from d0
    double r = rand_uniform(0.0, 1.0);
    double cumsum = 0;
    state->state = 0;
    for (int s = 0; s < cfg->num_states; s++) {
        cumsum += tensor1d_get_at(cfg->d0, s);
        if (r <= cumsum) {
            state->state = s;
            break;
        }
    }
    *(int*)env->observation = state->state;
    env->reward = 0;
    env->terminated = false;
    env->truncated = false;
    env->step_count = 0;
}

static void random_mdp_step(MDP* env, void* action) {
    RandomMDPConfig* cfg = (RandomMDPConfig*)env->config;
    RandomMDPState* state = (RandomMDPState*)env->state;
    int a = *(int*)action;
    int s = state->state;

    // Sample next state from T[s, a, :]
    double r = rand_uniform(0.0, 1.0);
    double cumsum = 0;
    int s_next = 0;
    for (int sp = 0; sp < cfg->num_states; sp++) {
        cumsum += tensor3d_get_at(cfg->T, s, a, sp);
        if (r <= cumsum) {
            s_next = sp;
            break;
        }
    }

    // Stochastic reward: R(s,a) + N(0,1)
    float mean_reward = tensor2d_get_at(cfg->R, s, a);
    env->reward = mean_reward + (float)rand_standard_normal();

    // Stochastic cost: C(s,a) + N(0,1)
    float mean_cost = tensor2d_get_at(cfg->C, s, a);
    env->cost = mean_cost + (float)rand_standard_normal();

    // Stochastic secondary reward: SR(s,a) + N(0,1)
    float mean_sr = tensor2d_get_at(cfg->SR, s, a);
    env->secondary_reward = mean_sr + (float)rand_standard_normal();

    state->state = s_next;
    *(int*)env->observation = s_next;
    env->step_count++;
    env->terminated = false;  // no absorbing states
    env->truncated = (cfg->max_steps > 0 && env->step_count >= cfg->max_steps);
}

static void random_mdp_free_config(MDP* env) {
    RandomMDPConfig* cfg = (RandomMDPConfig*)env->config;
    if (cfg) {
        tensor_free(cfg->T);
        tensor_free(cfg->R);
        tensor_free(cfg->C);
        tensor_free(cfg->SR);
        tensor_free(cfg->d0);
        free(cfg);
        env->config = NULL;
    }
}

static void random_mdp_free_state(MDP* env) {
    free(env->state);
    env->state = NULL;
}

MDP* random_mdp_create(int num_states, int num_actions, double alpha, double beta,
                       bool uniform_dist_rewards, int max_steps) {
    RandomMDPConfig* cfg = (RandomMDPConfig*)malloc(sizeof(RandomMDPConfig));
    cfg->num_states = num_states;
    cfg->num_actions = num_actions;
    cfg->max_steps = max_steps;
    cfg->alpha = alpha;
    cfg->beta = beta;
    cfg->uniform_dist_rewards = uniform_dist_rewards;

    // Allocate T (3D), R (2D), d0 (1D)
    cfg->T  = tensor3d_malloc(num_states, num_actions, num_states);
    cfg->R  = tensor2d_malloc(num_states, num_actions);
    cfg->C  = tensor2d_malloc(num_states, num_actions);
    cfg->SR = tensor2d_malloc(num_states, num_actions);
    cfg->d0 = tensor1d_malloc(num_states);

    // Fill transition matrix: for each (s,a), T[s,a,:] ~ Dirichlet(alpha)
    double* dirichlet_buf = (double*)malloc(sizeof(double) * num_states);
    for (int s = 0; s < num_states; s++) {
        for (int a = 0; a < num_actions; a++) {
            rand_dirichlet(dirichlet_buf, num_states, alpha);
            for (int sp = 0; sp < num_states; sp++) {
                tensor3d_set_at(cfg->T, s, a, sp, (float)dirichlet_buf[sp]);
            }
        }
    }
    free(dirichlet_buf);

    // Fill mean reward matrix R[s,a]
    for (int s = 0; s < num_states; s++) {
        for (int a = 0; a < num_actions; a++) {
            double r;
            if (uniform_dist_rewards) {
                r = rand_uniform(1.0 - beta, 1.0 + beta);
            } else {
                r = 1.0 + beta * rand_standard_normal();
            }
            tensor2d_set_at(cfg->R, s, a, (float)r);
        }
    }

    // Fill mean cost matrix C[s,a] ~ |N(0,1)| (non-negative costs)
    for (int s = 0; s < num_states; s++) {
        for (int a = 0; a < num_actions; a++) {
            double c = fabs(rand_standard_normal());
            tensor2d_set_at(cfg->C, s, a, (float)c);
        }
    }

    // Fill mean secondary reward matrix SR[s,a] ~ N(0, 1)
    for (int s = 0; s < num_states; s++) {
        for (int a = 0; a < num_actions; a++) {
            double sr = rand_standard_normal();
            tensor2d_set_at(cfg->SR, s, a, (float)sr);
        }
    }

    // Start state distribution: deterministic at state 0
    tensor_zero(cfg->d0);
    tensor1d_set_at(cfg->d0, 0, 1.0f);

    // Build MDP
    Space state_space  = { .type = SPACE_TYPE_DISCRETE, .n = num_states,  .lows = NULL, .highs = NULL };
    Space action_space = { .type = SPACE_TYPE_DISCRETE, .n = num_actions, .lows = NULL, .highs = NULL };
    RandomMDPState* state = (RandomMDPState*)malloc(sizeof(RandomMDPState));
    state->state = 0;

    MDP* env = mdp_malloc("RandomDiscreteMDP", state_space, action_space, cfg, state,
                          random_mdp_reset, random_mdp_step,
                          random_mdp_free_config, random_mdp_free_state);
    env->transition_probability  = random_mdp_transition_prob;
    env->reward_function         = random_mdp_reward_fn;
    env->cost_function           = random_mdp_cost_fn;
    env->secondary_reward_function = random_mdp_secondary_reward_fn;
    env->start_state_probability = random_mdp_start_prob;

    return env;
}
