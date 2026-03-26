#include "rl/envs/cartpole.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CARTPOLE_GRAVITY      9.8
#define CARTPOLE_MASSCART     1.0
#define CARTPOLE_MASSPOLE     0.1
#define CARTPOLE_TOTAL_MASS   (CARTPOLE_MASSCART + CARTPOLE_MASSPOLE)
#define CARTPOLE_LENGTH       0.5
#define CARTPOLE_POLEMASS_LEN (CARTPOLE_MASSPOLE * CARTPOLE_LENGTH)
#define CARTPOLE_FORCE_MAG    10.0
#define CARTPOLE_TAU          0.02
#define CARTPOLE_X_THRESHOLD  2.4
#define CARTPOLE_THETA_THRESHOLD_RAD (12.0 * 2.0 * M_PI / 360.0)
#define CARTPOLE_MAX_STEPS_DEFAULT    500

static void cartpole_reset(MDP* env) {
    CartPoleState* state = (CartPoleState*)env->state;
    float* obs = (float*)env->observation;
    for (int i = 0; i < 4; i++) {
        state->state[i] = rand_uniform(-0.05, 0.05);
        obs[i] = (float)state->state[i];
    }
    env->reward = 0;
    env->terminated = false;
    env->truncated = false;
    env->step_count = 0;
}

static void cartpole_step(MDP* env, void* action) {
    CartPoleState* state = (CartPoleState*)env->state;
    double x         = state->state[0];
    double x_dot     = state->state[1];
    double theta     = state->state[2];
    double theta_dot = state->state[3];

    int action_int = *(int*)action;
    double force = (action_int == 1) ? CARTPOLE_FORCE_MAG : -CARTPOLE_FORCE_MAG;

    double costheta = cos(theta);
    double sintheta = sin(theta);

    double temp = (force + CARTPOLE_POLEMASS_LEN * theta_dot * theta_dot * sintheta)
                  / CARTPOLE_TOTAL_MASS;
    double theta_acc = (CARTPOLE_GRAVITY * sintheta - costheta * temp)
                     / (CARTPOLE_LENGTH * (4.0/3.0 - CARTPOLE_MASSPOLE * costheta * costheta
                        / CARTPOLE_TOTAL_MASS));
    double x_acc = temp - CARTPOLE_POLEMASS_LEN * theta_acc * costheta / CARTPOLE_TOTAL_MASS;

    x         += CARTPOLE_TAU * x_dot;
    x_dot     += CARTPOLE_TAU * x_acc;
    theta     += CARTPOLE_TAU * theta_dot;
    theta_dot += CARTPOLE_TAU * theta_acc;

    float* obs = (float*)env->observation;
    obs[0] = (float)x;
    obs[1] = (float)x_dot;
    obs[2] = (float)theta;
    obs[3] = (float)theta_dot;

    state->state[0] = x;
    state->state[1] = x_dot;
    state->state[2] = theta;
    state->state[3] = theta_dot;

    env->step_count++;

    env->terminated = (x < -CARTPOLE_X_THRESHOLD || x > CARTPOLE_X_THRESHOLD
                    || theta < -CARTPOLE_THETA_THRESHOLD_RAD
                    || theta > CARTPOLE_THETA_THRESHOLD_RAD);

    env->truncated = (!env->terminated && env->step_count >= ((CartPoleConfig*)env->config)->max_steps);

    env->reward = 1;
}
CartPoleConfig* cartpole_config_create(void) {
    CartPoleConfig* cfg = (CartPoleConfig*)malloc(sizeof(CartPoleConfig));
    cfg->max_steps = CARTPOLE_MAX_STEPS_DEFAULT;
    return cfg;
}
static void cartpole_free_config(MDP* env) {
    free(env->config);
    env->config = NULL;
}

static void cartpole_free_state(MDP* env) {
    free(env->state);
    env->state = NULL;
}

MDP* cartpole_create(CartPoleConfig* config) {
    Space state_space  = { .type = SPACE_TYPE_BOX, .n = 4, .lows = NULL, .highs = NULL };
    Space action_space = { .type = SPACE_TYPE_DISCRETE, .n = 2, .lows = NULL, .highs = NULL };
    CartPoleState* state = (CartPoleState*)malloc(sizeof(CartPoleState));
    if (!config) config = cartpole_config_create();
    return mdp_malloc("CartPole-v1", state_space, action_space, config, state,
                      cartpole_reset, cartpole_step, cartpole_free_config, cartpole_free_state);
}
