#include "rl/envs/acrobot.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- Physical constants (Gymnasium defaults) ----

#define ACRO_LINK_LENGTH_1   1.0
#define ACRO_LINK_LENGTH_2   1.0
#define ACRO_LINK_MASS_1     1.0
#define ACRO_LINK_MASS_2     1.0
#define ACRO_LINK_COM_POS_1  0.5
#define ACRO_LINK_COM_POS_2  0.5
#define ACRO_LINK_MOI        1.0
#define ACRO_GRAVITY         9.8
#define ACRO_DT              0.2
#define ACRO_MAX_VEL_1       (4.0 * M_PI)
#define ACRO_MAX_VEL_2       (9.0 * M_PI)
#define ACRO_MAX_STEPS       500

// ---- Helpers ----

static inline double wrap_angle(double x) {
    while (x > M_PI)  x -= 2.0 * M_PI;
    while (x < -M_PI) x += 2.0 * M_PI;
    return x;
}

static inline double clip(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// ---- Equations of motion (Sutton & Barto) ----

static void acrobot_dsdt(const double s[4], double torque, double out[4]) {
    double theta1  = s[0];
    double theta2  = s[1];
    double dtheta1 = s[2];
    double dtheta2 = s[3];

    double m1  = ACRO_LINK_MASS_1;
    double m2  = ACRO_LINK_MASS_2;
    double l1  = ACRO_LINK_LENGTH_1;
    double lc1 = ACRO_LINK_COM_POS_1;
    double lc2 = ACRO_LINK_COM_POS_2;
    double I1  = ACRO_LINK_MOI;
    double I2  = ACRO_LINK_MOI;
    double g   = ACRO_GRAVITY;

    double d1 = m1 * lc1 * lc1
              + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * cos(theta2))
              + I1 + I2;
    double d2 = m2 * (lc2 * lc2 + l1 * lc2 * cos(theta2)) + I2;

    double phi2 = m2 * lc2 * g * cos(theta1 + theta2 - M_PI / 2.0);
    double phi1 = -m2 * l1 * lc2 * dtheta2 * dtheta2 * sin(theta2)
                - 2.0 * m2 * l1 * lc2 * dtheta1 * dtheta2 * sin(theta2)
                + (m1 * lc1 + m2 * l1) * g * cos(theta1 - M_PI / 2.0)
                + phi2;

    double ddtheta2 = (torque
                     + d2 / d1 * phi1
                     - m2 * l1 * lc2 * dtheta1 * dtheta1 * sin(theta2)
                     - phi2)
                    / (m2 * lc2 * lc2 + I2 - d2 * d2 / d1);
    double ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

    out[0] = dtheta1;
    out[1] = dtheta2;
    out[2] = ddtheta1;
    out[3] = ddtheta2;
}

// ---- RK4 integration ----

static void acrobot_rk4(double s[4], double torque) {
    double dt  = ACRO_DT;
    double dt2 = dt / 2.0;

    double k1[4], k2[4], k3[4], k4[4], tmp[4];

    acrobot_dsdt(s, torque, k1);
    for (int i = 0; i < 4; i++) tmp[i] = s[i] + dt2 * k1[i];
    acrobot_dsdt(tmp, torque, k2);
    for (int i = 0; i < 4; i++) tmp[i] = s[i] + dt2 * k2[i];
    acrobot_dsdt(tmp, torque, k3);
    for (int i = 0; i < 4; i++) tmp[i] = s[i] + dt * k3[i];
    acrobot_dsdt(tmp, torque, k4);

    for (int i = 0; i < 4; i++)
        s[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

static void acrobot_set_obs(MDP* env, const double s[4]) {
    float* obs = (float*)env->observation;
    obs[0] = (float)cos(s[0]);
    obs[1] = (float)sin(s[0]);
    obs[2] = (float)cos(s[1]);
    obs[3] = (float)sin(s[1]);
    obs[4] = (float)s[2];
    obs[5] = (float)s[3];
}

static bool acrobot_terminal(const double s[4]) {
    return -cos(s[0]) - cos(s[1] + s[0]) > 1.0;
}

static void acrobot_reset(MDP* env) {
    AcrobotState* state = (AcrobotState*)env->state;
    for (int i = 0; i < 4; i++)
        state->state[i] = rand_uniform(-0.1, 0.1);
    acrobot_set_obs(env, state->state);
    env->reward = 0;
    env->terminated = false;
    env->truncated = false;
    env->step_count = 0;
}

static void acrobot_step(MDP* env, void* action) {
    AcrobotState* state = (AcrobotState*)env->state;
    double* s = state->state;

    double torque = (double)(*(int*)action - 1);
    acrobot_rk4(s, torque);

    s[0] = wrap_angle(s[0]);
    s[1] = wrap_angle(s[1]);
    s[2] = clip(s[2], -ACRO_MAX_VEL_1, ACRO_MAX_VEL_1);
    s[3] = clip(s[3], -ACRO_MAX_VEL_2, ACRO_MAX_VEL_2);

    acrobot_set_obs(env, s);

    env->step_count++;
    env->terminated = acrobot_terminal(s);
    env->truncated = (!env->terminated && env->step_count >= ACRO_MAX_STEPS);
    env->reward = env->terminated ? 0.0 : -1.0;
}

static void acrobot_free_state(MDP* env) {
    free(env->state);
    env->state = NULL;
}

MDP* acrobot_create(void) {
    Space state_space  = { .type = SPACE_TYPE_BOX, .n = 6, .lows = NULL, .highs = NULL };
    Space action_space = { .type = SPACE_TYPE_DISCRETE, .n = 3, .lows = NULL, .highs = NULL };
    AcrobotState* state = (AcrobotState*)malloc(sizeof(AcrobotState));
    memset(state->state, 0, sizeof(state->state));
    return mdp_malloc("Acrobot-v1", state_space, action_space, NULL, state,
                      acrobot_reset, acrobot_step, NULL, acrobot_free_state);
}
