#ifndef CGYM_ACROBOT_H
#define CGYM_ACROBOT_H

#include "rl/core/mdp.h"

typedef struct AcrobotState {
    double state[4]; // [theta1, theta2, dtheta1, dtheta2] double-precision
} AcrobotState;

MDP* acrobot_create();

#endif // CGYM_ACROBOT_H
