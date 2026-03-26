#ifndef CGYM_CARTPOLE_H
#define CGYM_CARTPOLE_H

#include "rl/core/mdp.h"

typedef struct CartPoleConfig {
    int max_steps;
} CartPoleConfig;
CartPoleConfig* cartpole_config_create(void);

typedef struct CartPoleState {
    double state[4]; // double-precision internal state for accurate dynamics
} CartPoleState;

// passing NULL will use default config
MDP* cartpole_create(CartPoleConfig* config);

#endif // CGYM_CARTPOLE_H
