#ifndef RANDOM_MDP_H
#define RANDOM_MDP_H

#include "rl/core/mdp.h"

typedef struct RandomMDPConfig {
    int num_states;
    int num_actions;
    int max_steps;
    double alpha;               // Dirichlet concentration for transitions
    double beta;                // reward spread: R[s,a] ~ N(1, beta) or U[1-beta, 1+beta]
    bool uniform_dist_rewards;  // if true, rewards ~ Uniform; otherwise Normal
    Tensor* T;  // transition probabilities T[s, a, s'] = Pr(s'| s, a)
    Tensor* R;  // mean rewards R[s, a] (2D: num_states x num_actions)
    Tensor* C;  // mean costs C[s, a] (2D: num_states x num_actions)
    Tensor* SR; // mean secondary rewards SR[s, a] (2D: num_states x num_actions)
    Tensor* d0; // start state probabilities (num_states); deterministic at state 0
} RandomMDPConfig;

typedef struct RandomMDPState {
    int state;  // current state index
} RandomMDPState;

// Create a random discrete MDP (matches Julia RandomDiscreteMDP).
// alpha: Dirichlet concentration (transitions)
// beta:  reward spread around mean 1.0
// uniform_dist_rewards: if true, R[s,a] ~ U[1-beta, 1+beta]; else R[s,a] ~ N(1, beta)
// max_steps: episode truncation length (0 = no truncation)
// Step reward is stochastic: R[s,a] + N(0,1)
MDP* random_mdp_create(int num_states, int num_actions, double alpha, double beta,
                       bool uniform_dist_rewards, int max_steps);

#endif // RANDOM_MDP_H