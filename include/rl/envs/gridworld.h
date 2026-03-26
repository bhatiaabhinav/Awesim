#ifndef GRIDWORLD_H
#define GRIDWORLD_H

#include "rl/core/mdp.h"

#define GW_NUM_ACTIONS 5
#define GW_NORTH 0
#define GW_EAST  1
#define GW_SOUTH 2
#define GW_WEST  3
#define GW_NOOP  4

typedef struct GridWorldConfig {
    int nrows;
    int ncols;
    int num_states;
    int max_steps;
    unsigned char* grid;  // row-major, nrows * ncols
    double enter_rewards[256];
    double leave_rewards[256];
    double enter_costs[256];
    double leave_costs[256];
    double enter_secondary_rewards[256];
    double leave_secondary_rewards[256];
    double step_reward;          // added to reward on every non-absorbing step
    double step_cost;            // added to cost on every non-absorbing step
    double step_secondary_reward;// added to secondary reward on every non-absorbing step
    double noop_prob[256];     // per-tile failure mode: probability of unintended noop
    double slip_prob[256];     // per-tile failure mode: probability of 90-degree slip
    bool absorbing[256];       // which tile types are absorbing
    bool self_loops_accumulate_rewards;             //  true by default. If false, self-transitions yield 0 primary reward
    bool self_loops_accumulate_costs;               //  true by default. If false, self-transitions yield 0 cost
    bool self_loops_accumulate_secondary_rewards;   //  true by default. If false, self-transitions yield 0 secondary reward
    Tensor* T;   // transition probabilities T[s, a, s'] (derived from grid and failure probabilities). 'S' are always start, 'G' are always goal, 'O' are always blocked, but other tiles can be configured with different rewards/costs and failure probabilities.
    Tensor* R;   // rewards R[s, a, s'] (derived from enter/leave rewards and grid)
    Tensor* C;   // costs C[s, a, s'] (derived from enter/leave costs and grid)
    Tensor* SR;  // secondary rewards SR[s, a, s'] (derived from enter/leave secondary rewards and grid)
    Tensor* d0;  // start state distribution (derived from grid: uniform over 'S' tiles)
    // manual adjustments are possible post-creation by modifying T, R, C, SR, and d0 directly
    // continuous state mode: expose a continuous observation vector instead of a discrete state index
    bool continuous_state_mode;
    int continuous_state_dim;
    void (*get_state_representation)(MDP* env, int state, float* obs_out);
    void *extras; // for any future extensions.
    void (*free_extras)(MDP* env);
} GridWorldConfig;

typedef struct GridWorldState {
    int state;
} GridWorldState;

// Parameter struct for configuring a GridWorld before creation.
// Initialise with gridworld_params(), then set enter_rewards, absorbing, etc.
typedef struct GridWorldParams {
    int nrows;
    int ncols;
    const unsigned char* grid;   // row-major string, exactly nrows * ncols chars
    int max_steps;
    double enter_rewards[256];
    double leave_rewards[256];
    double enter_costs[256];
    double leave_costs[256];
    double enter_secondary_rewards[256];
    double leave_secondary_rewards[256];
    double step_reward;          // added to reward on every non-absorbing step (default 0)
    double step_cost;            // added to cost on every non-absorbing step (default 0)
    double step_secondary_reward;// added to secondary reward on every non-absorbing step (default 0)
    double noop_prob[256];
    double slip_prob[256];
    bool absorbing[256];
    bool self_loops_accumulate_rewards;             // true by default. If false, self-transitions yield 0 primary reward
    bool self_loops_accumulate_costs;               // true by default. If false, self-transitions yield 0 cost
    bool self_loops_accumulate_secondary_rewards;   // true by default. If false, self-transitions yield 0 secondary reward
    // continuous state mode (set by gridworld_continuous_create)
    int continuous_state_dim;                        // dimensionality of the observation vector
    void (*get_state_representation)(MDP* env, int state, float* obs_out); // NULL → default 2D normalised (row, col)
} GridWorldParams;

// Initialise a params struct with all-zero defaults.
GridWorldParams gridworld_params(int nrows, int ncols, const unsigned char* grid, int max_steps);

// Create a GridWorld MDP from params (precomputes T, R, d0).
MDP* gridworld_create(GridWorldParams* params);

// Create a GridWorld MDP with continuous (box) observation space.
// state_dim: dimensionality of the observation vector.
// get_state_repr: maps (env, discrete_state) → obs_out[state_dim]. NULL → default 2D normalised (row, col).
MDP* gridworld_continuous_create(GridWorldParams* params, int state_dim,
                                 void (*get_state_repr)(MDP* env, int state, float* obs_out));

// Print the grid (marks agent position with @).
void gridworld_print(MDP* env);

// Print value function overlaid on the grid.
void gridworld_print_values(MDP* env, Tensor* V);

// Print greedy policy arrows overlaid on the grid.
void gridworld_print_policy(MDP* env, Tensor* Q);


// exposed callback functions, so that another 'wrapper' environment can make customize these.

void gridworld_reset(MDP* env);
void gridworld_step(MDP* env, void* action);
void gridworld_free_config(MDP* env);
void gridworld_free_state(MDP* env);

// ---- SVG rendering ----

// Per-tile visual: display symbol + background colour for SVG rendering.
typedef struct {
    const char* symbol;             // UTF-8 display string (e.g. emoji). NULL → raw tile char.
    unsigned char bg_r, bg_g, bg_b; // RGB background colour
} GW_TileVisual;

// Render gridworld grid + optional policy arrow overlay as an SVG image file.
// policy:       if non-NULL, overlays greedy-action arrows on non-absorbing/non-obstacle tiles.
// tile_visuals: 256-entry lookup indexed by tile char. NULL → raw ASCII char on white.
// Best viewed in a web browser for proper emoji rendering.
void gridworld_render_svg(MDP* env, Policy* policy,
                          const GW_TileVisual tile_visuals[256],
                          const char* filename);

// --- Built-in environments ---
MDP* cliffwalk_create(void);
MDP* cliffwalk_costly_create(void);
MDP* cliffwalk_continuous_create(void);
MDP* cliffwalk_costly_continuous_create(void);

#endif // GRIDWORLD_H
