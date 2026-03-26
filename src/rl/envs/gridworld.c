#include "rl/envs/gridworld.h"
#include <string.h>
#include <stdio.h>
#include <errno.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

// ---- Action tables (0-indexed: N=0, E=1, S=2, W=3, NOOP=4) ----

static const int ACTION_DR[] = {-1,  0,  1,  0};  // row delta
static const int ACTION_DC[] = { 0,  1,  0, -1};  // col delta
// static const int OPPOSITE[]  = {2, 3, 0, 1};  // opposite action (unused — always prob 0)
static const int LEFT_90[]   = {3, 0, 1, 2};  // 90-degree left
static const int RIGHT_90[]  = {1, 2, 3, 0};  // 90-degree right

// ---- Coordinate helpers (row-major, 0-indexed) ----

static inline int rc_to_state(int r, int c, int ncols) {
    return r * ncols + c;
}

static inline void state_to_rc(int s, int ncols, int* r, int* c) {
    *r = s / ncols;
    *c = s % ncols;
}

// ---- Default continuous state representation: normalised (row, col) ----

static void gw_default_state_repr(MDP* env, int state, float* obs_out) {
    GridWorldConfig* cfg = (GridWorldConfig*)env->config;
    int r, c;
    state_to_rc(state, cfg->ncols, &r, &c);
    obs_out[0] = (cfg->nrows > 1) ? (float)r / (cfg->nrows - 1) : 0.0f;
    obs_out[1] = (cfg->ncols > 1) ? (float)c / (cfg->ncols - 1) : 0.0f;
}

// Returns the state reached by moving in direction a from s, or -1 if out of bounds.
static int hypothetical_next(int s, int a, int nrows, int ncols) {
    if (a == GW_NOOP) return s;
    int r, c;
    state_to_rc(s, ncols, &r, &c);
    int nr = r + ACTION_DR[a];
    int nc = c + ACTION_DC[a];
    if (nr >= 0 && nr < nrows && nc >= 0 && nc < ncols)
        return rc_to_state(nr, nc, ncols);
    return -1;
}

static bool is_ortho_neighbor(int s1, int s2, int ncols) {
    int r1, c1, r2, c2;
    state_to_rc(s1, ncols, &r1, &c1);
    state_to_rc(s2, ncols, &r2, &c2);
    int dr = r1 - r2; if (dr < 0) dr = -dr;
    int dc = c1 - c2; if (dc < 0) dc = -dc;
    return (dr + dc) <= 1;  // includes self
}

// ---- Transition probability (computed from config) ----

static double gw_compute_transition_prob(GridWorldConfig* cfg, int s, int a, int sp) {
    unsigned char tile_s  = cfg->grid[s];
    unsigned char tile_sp = cfg->grid[sp];

    // Absorbing or obstacle: deterministic self-loop
    if (cfg->absorbing[tile_s] || tile_s == 'O')
        return (sp == s) ? 1.0 : 0.0;

    // Can't enter an obstacle
    if (tile_sp == 'O')
        return 0.0;

    // Must be orthogonal neighbour (or self)
    if (!is_ortho_neighbor(s, sp, cfg->ncols))
        return 0.0;

    // Explicit NOOP action is deterministic self-loop.
    if (a == GW_NOOP)
        return (sp == s) ? 1.0 : 0.0;

    // Directional action
    int move  = hypothetical_next(s, a,             cfg->nrows, cfg->ncols);
    int left  = hypothetical_next(s, LEFT_90[a],    cfg->nrows, cfg->ncols);
    int right = hypothetical_next(s, RIGHT_90[a],   cfg->nrows, cfg->ncols);

    double p_noop = cfg->noop_prob[tile_s];
    double p_slip = cfg->slip_prob[tile_s];
    double p_move = 1.0 - p_noop - p_slip;

    bool move_blocked  = (move  < 0 || cfg->grid[move]  == 'O');
    bool left_blocked  = (left  < 0 || cfg->grid[left]  == 'O');
    bool right_blocked = (right < 0 || cfg->grid[right] == 'O');

    if (sp == s) {
        // Staying in place: p_noop + bounced probabilities
        double p = p_noop;
        if (move_blocked)  p += p_move;
        if (left_blocked)  p += p_slip / 2.0;
        if (right_blocked) p += p_slip / 2.0;
        return p;
    }
    if (move >= 0 && sp == move && !move_blocked)
        return p_move;
    if (left >= 0 && sp == left && !left_blocked)
        return p_slip / 2.0;
    if (right >= 0 && sp == right && !right_blocked)
        return p_slip / 2.0;
    // Opposite direction or non-neighbour: 0
    return 0.0;
}

// ---- Reward (computed from config) ----

static double gw_compute_reward(GridWorldConfig* cfg, int s, int a, int sp) {
    (void)a;
    unsigned char tile_s = cfg->grid[s];
    if (cfg->absorbing[tile_s])
        return 0.0;  // no reward from absorbing states
    double reward = cfg->step_reward;
    if (sp == s && !cfg->self_loops_accumulate_rewards)
        return reward;
    unsigned char tile_sp = cfg->grid[sp];
    return reward + cfg->leave_rewards[tile_s] + cfg->enter_rewards[tile_sp];
}

static double gw_compute_cost(GridWorldConfig* cfg, int s, int a, int sp) {
    (void)a;
    unsigned char tile_s = cfg->grid[s];
    if (cfg->absorbing[tile_s])
        return 0.0;  // no cost from absorbing states
    double cost = cfg->step_cost;
    if (sp == s && !cfg->self_loops_accumulate_costs)
        return cost;
    unsigned char tile_sp = cfg->grid[sp];
    return cost + cfg->leave_costs[tile_s] + cfg->enter_costs[tile_sp];
}

static double gw_compute_secondary_reward(GridWorldConfig* cfg, int s, int a, int sp) {
    (void)a;
    unsigned char tile_s = cfg->grid[s];
    if (cfg->absorbing[tile_s])
        return 0.0;  // no secondary reward from absorbing states
    double sr = cfg->step_secondary_reward;
    if (sp == s && !cfg->self_loops_accumulate_secondary_rewards)
        return sr;  // self-transitions: no enter/leave component unless enabled
    unsigned char tile_sp = cfg->grid[sp];
    return sr + cfg->leave_secondary_rewards[tile_s] + cfg->enter_secondary_rewards[tile_sp];
}

// ---- MDP callbacks ----

static float gw_transition_probability(MDP* env, int s, int a, int sp) {
    return (float)gw_compute_transition_prob((GridWorldConfig*)env->config, s, a, sp);
}

static float gw_reward_function(MDP* env, int s, int a, int sp) {
    return (float)gw_compute_reward((GridWorldConfig*)env->config, s, a, sp);
}

static float gw_cost_function(MDP* env, int s, int a, int sp) {
    return (float)gw_compute_cost((GridWorldConfig*)env->config, s, a, sp);
}

static float gw_secondary_reward_function(MDP* env, int s, int a, int sp) {
    return (float)gw_compute_secondary_reward((GridWorldConfig*)env->config, s, a, sp);
}

static float gw_start_state_probability(MDP* env, int s) {
    return tensor1d_get_at(((GridWorldConfig*)env->config)->d0, s);
}

void gridworld_reset(MDP* env) {
    GridWorldConfig* cfg = (GridWorldConfig*)env->config;
    GridWorldState* st = (GridWorldState*)env->state;

    double r = rand_uniform(0.0, 1.0);
    double cumsum = 0;
    st->state = 0;
    for (int s = 0; s < cfg->num_states; s++) {
        cumsum += tensor1d_get_at(cfg->d0, s);
        if (r <= cumsum) { st->state = s; break; }
    }
    *(int*)env->observation = st->state;
    env->reward = 0;
    env->terminated = false;
    env->truncated = false;
    env->step_count = 0;
    if (cfg->continuous_state_mode)
        cfg->get_state_representation(env, st->state, (float*)env->observation);
}

void gridworld_step(MDP* env, void* action) {
    GridWorldConfig* cfg = (GridWorldConfig*)env->config;
    GridWorldState* st = (GridWorldState*)env->state;
    int a = *(int*)action;
    int s = st->state;

    // Sample s' from T[s, a, :]
    double r = rand_uniform(0.0, 1.0);
    double cumsum = 0;
    int s_next = s;  // fallback: self-loop
    for (int sp = 0; sp < cfg->num_states; sp++) {
        cumsum += tensor3d_get_at(cfg->T, s, a, sp);
        if (r <= cumsum) { s_next = sp; break; }
    }

    env->reward = tensor3d_get_at(cfg->R, s, a, s_next);
    env->cost = tensor3d_get_at(cfg->C, s, a, s_next);
    env->secondary_reward = tensor3d_get_at(cfg->SR, s, a, s_next);
    st->state = s_next;
    *(int*)env->observation = s_next;
    env->step_count++;
    env->terminated = cfg->absorbing[cfg->grid[s_next]];
    env->truncated = (!env->terminated && cfg->max_steps > 0 && env->step_count >= cfg->max_steps);
    if (cfg->continuous_state_mode)
        cfg->get_state_representation(env, s_next, (float*)env->observation);
}

void gridworld_free_config(MDP* env) {
    GridWorldConfig* cfg = (GridWorldConfig*)env->config;
    if (cfg) {
        if (cfg->free_extras) {
            cfg->free_extras(env);  // free any extras first, since they might reference other config fields
        }
        free(cfg->grid);
        tensor_free(cfg->T);
        tensor_free(cfg->R);
        tensor_free(cfg->C);
        tensor_free(cfg->SR);
        tensor_free(cfg->d0);
        free(cfg);
        env->config = NULL;
    }
}

void gridworld_free_state(MDP* env) {
    free(env->state);
    env->state = NULL;
}

// Ensure all parent directories for `path` exist.
// Supports both '/' and '\\' separators.
static void ensure_parent_dirs(const char* path) {
    if (!path || !*path) return;

    char buf[1024];
    size_t n = strlen(path);
    if (n >= sizeof(buf)) return;
    memcpy(buf, path, n + 1);

    size_t start = 0;
#ifdef _WIN32
    if (n >= 2 && buf[1] == ':') {
        start = 2;
    }
#endif

    for (size_t i = start; i < n; i++) {
        if (buf[i] != '/' && buf[i] != '\\') continue;

        char saved = buf[i];
        buf[i] = '\0';
        if (buf[0] != '\0') {
#ifdef _WIN32
            if (_mkdir(buf) != 0 && errno != EEXIST) {
                buf[i] = saved;
                return;
            }
#else
            if (mkdir(buf, 0777) != 0 && errno != EEXIST) {
                buf[i] = saved;
                return;
            }
#endif
        }
        buf[i] = saved;
    }
}

// ===========================================================================
// Public API
// ===========================================================================

GridWorldParams gridworld_params(int nrows, int ncols, const unsigned char* grid, int max_steps) {
    GridWorldParams p;
    memset(&p, 0, sizeof(p));
    p.nrows = nrows;
    p.ncols = ncols;
    p.grid = grid;
    p.max_steps = max_steps;
    p.self_loops_accumulate_rewards = true;
    p.self_loops_accumulate_costs = true;
    p.self_loops_accumulate_secondary_rewards = true;
    return p;
}

MDP* gridworld_create(GridWorldParams* params) {
    int nrows = params->nrows;
    int ncols = params->ncols;
    int num_states = nrows * ncols;
    int num_actions = GW_NUM_ACTIONS;

    GridWorldConfig* cfg = (GridWorldConfig*)malloc(sizeof(GridWorldConfig));
    cfg->nrows = nrows;
    cfg->ncols = ncols;
    cfg->num_states = num_states;
    cfg->max_steps = params->max_steps;

    // Copy grid
    cfg->grid = (unsigned char*)malloc(num_states);
    memcpy(cfg->grid, params->grid, num_states);

    // Copy per-tile parameters
    memcpy(cfg->enter_rewards, params->enter_rewards, sizeof(cfg->enter_rewards));
    memcpy(cfg->leave_rewards, params->leave_rewards, sizeof(cfg->leave_rewards));
    memcpy(cfg->enter_costs,   params->enter_costs,   sizeof(cfg->enter_costs));
    memcpy(cfg->leave_costs,   params->leave_costs,   sizeof(cfg->leave_costs));
    memcpy(cfg->enter_secondary_rewards, params->enter_secondary_rewards, sizeof(cfg->enter_secondary_rewards));
    memcpy(cfg->leave_secondary_rewards, params->leave_secondary_rewards, sizeof(cfg->leave_secondary_rewards));
    cfg->step_reward = params->step_reward;
    cfg->step_cost = params->step_cost;
    cfg->step_secondary_reward = params->step_secondary_reward;
    memcpy(cfg->noop_prob,     params->noop_prob,     sizeof(cfg->noop_prob));
    memcpy(cfg->slip_prob,     params->slip_prob,     sizeof(cfg->slip_prob));
    memcpy(cfg->absorbing,     params->absorbing,     sizeof(cfg->absorbing));
    cfg->self_loops_accumulate_rewards           = params->self_loops_accumulate_rewards;
    cfg->self_loops_accumulate_costs             = params->self_loops_accumulate_costs;
    cfg->self_loops_accumulate_secondary_rewards = params->self_loops_accumulate_secondary_rewards;
    cfg->continuous_state_mode = false;
    cfg->continuous_state_dim = 0;
    cfg->get_state_representation = NULL;
    cfg->extras = NULL;
    cfg->free_extras = NULL;

    // Compute d0: uniform over 'S' tiles
    cfg->d0 = tensor1d_malloc(num_states);
    int num_start = 0;
    for (int s = 0; s < num_states; s++)
        if (cfg->grid[s] == 'S') num_start++;
    for (int s = 0; s < num_states; s++)
        tensor1d_set_at(cfg->d0, s, (cfg->grid[s] == 'S' && num_start > 0)
                                      ? (float)(1.0 / num_start) : 0.0f);

    // Precompute T, R, and C
    cfg->T = tensor3d_malloc(num_states, num_actions, num_states);
    cfg->R = tensor3d_malloc(num_states, num_actions, num_states);
    cfg->C = tensor3d_malloc(num_states, num_actions, num_states);
    cfg->SR = tensor3d_malloc(num_states, num_actions, num_states);
    for (int s = 0; s < num_states; s++) {
        for (int a = 0; a < num_actions; a++) {
            for (int sp = 0; sp < num_states; sp++) {
                tensor3d_set_at(cfg->T, s, a, sp,
                    (float)gw_compute_transition_prob(cfg, s, a, sp));
                tensor3d_set_at(cfg->R, s, a, sp,
                    (float)gw_compute_reward(cfg, s, a, sp));
                tensor3d_set_at(cfg->C, s, a, sp,
                    (float)gw_compute_cost(cfg, s, a, sp));
                tensor3d_set_at(cfg->SR, s, a, sp,
                    (float)gw_compute_secondary_reward(cfg, s, a, sp));
            }
        }
    }

    // Build MDP
    Space state_space  = { .type = SPACE_TYPE_DISCRETE, .n = num_states,  .lows = NULL, .highs = NULL };
    Space action_space = { .type = SPACE_TYPE_DISCRETE, .n = num_actions, .lows = NULL, .highs = NULL };
    GridWorldState* state = (GridWorldState*)malloc(sizeof(GridWorldState));
    state->state = -1;

    MDP* env = mdp_malloc("GridWorld", state_space, action_space, cfg, state,
                          gridworld_reset, gridworld_step, gridworld_free_config, gridworld_free_state);
    env->transition_probability  = gw_transition_probability;
    env->reward_function         = gw_reward_function;
    env->cost_function           = gw_cost_function;
    env->secondary_reward_function = gw_secondary_reward_function;
    env->start_state_probability = gw_start_state_probability;
    return env;
}

// ---- Printing utilities ----

void gridworld_print(MDP* env) {
    GridWorldConfig* cfg = (GridWorldConfig*)env->config;
    GridWorldState* st = (GridWorldState*)env->state;
    printf("GridWorld (%d x %d):\n", cfg->nrows, cfg->ncols);
    for (int r = 0; r < cfg->nrows; r++) {
        printf("  ");
        for (int c = 0; c < cfg->ncols; c++) {
            int s = rc_to_state(r, c, cfg->ncols);
            printf("%c", (s == st->state) ? '@' : cfg->grid[s]);
        }
        printf("\n");
    }
}

void gridworld_print_values(MDP* env, Tensor* V) {
    GridWorldConfig* cfg = (GridWorldConfig*)env->config;
    printf("Values (%d x %d):\n", cfg->nrows, cfg->ncols);
    for (int r = 0; r < cfg->nrows; r++) {
        printf("  ");
        for (int c = 0; c < cfg->ncols; c++) {
            int s = rc_to_state(r, c, cfg->ncols);
            printf("%7.2f ", (double)tensor1d_get_at(V, s));
        }
        printf("\n");
    }
}

void gridworld_print_policy(MDP* env, Tensor* Q) {
    GridWorldConfig* cfg = (GridWorldConfig*)env->config;
    static const char action_chars[] = {'^', '>', 'v', '<', 'o'};
    printf("Policy (%d x %d):\n", cfg->nrows, cfg->ncols);
    for (int r = 0; r < cfg->nrows; r++) {
        printf("  ");
        for (int c = 0; c < cfg->ncols; c++) {
            int s = rc_to_state(r, c, cfg->ncols);
            unsigned char tile = cfg->grid[s];
            if (cfg->absorbing[tile] || tile == 'O') {
                printf(" %c", cfg->grid[s]);
            } else {
                int best_a = 0;
                float best_q = tensor2d_get_at(Q, s, 0);
                for (int a = 1; a < GW_NUM_ACTIONS; a++) {
                    float q = tensor2d_get_at(Q, s, a);
                    if (q > best_q) { best_q = q; best_a = a; }
                }
                printf(" %c", action_chars[best_a]);
            }
        }
        printf("\n");
    }
}

// ---- SVG rendering ----

void gridworld_render_svg(MDP* env, Policy* policy,
                          const GW_TileVisual tile_visuals[256],
                          const char* filename)
{
    GridWorldConfig* cfg = (GridWorldConfig*)env->config;
    int nrows = cfg->nrows;
    int ncols = cfg->ncols;

    // Cell geometry
    const int cell  = 60;   // cell size in px
    const int pad   = 2;    // padding around grid
    int width  = ncols * cell + 2 * pad;
    int height = nrows * cell + 2 * pad;

    ensure_parent_dirs(filename);
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "gridworld_render_svg: cannot open '%s' for writing\n", filename);
        return;
    }

    // SVG header
    fprintf(fp,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        "width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" "
        "font-family=\"Segoe UI Emoji, Apple Color Emoji, Noto Color Emoji, sans-serif\">\n",
        width, height, width, height);

    // White background
    fprintf(fp, "<rect width=\"%d\" height=\"%d\" fill=\"white\"/>\n", width, height);

    // Arrow polygon template (points up in a cell-sized box, then rotated per action).
    // Arrows:  N=0 -> 0°, E=1 -> 90°, S=2 -> 180°, W=3 -> 270°
    // NOOP=4 is rendered as a circle.
    static const int action_rotation[] = {0, 90, 180, 270};

    // ---- Draw cells ----
    for (int r = 0; r < nrows; r++) {
        for (int c = 0; c < ncols; c++) {
            int s = r * ncols + c;
            unsigned char tile = cfg->grid[s];
            int x = pad + c * cell;
            int y = pad + r * cell;
            int cx = x + cell / 2;
            int cy = y + cell / 2;

            // Background colour
            unsigned char bg_r = 255, bg_g = 255, bg_b = 255;
            const char* sym = NULL;
            if (tile_visuals) {
                bg_r = tile_visuals[tile].bg_r;
                bg_g = tile_visuals[tile].bg_g;
                bg_b = tile_visuals[tile].bg_b;
                sym  = tile_visuals[tile].symbol;
            }

            // Stochastic-policy highlight: bold black border when >1 action has non-trivial probability.
            const char* border_color = "#666";
            double border_width = 0.5;
            if (policy && !(cfg->absorbing[tile] || tile == 'O')) {
                int nonzero_actions = 0;
                for (int a = 0; a < GW_NUM_ACTIONS; a++) {
                    float p = policy->action_probability(policy, s, a);
                    if (p > 1e-4f) nonzero_actions++;
                }
                if (nonzero_actions > 1) {
                    border_color = "#000";
                    border_width = 2.5;
                }
            }

            fprintf(fp, "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" "
                    "fill=\"rgb(%d,%d,%d)\" stroke=\"%s\" stroke-width=\"%.2f\"/>\n",
                    x, y, cell, cell, bg_r, bg_g, bg_b, border_color, border_width);

            // Tile symbol (emoji or ASCII char)
            if (sym) {
                fprintf(fp, "<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
                        "dominant-baseline=\"central\" font-size=\"%d\">%s</text>\n",
                        cx, cy, cell * 55 / 100, sym);
            } else {
                fprintf(fp, "<text x=\"%d\" y=\"%d\" text-anchor=\"middle\" "
                        "dominant-baseline=\"central\" font-size=\"%d\" "
                        "fill=\"#333\">%c</text>\n",
                        cx, cy, cell * 45 / 100, tile);
            }

            // Policy arrow overlay (all actions, size proportional to probability)
            if (policy && !(cfg->absorbing[tile] || tile == 'O')) {
                const int edge_margin = cell * 8 / 100;   // keep arrow tip away from border
                const int offset_mag  = cell * 22 / 100;  // push arrows toward their own edge
                const int max_ar = cell / 2 - edge_margin - offset_mag;
                for (int a = 0; a < GW_NUM_ACTIONS; a++) {
                    float p = policy->action_probability(policy, s, a);
                    if (p <= 0.0001f) continue;

                    if (a == GW_NOOP) {
                        int rr = (int)(cell * 0.22f * p); // NOOP circle radius proportional to probability
                        if (rr < 2) continue;
                        double opacity = 0.5;
                        fprintf(fp, "<circle cx=\"%d\" cy=\"%d\" r=\"%d\" "
                            "fill=\"black\" fill-opacity=\"%.3f\" stroke=\"none\"/>\n",
                            cx, cy, rr, opacity);
                        continue;
                    }

                    int rot = action_rotation[a];
                    int ar = (int)(cell * 0.5f * p);  // p=1.0 -> larger arrow, smaller for lower-probability actions
                    if (ar > max_ar) ar = max_ar;
                    if (ar < 3) ar = 3;

                    // Arrow points: upward triangle (before rotation): tip at (0,-ar), base at (-ar/2, ar/2), (ar/2, ar/2)
                    // Offset each arrow toward its own direction (N/E/S/W), but keep a border margin.
                    int ox = cx;
                    int oy = cy;
                    if (a == 0) oy -= offset_mag;      // N
                    else if (a == 1) ox += offset_mag; // E
                    else if (a == 2) oy += offset_mag; // S
                    else ox -= offset_mag;             // W

                    double opacity = 0.4;
                    fprintf(fp, "<polygon points=\"0,%d %d,%d %d,%d\" "
                            "transform=\"translate(%d,%d) rotate(%d)\" "
                            "fill=\"black\" fill-opacity=\"%.3f\" "
                            "stroke=\"white\" stroke-width=\"0.6\"/>\n",
                            -ar, -ar / 2, ar / 2, ar / 2, ar / 2,
                            ox, oy, rot, opacity);
                }
            }
        }
    }

    fprintf(fp, "</svg>\n");
    fclose(fp);
}

// ---- Built-in environments ----

MDP* cliffwalk_create(void) {
    const unsigned char grid[] =
        "            "   // row 0: 12 free spaces
        "            "   // row 1
        "            "   // row 2
        "SCCCCCCCCCCG";  // row 3: start, 10 cliffs, goal
    GridWorldParams p = gridworld_params(4, 12, grid, 200);
    p.enter_rewards['S'] = -1.0;
    p.enter_rewards[' '] = -1.0;
    p.enter_rewards['G'] = -1.0;
    p.enter_rewards['C'] = -100.0;
    p.absorbing['C'] = true;
    p.absorbing['G'] = true;
    return gridworld_create(&p);
}

MDP* cliffwalk_costly_create(void) {
    //  Grid layout (4 rows x 12 cols):
    //  Row 0:  "     $      "   1 toll tile
    //  Row 1:  "    $$      "   2 toll tiles
    //  Row 2:  "   $$$      "   3 toll tiles (above cliff)
    //  Row 3:  "SCCCCCCCCCCG"   start, 10 cliffs, goal
    const unsigned char grid[] =
        "     $      "
        "    $$      "
        "   $$$      "
        "SCCCCCCCCCCG";
    GridWorldParams p = gridworld_params(4, 12, grid, 200);
    p.enter_rewards['S'] = -1.0;
    p.enter_rewards[' '] = -1.0;
    p.enter_rewards['G'] = -1.0;
    p.enter_rewards['$'] = -1.0;   // same step penalty as normal tiles
    p.enter_rewards['C'] = -100.0;
    p.enter_costs['$']   =  1.0;   // toll cost for entering $ tiles
    p.absorbing['C'] = true;
    p.absorbing['G'] = true;
    return gridworld_create(&p);
}

// ---- Continuous state wrappers ----

MDP* gridworld_continuous_create(GridWorldParams* params, int state_dim,
                                 void (*get_state_repr)(MDP* env, int state, float* obs_out)) {
    if (!get_state_repr) {
        get_state_repr = gw_default_state_repr;
        state_dim = 2;
    }
    // Store continuous params so gridworld_create can read them back via config
    params->continuous_state_dim = state_dim;
    params->get_state_representation = get_state_repr;

    MDP* env = gridworld_create(params);
    GridWorldConfig* cfg = (GridWorldConfig*)env->config;
    cfg->continuous_state_mode = true;
    cfg->continuous_state_dim = state_dim;
    cfg->get_state_representation = get_state_repr;

    // Switch state space from discrete to box
    env->state_space.type = SPACE_TYPE_BOX;
    env->state_space.n = state_dim;
    env->state_space.lows  = (double*)calloc(state_dim, sizeof(double));
    env->state_space.highs = (double*)malloc(state_dim * sizeof(double));
    for (int i = 0; i < state_dim; i++) env->state_space.highs[i] = 1.0;

    // Re-allocate observation buffer for continuous vector
    free(env->observation);
    env->observation = malloc(sizeof(float) * state_dim);

    return env;
}

MDP* cliffwalk_continuous_create(void) {
    const unsigned char grid[] =
        "            "
        "            "
        "            "
        "SCCCCCCCCCCG";
    GridWorldParams p = gridworld_params(4, 12, grid, 200);
    p.enter_rewards['S'] = -1.0;
    p.enter_rewards[' '] = -1.0;
    p.enter_rewards['G'] = -1.0;
    p.enter_rewards['C'] = -100.0;
    p.absorbing['C'] = true;
    p.absorbing['G'] = true;
    return gridworld_continuous_create(&p, 0, NULL);
}

MDP* cliffwalk_costly_continuous_create(void) {
    const unsigned char grid[] =
        "     $      "
        "    $$      "
        "   $$$      "
        "SCCCCCCCCCCG";
    GridWorldParams p = gridworld_params(4, 12, grid, 200);
    p.enter_rewards['S'] = -1.0;
    p.enter_rewards[' '] = -1.0;
    p.enter_rewards['G'] = -1.0;
    p.enter_rewards['$'] = -1.0;
    p.enter_rewards['C'] = -100.0;
    p.enter_costs['$']   =  1.0;
    p.absorbing['C'] = true;
    p.absorbing['G'] = true;
    return gridworld_continuous_create(&p, 0, NULL);
}
