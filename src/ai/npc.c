#include "ai.h"
#include "map.h"
#include "math.h"
#include "utils.h"
#include "logging.h"
#include "sim.h"
#include <stdio.h>
#include <stdlib.h>

// #define BENCHMARK_NPC_DECISIONS

#ifdef BENCHMARK_NPC_DECISIONS
#ifdef _WIN32
#include <windows.h>
static double _bnpc_get_time_us(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double _bnpc_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}
#endif
static double _bnpc_npc_logic_us = 0;
static double _bnpc_configure_us = 0;
static double _bnpc_control_car_us = 0;
static int _bnpc_call_count = 0;
#define BNPC_INTERVAL 500000
#endif

void npc_car_make_decisions(Car* self, Simulation* sim) {
    #ifdef BENCHMARK_NPC_DECISIONS
    double _bnpc_t0 = _bnpc_get_time_us();
    #endif
    Seconds dt = sim_get_dt(sim);
    SituationalAwareness* situation = sim_get_situational_awareness(sim, self->id);
    DrivingAssistant* das = sim_get_driving_assistant(sim, self->id);

    CarIndicator turn_indicator;
    CarIndicator lane_change_indicator;

    double r1 = rand_0_to_1();
    double r2 = rand_0_to_1();
    double r3 = rand_0_to_1();
    CarIndicator rand_turn = turn_sample_possible(situation);
    CarIndicator rand_lane_change = lane_change_sample_possible(situation);

    turn_indicator = (r1 < dt / 4) ? rand_turn : car_get_indicator_turn(self);
    if (!situation->is_turn_possible[turn_indicator]) {
        turn_indicator = rand_turn;
    }

    CarIndicator current_lane_indicator = car_get_indicator_lane(self);
    if (current_lane_indicator == INDICATOR_NONE) {
        lane_change_indicator = (r2 < dt / 5) ? rand_lane_change : INDICATOR_NONE;  // about every 5 seconds we have been travelling straight without indicating, decide to change lanes randomly
    } else {
        lane_change_indicator = (r2 < dt / 5) ? INDICATOR_NONE : current_lane_indicator; // about every 5 seconds we have been indicating unsuccessfully, decide to cancel the lane change indicator
    }

    // make sure turn indicator and lane change indicator are compatible
    if (turn_indicator == INDICATOR_LEFT && lane_change_indicator == INDICATOR_RIGHT) {
        lane_change_indicator = INDICATOR_NONE;
    } else if (turn_indicator == INDICATOR_RIGHT && lane_change_indicator == INDICATOR_LEFT) {
        lane_change_indicator = INDICATOR_NONE;
    }

    // If approaching a T-junction while on highway, such that we can exit but want to go straight, move to another lane to stay on highway without having to wait behind exiters or decide to exit with them with 50% probability. (This probability is in addition to the random chance of already exiting due to turn sampling above).
    if (situation->is_an_intersection_upcoming && situation->intersection->is_T_junction && situation->is_T_junction_exit_available && situation->road->num_lanes >= 3 && turn_indicator == INDICATOR_NONE) {
        if (r3 < 0.5) {
            // exit the highway
            turn_indicator = situation->is_T_junction_exit_left_available ? INDICATOR_LEFT : INDICATOR_RIGHT;
            lane_change_indicator = INDICATOR_NONE;
        } else {
            // stay on highway but move away from exit lane
            turn_indicator = INDICATOR_NONE;
            lane_change_indicator = situation->is_lane_change_left_possible ? INDICATOR_LEFT : INDICATOR_RIGHT;
        }
    }

    // on entry/exit ramp based highway, similar logic:
    if (situation->is_exit_available_eventually && situation->road->num_lanes >= 3 && lane_change_indicator == INDICATOR_NONE) {
        if (r3 < 0.5) {
            // exit the highway
            lane_change_indicator = INDICATOR_RIGHT;
            turn_indicator = INDICATOR_NONE;
        } else {
            // stay on highway but move away from exit lane
            turn_indicator = INDICATOR_NONE;
            lane_change_indicator = situation->is_lane_change_left_possible ? INDICATOR_LEFT : INDICATOR_RIGHT;
        }
    }

    // on entry ramp, try to move to leftmost lane
    if (situation->is_on_entry_ramp && lane_change_indicator == INDICATOR_NONE) {
        lane_change_indicator = INDICATOR_LEFT;
        turn_indicator = INDICATOR_NONE;
    }

    driving_assistant_configure_turn_intent(das, self, sim, turn_indicator);
    driving_assistant_configure_merge_intent(das, self, sim, lane_change_indicator);
    #ifdef BENCHMARK_NPC_DECISIONS
    double _bnpc_t1 = _bnpc_get_time_us();
    #endif

    driving_assistant_control_car(das, self, sim);
    #ifdef BENCHMARK_NPC_DECISIONS
    double _bnpc_t2 = _bnpc_get_time_us();
    _bnpc_npc_logic_us += _bnpc_t1 - _bnpc_t0;
    _bnpc_control_car_us += _bnpc_t2 - _bnpc_t1;
    _bnpc_call_count++;
    if (_bnpc_call_count % BNPC_INTERVAL == 0) {
        double n = (double)_bnpc_call_count;
        printf("[BENCH npc_decisions] Avg over %d calls: NpcLogic+Configure=%.3f us | ControlCar=%.3f us | Total=%.3f us\n",
            _bnpc_call_count,
            _bnpc_npc_logic_us / n,
            _bnpc_control_car_us / n,
            (_bnpc_npc_logic_us + _bnpc_control_car_us) / n);
    }
    #endif
}
