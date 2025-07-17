#include <stdlib.h>
#include "ai.h"
#include "procedures.h"
#include "logging.h"


// --- MERGE procedure ---

ProcedureStatusCode procedure_merge_init(Simulation* sim, Car* car, Procedure* procedure, const double* args) {

    // Read arguments
    Seconds timeout_duration = args[0];
    CarIndictor ind = (CarIndictor) args[1];
    MetersPerSecond cruise_speed_desired = args[2];

    
    // validate arguments
    if (timeout_duration <= 0) {
        LOG_ERROR("Car %d: Invalid timeout duration for merge procedure: %.2f seconds.", car->id, timeout_duration);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR;
    }
    if (ind != INDICATOR_LEFT && ind != INDICATOR_RIGHT) {
        if (ind == INDICATOR_NONE) {
            LOG_WARN("Car %d: No merge requested, procedure not necessary.", car->id);
            return PROCEDURE_STATUS_INIT_FAILED_REASON_UNNECESSARY; // No merge requested
        } else {
            LOG_ERROR("Car %d: Invalid merge direction: %d. Expected INDICATOR_LEFT or INDICATOR_RIGHT.", car->id, ind);
            return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR; // Invalid direction
        }
    }

    // validate possibility
    SituationalAwareness* situation = sim_get_situational_awareness(sim, car->id);
    const Lane* target_lane = situation->lane_target_for_indicator[ind];
    if (!target_lane) {
        return PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE;
    }

    // All looks good, store state variables
    procedure->state[0] = timeout_duration;
    procedure->state[1] = ind;
    procedure->state[2] = cruise_speed_desired;
    procedure->state[3] = target_lane->id;
    procedure->state[4] = sim_get_time(sim);    // record start time

    return PROCEDURE_STATUS_INITIALIZED;
}


static void _adaptive_cruise_control(Car* car, Simulation* sim, const SituationalAwareness* s, MetersPerSecond cruise_speed_desired) {
    // TODO: ai.h will provide a function to compute adaptive cruise control speed based on the leading car's speed and distance. This is a dirty implementation for now.
    // TODO: check for other cars in the lane and adjust speed accordingly
    car_set_acceleration(car, car_compute_acceleration_cruise(car, cruise_speed_desired));
}

ProcedureStatusCode procedure_merge_step(Simulation* sim, Car* car, Procedure* procedure) {

    SituationalAwareness* situation = sim_get_situational_awareness(sim, car->id);

    // extract state
    Seconds timeout_duration = procedure->state[0];
    CarIndictor ind = (CarIndictor) procedure->state[1];
    MetersPerSecond cruise_speed_desired = procedure->state[2];
    LaneId target_lane_id = (LaneId) procedure->state[3];
    double start_time = procedure->state[4];

    // check success
    if (situation->lane->id == target_lane_id) {
        car_set_indicator_lane(car, INDICATOR_NONE);    // turn off indicator
        car_set_request_indicated_lane(car, false);     // reset request
        _adaptive_cruise_control(car, sim, situation, cruise_speed_desired); // maintain cruise
        return PROCEDURE_STATUS_COMPLETED;
    }

    // check timeout
    double now = sim_get_time(sim);
    if (now - start_time > timeout_duration) {
        LOG_WARN("Car %d: merge procedure timed out after %.2f seconds.", car->id, timeout_duration);
        car_set_indicator_lane(car, INDICATOR_NONE);    // turn off indicator
        car_set_request_indicated_lane(car, false);     // reset request
        _adaptive_cruise_control(car, sim, situation, cruise_speed_desired);  // maintain cruise
        return PROCEDURE_STATUS_FAILED_REASON_TIMEDOUT;
    }

    // check if merge is still possible
    const Lane* target_lane_as_of_now = situation->lane_target_for_indicator[ind];
    if (!target_lane_as_of_now) {
        LOG_WARN("Car %d: merge procedure failed because target lane %d is no longer available.", car->id, target_lane_id);
        car_set_indicator_lane(car, INDICATOR_NONE);    // turn off indicator
        car_set_request_indicated_lane(car, false);     // reset request
        _adaptive_cruise_control(car, sim, situation, cruise_speed_desired); // maintain cruise
        return PROCEDURE_STATUS_FAILED_REASON_NOW_IMPOSSIBLE;
    } else {
        // update target lane id in case it has changed
        target_lane_id = target_lane_as_of_now->id;
        procedure->state[3] = target_lane_id; // update state with new target lane id
    }

    // check safety
    if (car_is_lane_change_dangerous(car, sim, situation, ind)) {
        LOG_TRACE("Car %d: merge procedure is dangerous, waiting for safe conditions.", car->id);
        car_set_indicator_lane(car, ind);  // keep signaling intent
        car_set_request_indicated_lane(car, false); // don't request lane change yet
        _adaptive_cruise_control(car, sim, situation, cruise_speed_desired); // maintain cruise
        return PROCEDURE_STATUS_IN_PROGRESS; // still waiting for safe conditions
    }

    // safe, request the sim to merge
    car_set_indicator_lane(car, ind);  // signal intent to merge
    car_set_request_indicated_lane(car, true); // request lane change
    _adaptive_cruise_control(car, sim, situation, cruise_speed_desired); // maintain cruise speed
    LOG_TRACE("Car %d: merge procedure in progress, requesting merge into lane %d", car->id, target_lane_id);
    return PROCEDURE_STATUS_IN_PROGRESS; // still in progress
}

void procedure_merge_cancel(Simulation* sim, Car* car, Procedure* procedure) {
    LOG_INFO("Cancelling merge procedure for car %d.", car->id);
    car_set_indicator_lane(car, INDICATOR_NONE);    // turn off indicator
    car_set_request_indicated_lane(car, false);     // reset request
    car_set_acceleration(car, 0);                   // stop acceleration
    return;
}
