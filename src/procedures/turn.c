#include <stdlib.h>
#include "procedures.h"
#include "logging.h"


// --- TURN procedure ---

ProcedureStatusCode procedure_turn_init(Simulation* sim, Car* ego_car, Procedure* procedure, const double* args) {
    // Read arguments
    CarIndicator turn_ind = (CarIndicator) args[0];
    Seconds timeout_duration = args[1];
    bool should_brake = args[2] > 0;
    bool use_preferred_accel_profile = args[3] > 0;

    // validate arguments
    if (timeout_duration <= 0) {
        LOG_ERROR("Car %d: Invalid timeout_duration for pass procedure: %.2f seconds.", ego_car->id, timeout_duration);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR;
    }

    procedure->state[0] = turn_ind;
    procedure->state[1] = sim_get_time(sim);
    procedure->state[2] = timeout_duration;
    procedure->state[3] = should_brake;
    procedure->state[4] = use_preferred_accel_profile;

    // Clear car's control variables
    car_reset_all_control_variables(ego_car);

    LOG_DEBUG("Car %d: Pass procedure initialized. timeout_duration: %.2f seconds, should break: %d, Use Preferred Accel Profile: %d",
              ego_car->id, timeout_duration, should_brake, use_preferred_accel_profile);

    return PROCEDURE_STATUS_INITIALIZED;
}

ProcedureStatusCode procedure_turn_step(Simulation* sim, Car* car, Procedure* procedure) {
    SituationalAwareness* situation = sim_get_situational_awareness(sim, car->id);

    // extract state
    CarIndicator turn_ind = (CarIndicator) procedure->state[0];
    Seconds start_time = procedure->state[1];
    Seconds timeout_duration = procedure->state[2];
    bool should_brake = procedure->state[3] > 0;
    bool use_preferred_acc_profile = procedure->state[4] > 0;

    // Check timeout
    double now = sim_get_time(sim);
    if (now - start_time > timeout_duration) {
        LOG_WARN("Car %d: pass procedure timed out after %.2f seconds.", car->id, timeout_duration);
        car_set_indicator_lane(car, INDICATOR_NONE);    // turn off indicator
        car_set_request_indicated_lane(car, false);     // reset request
        return PROCEDURE_STATUS_FAILED_REASON_TIMEDOUT;
    }

    // check success // TODO: Better success indicator?
    if (!situation->is_on_intersection && !situation->is_an_intersection_upcoming) {
        LOG_DEBUG("Car %d: turn procedure completed successfully into lane.", car->id);
        car_set_indicator_lane(car, INDICATOR_NONE);    // turn off indicator
        car_set_request_indicated_lane(car, false);     // reset request
        return PROCEDURE_STATUS_COMPLETED;
    }

    // Check if turn is even possible
    if (!situation->is_turn_possible[turn_ind]) {
        LOG_ERROR("Car %d: requested turn (%d) impossible from current lane.", car->id, turn_ind);
        return PROCEDURE_STATUS_FAILED_REASON_NOW_IMPOSSIBLE;
    }
    
    // set indicator on and request if not already
    // Turn is possible, so turn indicator on and notify the simulator
    if (car_get_indicator_turn(car) == INDICATOR_NONE) {
        car_set_indicator_turn_and_request(car, turn_ind);
    }

    // Make the turn
    MetersPerSecondSquared a;
    if (should_brake) {
        // Aim to stop smoothly
        double  car_pos = car_get_lane_progress_meters(car);
        Meters  stop_line_dist = situation->distance_to_end_of_lane;
        Meters  target_pos = car_pos + stop_line_dist;
        a = car_compute_acceleration_chase_target(car, target_pos, 0, meters(1.0), 0, use_preferred_acc_profile);
    } else {
        // Cruise with the preferred turn-in speed
        a = car_compute_acceleration_cruise(car, car->preferences.turn_speed, use_preferred_acc_profile);
    }
    // Smooth a bit (same trick NPC uses) and apply. */
    a = 0.95 * a + 0.05 * car_get_acceleration(car);
    car_set_acceleration(car, a);

    return PROCEDURE_STATUS_IN_PROGRESS;
}

void procedure_turn_cancel(Simulation* sim, Car* car, Procedure* procedure) {
    LOG_DEBUG("Cancelling turn procedure for car %d.", car->id);
    car_reset_all_control_variables(car); // reset all control variables
    return;
}