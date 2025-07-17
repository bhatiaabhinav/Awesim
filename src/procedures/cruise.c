#include <stdlib.h>
#include "procedures.h"
#include "logging.h"


// --- CRUISE procedure ---

ProcedureStatusCode procedure_cruise_init(Simulation* sim, Car* car, Procedure* procedure, const double* args) {

    // Read arguments
    Seconds duration = args[0];
    MetersPerSecond cruise_speed_desired = args[1];
    bool is_cruise_adaptive = args[2] > 0;
    Seconds follow_distance_in_seconds = args[3];
    bool use_preferred_accel_profile = args[4] > 0;

    // validate arguments
    if (duration <= 0) {
        LOG_ERROR("Car %d: Invalid duration for cruise procedure: %.2f seconds.", car->id, duration);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR;
    }
    if (cruise_speed_desired < 0) {
        LOG_ERROR("Car %d: Invalid cruise speed desired for cruise procedure: %.2f m/s.", car->id, cruise_speed_desired);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR;
    }
    if (is_cruise_adaptive && follow_distance_in_seconds <= 0) {
        LOG_ERROR("Car %d: Invalid follow distance for adaptive cruise: %.2f seconds.", car->id, follow_distance_in_seconds);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR; // Invalid follow distance
    }


    // validate possibility
    if (cruise_speed_desired > car->capabilities.top_speed) {
        LOG_ERROR("Car %d: Cruise speed %.2f mph exceeds car's top speed %.2f mph.", car->id, to_mph(cruise_speed_desired), to_mph(car->capabilities.top_speed));
        return PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE; // Cruise speed exceeds max speed
    }

    // All looks good, store state variables
    procedure->state[0] = sim_get_time(sim); // record start time
    procedure->state[1] = duration;
    procedure->state[2] = cruise_speed_desired;
    procedure->state[3] = is_cruise_adaptive;
    procedure->state[4] = follow_distance_in_seconds;
    procedure->state[5] = use_preferred_accel_profile;

    // Clear car's control variables
    car_reset_all_control_variables(car);

    LOG_DEBUG("Car %d: Cruise procedure initialized. Duration: %.2f seconds, Desired Speed: %.2f mph (%s), Adaptive: %d, Follow Distance: %.2f seconds, Use Preferred Accel Profile: %d",
              car->id, duration, to_mph(cruise_speed_desired), is_cruise_adaptive ? "Yes" : "No", follow_distance_in_seconds, use_preferred_accel_profile);

    return PROCEDURE_STATUS_INITIALIZED;
}

ProcedureStatusCode procedure_cruise_step(Simulation* sim, Car* car, Procedure* procedure) {

    // extract state
    Seconds start_time = procedure->state[0];
    Seconds duration = procedure->state[1];
    MetersPerSecond cruise_speed_desired = procedure->state[2];
    bool is_cruise_adaptive = procedure->state[3] > 0;
    Seconds follow_distance_in_seconds = procedure->state[4];
    bool use_preferred_accel_profile = procedure->state[5] > 0;

    SituationalAwareness* situation = sim_get_situational_awareness(sim, car->id);

    // check success
    double now = sim_get_time(sim);
    if (now - start_time >= duration) {
        LOG_DEBUG("Car %d: cruise procedure completed successfully after %.2f seconds.", car->id, now - start_time);
        set_acceleration_cruise_control(car, situation, cruise_speed_desired, is_cruise_adaptive, follow_distance_in_seconds, use_preferred_accel_profile); // maintain cruise
        return PROCEDURE_STATUS_COMPLETED;
    }

    // set acceleration for cruise control
    set_acceleration_cruise_control(car, situation, cruise_speed_desired, is_cruise_adaptive, follow_distance_in_seconds, use_preferred_accel_profile);
    LOG_TRACE("Car %d: Cruise control active. Current Speed: %.2f mph, Desired speed: %.2f mph, Adaptive: %d", car->id, to_mph(car_get_speed(car)), to_mph(cruise_speed_desired), is_cruise_adaptive);
    return PROCEDURE_STATUS_IN_PROGRESS;
}

void procedure_cruise_cancel(Simulation* sim, Car* car, Procedure* procedure) {
    LOG_DEBUG("Cancelling cruise procedure for car %d.", car->id);
    car_reset_all_control_variables(car);
    return;
}