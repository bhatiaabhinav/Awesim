#include <stdlib.h>
#include "procedures.h"
#include "logging.h"
#include "math.h"


// --- ADJUST SPEED procedure ---

ProcedureStatusCode procedure_adjust_speed_init(Simulation* sim, Car* car, Procedure* procedure, const double* args) {

    // Read arguments
    MetersPerSecond target_speed = args[0];
    MetersPerSecond tolerance = args[1];
    Seconds timeout_duration = args[2];
    bool use_preferred_accel_profile = args[3] > 0;

    // validate arguments
    if (timeout_duration <= 0) {
        LOG_ERROR("Car %d: Invalid timeout duration for adjust speed procedure: %.2f seconds.", car->id, timeout_duration);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR;
    }
    if (tolerance < 0) {
        LOG_ERROR("Car %d: Invalid tolerance for adjust speed procedure: %.2f mph. Must be > 0 mph. DO NOT SET ZERO.", car->id, to_mph(tolerance));
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR; // Invalid tolerance
    }

    // validate possibility
     if (fabs(target_speed) > car->capabilities.top_speed) {
        LOG_ERROR("Car %d: Target speed %.2f mph exceeds car's top speed %.2f mph.", car->id, to_mph(target_speed), to_mph(car->capabilities.top_speed));
        return PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE; // Target speed exceeds max speed
    }

    // validate necessity
    if (to_mph(fabs(car->speed - target_speed)) < tolerance) {
        LOG_INFO("Car %d: Target speed is already reached: %.2f mph.", car->id, to_mph(car->speed));
        return PROCEDURE_STATUS_INIT_FAILED_REASON_UNNECESSARY; // Target speed already reached
    }

    // All looks good, store state variables
    procedure->state[0] = target_speed; // target speed
    procedure->state[1] = tolerance; // tolerance
    procedure->state[2] = sim_get_time(sim); // record start time
    procedure->state[3] = timeout_duration; // timeout duration
    procedure->state[4] = use_preferred_accel_profile; // use preferred acceleration profile

    // Clear car's acceleration, leave indicators and requests unchanged
    car_set_acceleration(car, 0.0);

    LOG_DEBUG("Car %d: Adjust speed procedure initialized. Target speed: %.2f mph, Timeout: %.2f seconds, Use Preferred Accel Profile: %d",
              car->id, to_mph(target_speed), timeout_duration, use_preferred_accel_profile);

    return PROCEDURE_STATUS_INITIALIZED;
}

ProcedureStatusCode procedure_adjust_speed_step(Simulation* sim, Car* car, Procedure* procedure) {

    // extract state
    MetersPerSecond target_speed = procedure->state[0];
    MetersPerSecond tolerance = procedure->state[1];
    Seconds start_time = procedure->state[2];
    Seconds timeout_duration = procedure->state[3];
    bool use_preferred_accel_profile = procedure->state[4] > 0;

    // check success
    if (fabs(car->speed - target_speed) < tolerance) {
        LOG_DEBUG("Car %d: Adjust speed procedure completed successfully. Current speed: %.2f mph, Target speed: %.2f mph.",
                  car->id, to_mph(car->speed), to_mph(target_speed));
        car_set_acceleration(car, 0.0); // Stop acceleration
        return PROCEDURE_STATUS_COMPLETED; // Target speed reached
    }

    // check timeout
    double now = sim_get_time(sim);
    if (now - start_time >= timeout_duration) {
        LOG_WARN("Car %d: Adjust speed procedure timed out after %.2f seconds. Current speed: %.2f mph, Target speed: %.2f mph.",
                 car->id, now - start_time, to_mph(car->speed), to_mph(target_speed));
        car_set_acceleration(car, car_compute_acceleration_adjust_speed(car, target_speed, use_preferred_accel_profile)); // Continue acceleration to adjust speed
        return PROCEDURE_STATUS_FAILED_REASON_TIMEDOUT; // Procedure timed out
    }

    // set acceleration for speed adjustment
    MetersPerSecondSquared acc = car_compute_acceleration_adjust_speed(car, target_speed, use_preferred_accel_profile);
    car_set_acceleration(car, acc);
    LOG_TRACE("Car %d: Adjust speed active. Current Speed: %.2f mph, Target Speed: %.2f mph, Acceleration: %.2f m/s²",
              car->id, to_mph(car_get_speed(car)), to_mph(target_speed), acc);

    return PROCEDURE_STATUS_IN_PROGRESS; // Procedure still in progress
}

void procedure_adjust_speed_cancel(Simulation* sim, Car* car, Procedure* procedure) {
    LOG_DEBUG("Car %d: Adjust speed procedure canceled.", car->id);
    car_set_acceleration(car, 0.0); // Stop acceleration
    return;
}