#include <stdlib.h>
#include "procedures.h"
#include "logging.h"
#include "math.h"


// --- STOP procedure ---

ProcedureStatusCode procedure_stop_init(Simulation* sim, Car* car, Procedure* procedure, const double* args) {

    // Read arguments
    bool use_linear_braking = args[0] > 0; // 0 = smooth braking, 1 = linear braking
    Seconds timeout_duration = args[1];
    bool use_preferred_accel_profile = args[2] > 0;

    // validate argument
    if (timeout_duration <= 0) {
        LOG_ERROR("Car %d: Invalid timeout duration for stop procedure: %.2f seconds.", car->id, timeout_duration);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR; // Invalid timeout duration
    }

    // validate necessity
    if (to_mph(fabs(car->speed)) < 1e-4) {
        LOG_INFO("Car %d: Already stopped. No need to stop again.", car->id);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_UNNECESSARY; // Already stopped
    }

    // All looks good, store state variables
    procedure->state[0] = sim_get_time(sim); // record start time
    procedure->state[1] = timeout_duration; // timeout duration
    procedure->state[2] = use_linear_braking ? 1.0 : 0.0; // use linear braking
    procedure->state[3] = use_preferred_accel_profile ? 1.0 : 0.0; // use preferred acceleration profile

    // Clear car's control variables
    car_set_acceleration(car, 0.0); // Set acceleration to zero
    // leave indicators as it is.

    LOG_DEBUG("Car %d: Stop procedure initialized. Timeout: %.2f seconds, Use Linear Braking: %d, Use Preferred Accel Profile: %d",
              car->id, timeout_duration, use_linear_braking, use_preferred_accel_profile);

    return PROCEDURE_STATUS_INITIALIZED;
}


static void apply_braking(Car* car, bool use_linear_braking, bool use_preferred_accel_profile) {
    MetersPerSecondSquared acc;
    if (use_linear_braking) {
        // Apply linear braking
        acc = car_compute_acceleration_stop_max_brake(car, use_preferred_accel_profile);
    } else {
        // Apply smooth braking
        acc = car_compute_acceleration_stop(car, use_preferred_accel_profile);
    }
    car_set_acceleration(car, acc);
}

ProcedureStatusCode procedure_stop_step(Simulation* sim, Car* car, Procedure* procedure) {

    // extract state
    Seconds start_time = procedure->state[0];
    Seconds timeout_duration = procedure->state[1];
    bool use_linear_braking = procedure->state[2] > 0;
    bool use_preferred_accel_profile = procedure->state[3] > 0;

    // check success
    if (fabs(car->speed) < 1e-4) {
        LOG_DEBUG("Car %d: Stop procedure completed successfully. Current speed: %.2f mph.",
                  car->id, to_mph(car->speed));
        apply_braking(car, use_linear_braking, use_preferred_accel_profile); // continue braking to erase any residual speed
        return PROCEDURE_STATUS_COMPLETED; // Target speed reached
    }

    // check timeout
    double now = sim_get_time(sim);
    if (now - start_time >= timeout_duration) {
        LOG_WARN("Car %d: Stop procedure timed out after %.2f seconds.", car->id, timeout_duration);
        apply_braking(car, use_linear_braking, use_preferred_accel_profile); // continue braking to erase any remaining speed
        return PROCEDURE_STATUS_FAILED_REASON_TIMEDOUT; // Procedure timed out
    }

    // set acceleration for braking
    apply_braking(car, use_linear_braking, use_preferred_accel_profile);
    LOG_TRACE("Car %d: Stop procedure in progress. Current Speed: %.2f mph, Acceleration: %.2f m/s²",
              car->id, to_mph(car_get_speed(car)), car_get_acceleration(car));

    return PROCEDURE_STATUS_IN_PROGRESS; // Procedure still in progress
}

void procedure_stop_cancel(Simulation* sim, Car* car, Procedure* procedure) {
    LOG_DEBUG("Car %d: Stop procedure canceled.", car->id);
    car_set_acceleration(car, 0.0); // Stop braking
    return;
}