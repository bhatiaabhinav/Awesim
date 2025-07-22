#include <stdlib.h>
#include "procedures.h"
#include "logging.h"


// --- NO_OP procedure ---

ProcedureStatusCode procedure_no_op_init(Simulation* sim, Car* car, Procedure* procedure, const double* args) {

    // Read arguments
    Seconds duration = args[0];
    bool leave_indicators_on = args[1] > 0;

    // validate arguments
    if (duration <= 0) {
        LOG_ERROR("Car %d: Invalid duration for NO_OP procedure: %.2f seconds.", car->id, duration);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR; // Invalid duration
    }

    // Clear acceleration.
    car_set_acceleration(car, 0.0); // Set acceleration to zero
    if (!leave_indicators_on) {
        car->indicator_lane = INDICATOR_NONE; // Turn off lane change indicator
        car->indicator_turn = INDICATOR_NONE; // Turn off turn indicator
        car->request_indicated_lane = false; // Reset lane change request
        car->request_indicated_turn = false; // Reset turn request
    }

    LOG_DEBUG("Car %d: NO_OP procedure initialized. Duration: %.2f seconds, Leave Indicators On: %s",
              car->id, duration, leave_indicators_on ? "Yes" : "No");


    // Store state variables
    procedure->state[0] = sim_get_time(sim); // record start time
    procedure->state[1] = duration;
    procedure->state[2] = leave_indicators_on ? 1.0 : 0.0;


    return PROCEDURE_STATUS_INITIALIZED;
}

ProcedureStatusCode procedure_no_op_step(Simulation* sim, Car* car, Procedure* procedure) {

    // extract state
    Seconds start_time = procedure->state[0];
    Seconds duration = procedure->state[1];
    bool leave_indicators_on = procedure->state[2] > 0;

    // check success
    double now = sim_get_time(sim);
    if (now - start_time >= duration) {
        LOG_DEBUG("Car %d:  procedure no_op completed successfully after %.2f seconds.", car->id, now - start_time);
        car_set_acceleration(car, 0.0); // Set acceleration to zero
        if (!leave_indicators_on) {
            car->indicator_lane = INDICATOR_NONE; // Turn off lane change indicator
            car->indicator_turn = INDICATOR_NONE; // Turn off turn indicator
            car->request_indicated_lane = false; // Reset lane change request
            car->request_indicated_turn = false; // Reset turn request
        }
        return PROCEDURE_STATUS_COMPLETED;
    }

    // apply no_op controls
    car_set_acceleration(car, 0.0); // Set acceleration to zero
    if (leave_indicators_on) {
        // Leave indicators as they are
    } else {
        car->indicator_lane = INDICATOR_NONE; // Turn off lane change indicator
        car->indicator_turn = INDICATOR_NONE; // Turn off turn indicator
        car->request_indicated_lane = false; // Reset lane change request
        car->request_indicated_turn = false; // Reset turn request
    }

    LOG_TRACE("Car %d: NO_OP procedure in progress. Indicators left on: %s, Duration left: %.2f seconds.", car->id, leave_indicators_on ? "Yes" : "No", duration - (now - start_time));

    return PROCEDURE_STATUS_IN_PROGRESS;
}

void procedure_no_op_cancel(Simulation* sim, Car* car, Procedure* procedure) {
    LOG_DEBUG("Cancelling NO_OP procedure for car %d.", car->id);
    // Do not touch anything. If indicators were left on, they will remain on.
    return;
}