#include <stdlib.h>
#include "procedures.h"
#include "ai.h"
#include "logging.h"

ProcedureStatusCode procedure_init(Simulation* sim, Car* car, Procedure* procedure, ProcedureType procedure_type, const double* args, int num_args) {
    if (procedure->status == PROCEDURE_STATUS_INITIALIZED || procedure->status == PROCEDURE_STATUS_IN_PROGRESS) {
        LOG_ERROR("Procedure already initialized or in progress for car %d. Please let it finish or cancel it before initializing a new one.", car->id);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ANOTHER_PROCEDURE_IN_PROGRESS; // Another procedure is already in progress
    }
    procedure->type = procedure_type;
    ProcedureStatusCode status = PROCEDURE_STATUS_NONE;
    for (int i = 0; i < MAX_PROCEDURE_STATE_VARS; i++) {
        procedure->state[i] = 0.0; // Initialize state variables to zero
    }
    switch (procedure_type) {
        case PROCEDURE_NO_OP:
            status = procedure_no_op_init(sim, car, procedure, args);
            break;
        case PROCEDURE_STOP:
            status = procedure_stop_init(sim, car, procedure, args);
            break;
        case PROCEDURE_TURN:
            status = procedure_turn_init(sim, car, procedure, args);
            break;
        case PROCEDURE_MERGE:
            status = procedure_merge_init(sim, car, procedure, args);
            break;
        case PROCEDURE_PASS:
            status = procedure_pass_init(sim, car, procedure, args);
            break;
        case PROCEDURE_ADJUST_SPEED:
            status = procedure_adjust_speed_init(sim, car, procedure, args);
            break;
        case PROCEDURE_CRUISE:
            status = procedure_cruise_init(sim, car, procedure, args);
            break;
        default:
            LOG_ERROR("Invalid procedure type %d for car %d. Cannot initialize.", procedure_type, car->id);
            status = PROCEDURE_STATUS_INIT_FAILED_REASON_NOT_IMPLEMENTED; // Invalid procedure type
    }
    if (status == PROCEDURE_STATUS_INITIALIZED) {
        LOG_DEBUG("Procedure %d initialized for car %d.", procedure_type, car->id);
        procedure->status = status;
    } else {
        switch (status)
        {
            case PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR:
                LOG_ERROR("Procedure %d initialization failed for car %d due to argument error.", procedure_type, car->id);
                break;
            case PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE:
                LOG_ERROR("Procedure %d initialization failed for car %d due to impossibility.", procedure_type, car->id);
                break;
            case PROCEDURE_STATUS_INIT_FAILED_REASON_UNNECESSARY:
                LOG_WARN("Procedure %d initialization for car %d is unnecessary.", procedure_type, car->id);
                break;
            case PROCEDURE_STATUS_INIT_FAILED_REASON_NOT_IMPLEMENTED:
                LOG_ERROR("Procedure %d is not implemented for car %d.", procedure_type, car->id);
                break;
            default:
                LOG_ERROR("Procedure %d initialization failed for car %d with unknown reason.", procedure_type, car->id);
        }
        procedure->type = PROCEDURE_NONE; // Reset procedure type if initialization failed
        procedure->status = PROCEDURE_STATUS_NONE;  // Reset status if initialization failed
        for (int i = 0; i < MAX_PROCEDURE_STATE_VARS; i++) {
            procedure->state[i] = 0.0; // Reset state variables if initialization failed
        }
    }
    return status;
}

ProcedureStatusCode procedure_step(Simulation* sim, Car* car, Procedure* procedure) {
    if (!(procedure->status == PROCEDURE_STATUS_INITIALIZED || procedure->status == PROCEDURE_STATUS_IN_PROGRESS)) {
        LOG_ERROR("No procedure initialized or in progress for car %d. Cannot step.", car->id);
        return PROCEDURE_STATUS_FAILED_REASON_UNINITIALIZED; // Procedure not initialized properly
    }
    ProcedureStatusCode status;
    switch (procedure->type) {
        case PROCEDURE_NO_OP:
            status = procedure_no_op_step(sim, car, procedure);
            break;
        case PROCEDURE_STOP:
            status = procedure_stop_step(sim, car, procedure);
            break;
        case PROCEDURE_TURN:
            status = procedure_turn_step(sim, car, procedure);
            break;
        case PROCEDURE_MERGE:
            status = procedure_merge_step(sim, car, procedure);
            break;
        case PROCEDURE_PASS:
            status = procedure_pass_step(sim, car, procedure);
            break;
        case PROCEDURE_ADJUST_SPEED:
            status = procedure_adjust_speed_step(sim, car, procedure);
            break;
        case PROCEDURE_CRUISE:
            status = procedure_cruise_step(sim, car, procedure);
            break;
        default:
            status = PROCEDURE_STATUS_FAILED_REASON_UNINITIALIZED; // Invalid procedure type
    }
    procedure->status = status; // Update procedure status
    switch (status)
    {
        case PROCEDURE_STATUS_IN_PROGRESS:
            LOG_TRACE("Procedure %d is in progress for car %d.", procedure->type, car->id);
            break;
        case PROCEDURE_STATUS_COMPLETED:
            LOG_DEBUG("Procedure %d completed successfully for car %d.", procedure->type, car->id);
            break;
        case PROCEDURE_STATUS_FAILED_REASON_NOW_IMPOSSIBLE:
            LOG_WARN("Procedure %d failed for car %d due to impossibility.", procedure->type, car->id);
            break;
        case PROCEDURE_STATUS_FAILED_REASON_TIMEDOUT:
            LOG_WARN("Procedure %d timed out for car %d.", procedure->type, car->id);
            break;
        case PROCEDURE_STATUS_FAILED_REASON_UNINITIALIZED:
            LOG_ERROR("Procedure %d was not initialized properly for car %d.", procedure->type, car->id);
            break;
        default:
            break;
    }
    return status;
}

void procedure_cancel(Simulation* sim, Car* car, Procedure* procedure) {
    if (procedure->status == PROCEDURE_STATUS_INITIALIZED || procedure->status == PROCEDURE_STATUS_IN_PROGRESS) {
        LOG_INFO("Cancelling procedure %d for car %d.", procedure->type, car->id);
        switch (procedure->type) {
            case PROCEDURE_NO_OP:
                procedure_no_op_cancel(sim, car, procedure);
                break;
            case PROCEDURE_STOP:
                procedure_stop_cancel(sim, car, procedure);
                break;
            case PROCEDURE_TURN:
                procedure_turn_cancel(sim, car, procedure);
                break;
            case PROCEDURE_MERGE:
                procedure_merge_cancel(sim, car, procedure);
                break;
            case PROCEDURE_PASS:
                procedure_pass_cancel(sim, car, procedure);
                break;
            case PROCEDURE_ADJUST_SPEED:
                procedure_adjust_speed_cancel(sim, car, procedure);
                break;
            case PROCEDURE_CRUISE:
                procedure_cruise_cancel(sim, car, procedure);
                break;
            default:
                // No valid procedure to cancel
                break;
        }
    } else {
        LOG_WARN("No procedure to cancel for car %d. Current status: %d", car->id, procedure->status);
    }
    procedure->type = PROCEDURE_NONE; // Reset procedure type after cancellation
    procedure->status = PROCEDURE_STATUS_NONE; // Reset status after cancellation
    for (int i = 0; i < MAX_PROCEDURE_STATE_VARS; i++) {
        procedure->state[i] = 0.0; // Reset state variables after cancellation
    }
}


// --- Utility functions ---

void set_acceleration_cruise_control(Car* car, const SituationalAwareness* situation, MetersPerSecond cruise_speed_desired, bool adaptive, MetersPerSecond follow_distance_in_seconds, bool use_preferred_acc_profile) {
    MetersPerSecondSquared acc;
    if (adaptive) {
        acc = car_compute_acceleration_adaptive_cruise(car, situation, cruise_speed_desired, follow_distance_in_seconds, from_feet(3), use_preferred_acc_profile);
    } else {
        acc = car_compute_acceleration_cruise(car, cruise_speed_desired, use_preferred_acc_profile);
    }
    car_set_acceleration(car, acc);
}
