#pragma once

#include "car.h"
#include "ai.h"


// TODO: Document meaning of each procedure, their arguments, and their success & failure conditions.

typedef enum {
    PROCEDURE_NONE,         // No procedure going on. The car may be setting its control variables directly instead of through a high-level procedure.
    PROCEDURE_TURN,
    PROCEDURE_MERGE,        // Merge into a lane (which maybe an adjacent lane, highway merge, or highway exit). The car will try to merge into the indicated direction while maintaining a safe distance from the lead vehicle. Assumes no turns and lane changes other than the requested merge.
    PROCEDURE_PASS,         // Overtake another vehicle. The car will try to pass the target by first attempting to merge on the left lane, then accelerating to the desired speed. If allowed, the ego car will merge into the target lane.
    PROCEDURE_ADJUST_SPEED, // Adjust speed to a target speed. Procedure ends successfully when the speed is within a small epsilon of the target speed. Assumes no lane changes and turns and ignores traffic.
    PROCEDURE_CRUISE,       // Cruise control procedure. The car will try to maintain a target speed for a specified duration. IF adaptive cruise is enabled, it will also maintain a safe distance from the lead vehicle. Assumes no lane changes and turns.
    NUM_PROCEDURES
} ProcedureType;

typedef enum {
    PROCEDURE_STATUS_NONE,                              // No procedure in progress.
    PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR, // Returned by procedure_init if the arguments are invalid, e.g., target lane id is invalid.
    PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE,     // Returned by procedure_init if the procedure request is impossible, e.g., turn right command when there is no right turn available.
    PROCEDURE_STATUS_INIT_FAILED_REASON_NOT_IMPLEMENTED, // Returned by procedure_init if the procedure request is not implemented.
    PROCEDURE_STATUS_INIT_FAILED_REASON_UNNECESSARY,    // Returned by procedure_init if the procedure is unnecessary, e.g., lane change request when we are already in the target lane.
    PROCEDURE_STATUS_INIT_FAILED_REASON_ANOTHER_PROCEDURE_IN_PROGRESS, // Returned by procedure_init if another procedure is already in progress.
    PROCEDURE_STATUS_INITIALIZED,                       // Returned by procedure_init if the procedure is possible and the state variables have been initialized.
    PROCEDURE_STATUS_IN_PROGRESS,                       // Returned by procedure_step if the procedure is in progress and will continue updating in the next step.
    PROCEDURE_STATUS_COMPLETED,                         // Returned by procedure_step if the procedure has completed successfully.
    PROCEDURE_STATUS_FAILED_REASON_NOW_IMPOSSIBLE,      // Returned by procedure_step if the procedure becomes impossible during execution.
    PROCEDURE_STATUS_FAILED_REASON_TIMEDOUT,            // Returned by procedure_step if the procedure has timed out.
    PROCEDURE_STATUS_FAILED_REASON_UNINITIALIZED,       // Returned by procedure_step if the procedure was not initialized properly.
} ProcedureStatusCode;

#define MAX_PROCEDURE_STATE_VARS 16              // Maximum number of state variables for a procedure

typedef struct {
    ProcedureType type;                         // Type of the procedure
    ProcedureStatusCode status;                 // Status of the procedure, if initialized, in progress, completed, or failed.
    double state[MAX_PROCEDURE_STATE_VARS];     // State variables for the procedure, e.g., progress, start time, etc.
    void *opaque;          /* user‑defined scratch pointer */
} Procedure;

// Initializes a procedure with the specified type and arguments. Will invoke the appropriate initialization function (turn_init, merge_init etc.) based on the procedure type.
// Returns a ProcedureStatusCode indicating the result of initialization.
ProcedureStatusCode procedure_init(Simulation* sim, Car* car, Procedure* procedure, ProcedureType procedure_type, const double* args, int num_args);

// Advances the procedure by one simulation step. Will invoke the appropriate step function (turn_step, merge_step etc.) based on the procedure type.
// Returns a ProcedureStatusCode indicating whether the procedure is in progress, completed, or failed.
ProcedureStatusCode procedure_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels the procedure if it was initialized or is in progress. Will invoke the appropriate cancel function (turn_cancel, merge_cancel etc.) based on the procedure type.
void procedure_cancel(Simulation* sim, Car* car, Procedure* procedure);







// ! THE FOLLOWING FUNCTIONS MUST *NOT* CHANGE `type` or `status` of the procedure. They should only manage the `state` variables of the procedure and return the appropriate status code for the function outcome.



// ---- TURN procedure ----

// Initializes a TURN procedure.
ProcedureStatusCode procedure_turn_init(Simulation* sim, Car* car, Procedure* procedure, const double* args);

// Advances a TURN procedure.
ProcedureStatusCode procedure_turn_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels a TURN procedure.
void procedure_turn_cancel(Simulation* sim, Car* car, Procedure* procedure);


// ---- MERGE procedure ----

// Initializes a MERGE procedure. Arguments: CarIndicator direction (left/right), timeout_duration, desired cruise speed, whether cruise is adaptive, follow distance in seconds (if adaptive), whether to use preferred acceleration profile (0 or 1)
ProcedureStatusCode procedure_merge_init(Simulation* sim, Car* car, Procedure* procedure, const double* args);

// Advances a MERGE procedure.
ProcedureStatusCode procedure_merge_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels a MERGE procedure.
void procedure_merge_cancel(Simulation* sim, Car* car, Procedure* procedure);


// ---- PASS procedure ----

// Initializes a PASS procedure. Arguments: timeout duration, desired pass speed, whether the car should merge back into original lane, distance buffer to determine if passed, whether to use preferred acceleration profile (0 or 1)
ProcedureStatusCode procedure_pass_init(Simulation* sim, Car* car, Procedure* procedure, const double* args);

// Advances a PASS procedure.
ProcedureStatusCode procedure_pass_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels a PASS procedure.
void procedure_pass_cancel(Simulation* sim, Car* car, Procedure* procedure);



// ---- ADJUST SPEED procedure ----

// Initializes an ADJUST SPEED procedure. Arguments: desired speed, tolerance, timeout duration, whether to use preferred acceleration profile (0 or 1)
ProcedureStatusCode procedure_adjust_speed_init(Simulation* sim, Car* car, Procedure* procedure, const double* args);

// Advances an ADJUST SPEED procedure.
ProcedureStatusCode procedure_adjust_speed_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels an ADJUST SPEED procedure.
void procedure_adjust_speed_cancel(Simulation* sim, Car* car, Procedure* procedure);


// ---- CRUISE procedure ----

// Initializes a CRUISE procedure. Arguments: duration, desired cruise speed, whether adaptive, follow duration in seconds (if adaptive), whether to use preferred acceleration profile (0 or 1)
ProcedureStatusCode procedure_cruise_init(Simulation* sim, Car* car, Procedure* procedure, const double* args);

// Advances a CRUISE procedure.
ProcedureStatusCode procedure_cruise_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels a CRUISE procedure.
void procedure_cruise_cancel(Simulation* sim, Car* car, Procedure* procedure);



// ---- Utility functions ----


// Sets acceleration for adaptive (if enabled) cruise control based on desired cruise speed, follow distance in seconds to next vehicle (ignored if not adaptive), and acceleration profile.
void set_acceleration_cruise_control(Car* car, const SituationalAwareness* situation, MetersPerSecond cruise_speed_desired, bool adaptive, MetersPerSecond follow_distance_in_seconds, bool use_preferred_acc_profile);