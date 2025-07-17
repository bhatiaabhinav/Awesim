#pragma once

#include "car.h"

typedef enum {
    PROCEDURE_NONE,     // No procedure going on. The car may be setting its control variables directly instead of through a high-level procedure.
    PROCEDURE_TURN,
    PROCEDURE_MERGE,
    PROCEDURE_PASS,
    PROCEDURE_CRUISE,
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

#define MAX_PROCEDURE_STATE_VARS 5              // Maximum number of state variables for a procedure

typedef struct {
    ProcedureType type;                         // Type of the procedure
    ProcedureStatusCode status;                 // Status of the procedure, if initialized, in progress, completed, or failed.
    double state[MAX_PROCEDURE_STATE_VARS];     // State variables for the procedure, e.g., progress, start time, etc.
} Procedure;

// Initializes a procedure with the specified type and arguments. Will invoke the appropriate initialization function (turn_init, merge_init etc.) based on the procedure type.
// Returns a ProcedureStatusCode indicating the result of initialization.
ProcedureStatusCode procedure_init(Simulation* sim, Car* car, Procedure* procedure, ProcedureType procedure_type, const double* args);

// Advances the procedure by one simulation step. Will invoke the appropriate step function (turn_step, merge_step etc.) based on the procedure type.
// Returns a ProcedureStatusCode indicating whether the procedure is in progress, completed, or failed.
ProcedureStatusCode procedure_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels the procedure if it was initialized or is in progress. Will invoke the appropriate cancel function (turn_cancel, merge_cancel etc.) based on the procedure type.
void procedure_cancel(Simulation* sim, Car* car, Procedure* procedure);


// ---- TURN procedure ----

// Initializes a TURN procedure.
ProcedureStatusCode procedure_turn_init(Simulation* sim, Car* car, Procedure* procedure, const double* args);

// Advances a TURN procedure.
ProcedureStatusCode procedure_turn_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels a TURN procedure.
void procedure_turn_cancel(Simulation* sim, Car* car, Procedure* procedure);


// ---- MERGE procedure ----

// Initializes a MERGE procedure.
ProcedureStatusCode procedure_merge_init(Simulation* sim, Car* car, Procedure* procedure, const double* args);

// Advances a MERGE procedure.
ProcedureStatusCode procedure_merge_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels a MERGE procedure.
void procedure_merge_cancel(Simulation* sim, Car* car, Procedure* procedure);


// ---- PASS procedure ----

// Initializes a PASS procedure.
ProcedureStatusCode procedure_pass_init(Simulation* sim, Car* car, Procedure* procedure, const double* args);

// Advances a PASS procedure.
ProcedureStatusCode procedure_pass_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels a PASS procedure.
void procedure_pass_cancel(Simulation* sim, Car* car, Procedure* procedure);


// ---- CRUISE procedure ----

// Initializes a CRUISE procedure.
ProcedureStatusCode procedure_cruise_init(Simulation* sim, Car* car, Procedure* procedure, const double* args);

// Advances a CRUISE procedure.
ProcedureStatusCode procedure_cruise_step(Simulation* sim, Car* car, Procedure* procedure);

// Cancels a CRUISE procedure.
void procedure_cruise_cancel(Simulation* sim, Car* car, Procedure* procedure);

