#pragma once

/*
 * High-level driving actions for NPC and scripted AI cars.
 * Each routine returns `true` once the manoeuvre is finished
 * (successfully or aborted) and `false` while it is still in progress.
 *
 * The implementation lives in actions.c.
 */

#include "ai.h"     /* Car, Simulation, numeric units, enums, …              */
#include "map.h"    /* Lane, road geometry utilities                         */
#include "utils.h"  /* Misc helpers, logging macros, rand_0_to_1(), …        */

/* ──────────────────────────────────────────────────────────────── */
/* Intersection turning                                             */
/* ──────────────────────────────────────────────────────────────── */

/** Execute a left-turn through an intersection. */
bool left_turn(Car *self, Simulation *sim);

/** Execute a right-turn through an intersection. */
bool right_turn(Car *self, Simulation *sim);

/* ──────────────────────────────────────────────────────────────── */
/* Lane merging                                                     */
/* ──────────────────────────────────────────────────────────────── */

/**
 * Initiate or continue a lane-merge manoeuvre.
 *
 * @param direction  "left", "l", "right", or "r" (case-insensitive).
 * @param duration   Hard time-out (seconds) before the merge aborts.
 */
bool merge(Car *self,
           Simulation *sim,
           const char *direction,
           double duration);

/* ──────────────────────────────────────────────────────────────── */
/* Overtaking / passing                                             */
/* ──────────────────────────────────────────────────────────────── */

/**
 * Pass a slower vehicle and (optionally) merge back.
 *
 * @param target         Vehicle to overtake.
 * @param should_merge   Merge back into the original lane afterwards?
 * @param merge_buffer   Longitudinal gap to leave before merging back (m).
 * @param speed_buffer   Extra speed relative to @p target while passing (m/s).
 * @param duration       Hard time-out (seconds) for the entire manoeuvre.
 */
bool pass_vehicle(Car *self,
                  Simulation *sim,
                  Car *target,
                  bool should_merge,
                  Meters merge_buffer,
                  MetersPerSecond speed_buffer,
                  double duration);

/* ──────────────────────────────────────────────────────────────── */
/* Basic cruising helpers                                           */
/* ──────────────────────────────────────────────────────────────── */

/**
 * Cruise toward a target speed.
 *
 * Returns `true` once the ego vehicle is within ±1 mph of the target.
 */
bool cruise(Car *self, float target_speed);

/**
 * Maintain current speed for a fixed duration (no-op helper).
 *
 * Useful for stitching actions in behaviour scripts.
 */
bool cruise_continue(Car *self, Simulation *sim, double duration);



typedef enum {
    PROCEDURE_NOOP,
    PROCEDURE_TURN,
    PROCEDURE_MERGE,
    PROCEDURE_PASS,
    PROCEDURE_CRUISE,
    NUM_PROCEDURES
} ProcedureType;

typedef enum {
    PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR, // Returned by procedure_init if the arguments are invalid, e.g., target lane id is invalid.
    PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE,     // Returned by procedure_init if the procedure request is impossible, e.g., turn right command when there is no right turn available.
    PROCEDURE_STATUS_INIT_FAILED_REASON_UNNECESSARY,    // Returned by procedure_init if the procedure is unnecessary, e.g., lane change request when we are already in the target lane.
    PROCEDURE_STATUS_INITIALIZED,                       // Returned by procedure_init if the procedure is possible and the state variables have been initialized.
    PROCEDURE_STATUS_IN_PROGRESS,                       // Returned by procedure_step if the procedure is in progress and will continue updating in the next step.
    PROCEDURE_STATUS_COMPLETED,                         // Returned by procedure_step if the procedure has completed successfully.
    PROCEDURE_STATUS_FAILED_REASON_IMPOSSIBLE,          // Returned by procedure_step if the procedure becomes impossible during execution.
    PROCEDURE_STATUS_FAILED_REASON_TIMEDOUT,            // Returned by procedure_step if the procedure has timed out.
} ProcedureStatusCode;

#define MAX_PROCEDURE_ARGS 5                    // Maximum number of arguments for a procedure
#define MAX_PROCEDURE_STATE_VARS 5              // Maximum number of state variables for a procedure

typedef struct {
    ProcedureType procedure_type;               // Type of the procedure
    CarId car_id;                               // ID of the car executing the procedure
    double args[MAX_PROCEDURE_ARGS];            // Arguments for the procedure, e.g., lane merge direction, target speed, turn direction etc.
    double state[MAX_PROCEDURE_STATE_VARS];     // State variables for the procedure, e.g., progress, start time, etc.
} Procedure;

// Initializes a procedure with the specified type and arguments. Will invoke the appropriate initialization function (turn_init, merge_init etc.) based on the procedure type.
// Returns a ProcedureStatusCode indicating the result of initialization.
ProcedureStatusCode procedure_init(Procedure* procedure, ProcedureType procedure_type, CarId car_id, int num_args, const double* args, Simulation* sim);

// Advances the procedure by one simulation step. Will invoke the appropriate step function (turn_step, merge_step etc.) based on the procedure type.
// Returns a ProcedureStatusCode indicating whether the procedure is in progress, completed, or failed.
ProcedureStatusCode procedure_step(Procedure* procedure, Simulation* sim);

// Cancels the procedure if it was initialized or is in progress. Will invoke the appropriate cancel function (turn_cancel, merge_cancel etc.) based on the procedure type.
void procedure_cancel(Procedure* procedure, Simulation* sim);


// ---- TURN procedure ----

// Initializes a TURN procedure.
ProcedureStatusCode procedure_turn_init(Procedure* procedure, Simulation* sim);

// Advances a TURN procedure.
ProcedureStatusCode procedure_turn_step(Procedure* procedure, Simulation* sim);

// Cancels a TURN procedure.
void procedure_turn_cancel(Procedure* procedure, Simulation* sim);


// ---- MERGE procedure ----

// Initializes a MERGE procedure.
ProcedureStatusCode procedure_merge_init(Procedure* procedure, Simulation* sim);

// Advances a MERGE procedure.
ProcedureStatusCode procedure_merge_step(Procedure* procedure, Simulation* sim);

// Cancels a MERGE procedure.
void procedure_merge_cancel(Procedure* procedure, Simulation* sim);


// ---- PASS procedure ----

// Initializes a PASS procedure.
ProcedureStatusCode procedure_pass_init(Procedure* procedure, Simulation* sim);

// Advances a PASS procedure.
ProcedureStatusCode procedure_pass_step(Procedure* procedure, Simulation* sim);

// Cancels a PASS procedure.
void procedure_pass_cancel(Procedure* procedure, Simulation* sim);


// ---- CRUISE procedure ----

// Initializes a CRUISE procedure.
ProcedureStatusCode procedure_cruise_init(Procedure* procedure, Simulation* sim);

// Advances a CRUISE procedure.
ProcedureStatusCode procedure_cruise_step(Procedure* procedure, Simulation* sim);

// Cancels a CRUISE procedure.
void procedure_cruise_cancel(Procedure* procedure, Simulation* sim);


// ---- NOOP procedure ----

// Initializes a NOOP procedure (does nothing).
ProcedureStatusCode procedure_noop_init(Procedure* procedure, Simulation* sim);

// Advances a NOOP procedure (typically a no-op).
ProcedureStatusCode procedure_noop_step(Procedure* procedure, Simulation* sim);

// Cancels a NOOP procedure.
void procedure_noop_cancel(Procedure* procedure, Simulation* sim);
