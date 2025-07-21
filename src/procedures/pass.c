#include <stdlib.h>
#include <string.h>
#include "procedures.h"
#include "logging.h"
/* ------------  layout of pass->state  ---------------------------------
 *  0  target car id
 *  1  start_time
 *  2  duration
 *  3  pass_speed_desired
 *  4  should_merge
 *  5  merge_dist_buffer
 *  6  use_preferred_accel_profile
 *  7  <spare for future use>     7 words for “pass” itself
 *  8 … 15   =  8 words for “merge” state (indices 0–7 inside merge)
 *--------------------------------------------------------------------- */
#define PASS_LOCALS 8   /* 0‑7 */
#define MERGE_STATE_WORDS 8   /* 8‑15 */
#define PASS_ORIG_LANE (PASS_LOCALS + MERGE_STATE_WORDS)      /* 16 */
#define PASS_STATE_WORDS (PASS_LOCALS + MERGE_STATE_WORDS + 1)  /* 17 */

// Ensure enough space available for multiple procedures
// _Static_assert(PASS_STATE_WORDS <= MAX_PROCEDURE_STATE_VARS,
//                "Procedure state[] too small for pass+merge");


typedef enum {
    PASS_PHASE_MERGE_OUT = 0, // Leave the target cars lane
    PASS_PHASE_OVERTAKE = 1, // Speed up to pass
    PASS_PHASE_MERGE_BACK = 2, // Get back into original lane to complete pass (optional)
    PASS_PHASE_DONE = 3
} PassPhase;
// --- PASS procedure ---

ProcedureStatusCode procedure_pass_init(Simulation* sim, Car* ego_car, Procedure* procedure, const double* args) {
    // Read arguments
    CarId target_car_id = (CarId)args[0];
    Seconds duration = args[1];
    MetersPerSecond pass_speed_desired = args[2];
    bool should_merge_back = args[3] > 0;
    Meters merge_dist_buffer = args[4];
    bool use_preferred_accel_profile = args[5] > 0;

    // validate arguments
    Car* target = sim_get_car(sim, target_car_id);
    if (!target) {
        LOG_ERROR("Car %d: Invalid target car id for pass procedure.", ego_car->id);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR;
    }
    if (duration <= 0) {
        LOG_ERROR("Car %d: Invalid duration for pass procedure: %.2f seconds.", ego_car->id, duration);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR;
    }
    if (pass_speed_desired < 0) {
        LOG_ERROR("Car %d: Invalid pass speed buffer for cruise procedure: %.2f m/s.", ego_car->id, pass_speed_desired);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR;
    }
    if (merge_dist_buffer < 0) {
        LOG_ERROR("Car %d: Invalid merge distance buffer for cruise procedure: %.2f m/s.", ego_car->id, merge_dist_buffer);
        return PROCEDURE_STATUS_INIT_FAILED_REASON_ARGUMENT_ERROR;
    }
    
    // validate possibility
    if (pass_speed_desired > ego_car->capabilities.top_speed) {
        LOG_ERROR("Car %d: Pass speed %.2f mph exceeds car's top speed %.2f mph.", ego_car->id, to_mph(pass_speed_desired), to_mph(ego_car->capabilities.top_speed));
        return PROCEDURE_STATUS_INIT_FAILED_REASON_IMPOSSIBLE; // Cruise speed exceeds max speed
    }

    procedure->state[0] = target_car_id;
    procedure->state[1] = sim_get_time(sim);
    procedure->state[2] = duration;
    procedure->state[3] = pass_speed_desired;
    procedure->state[4] = should_merge_back;
    procedure->state[5] = merge_dist_buffer;
    procedure->state[6] = use_preferred_accel_profile;
    procedure->state[7] = PASS_PHASE_MERGE_OUT; // Start at phase 0

    /* remember original lane so we know where to merge back
       (store it *after* the merge slice so we don’t overwrite
       merge’s own state).                                            */
    // procedure->state[PASS_LOCALS + MERGE_STATE_WORDS] = (double)car_get_lane_id(ego_car);
    procedure->state[PASS_ORIG_LANE] = (double)car_get_lane_id(ego_car);
    
    // ------- Merge Procedure --------
    
    // merge sub procedure init
    double merge_args[6] = {
        INDICATOR_LEFT,
        duration * 0.30,
        pass_speed_desired,
        1,                      // adaptive
        2,                      // follow distance in seconds
        use_preferred_accel_profile
    };
    // Initialize the merge procedure

    /* --- initialise merge sub‑procedure in the reserved slice ------- */
    Procedure merge_proc = {0};                       /* wrapper object */
    memcpy(merge_proc.state, procedure->state + PASS_LOCALS, sizeof(double) * MERGE_STATE_WORDS);

    ProcedureStatusCode mstat = procedure_merge_init(sim, ego_car, &merge_proc, merge_args);
    // Procedure* merge_procedure = sim_get_ongoing_procedure(sim, ego_car->id);
    // ProcedureStatusCode mstat = procedure_merge_init(sim, ego_car, merge_procedure, merge_args);
    
    if (mstat != PROCEDURE_STATUS_INITIALIZED)
        return mstat;                  /* bubble the error up */

    // Flag to remember that merge is “active” (0 = running)
    procedure->state[PASS_ORIG_LANE] = (double)car_get_lane_id(ego_car);

    car_reset_all_control_variables(ego_car);

    LOG_DEBUG("Car %d: Pass procedure initialized. Duration: %.2f seconds, Desired Pass Speed: %.2f mph, Merge Distance Buffer: %.2f Meters, Use Preferred Accel Profile: %d",
              ego_car->id, duration, to_mph(pass_speed_desired), meters(merge_dist_buffer), use_preferred_accel_profile);

    return PROCEDURE_STATUS_INITIALIZED;
}

ProcedureStatusCode procedure_pass_step(Simulation* sim, Car* car, Procedure* procedure) {
    // TODO: Can be made smarter by going into lane that is the most "Free" for passing
    // extract state
    CarId target_car_id = (CarId) procedure->state[0];
    Seconds start_time = procedure->state[1];
    Seconds duration = procedure->state[2];
    MetersPerSecond pass_speed_desired = procedure->state[3];
    bool should_merge_back = procedure->state[4] > 0;
    Meters merge_dist_buffer = procedure->state[5];
    bool use_preferred_acc_profile = procedure->state[6] > 0;
    PassPhase phase = (PassPhase) procedure->state[7];

    // Merge Sub-procedure
    /* Re‑attach wrapper to the same slice each tick */
    Procedure merge_proc = {0};
    memcpy(merge_proc.state, procedure->state + PASS_LOCALS, sizeof(double) * MERGE_STATE_WORDS);

    Car* target = sim_get_car(sim, target_car_id);

    double now = sim_get_time(sim);
    if (now - start_time > duration) {
        LOG_WARN("Car %d: pass procedure timed out after %.2f seconds.", car->id, duration);
        procedure_merge_cancel(sim, car, &merge_proc);
        return PROCEDURE_STATUS_FAILED_REASON_TIMEDOUT;
    }

    // ------- Phase-0 (Leave target car's lane, try left lane first) -------
    if (phase == PASS_PHASE_MERGE_OUT) {
        ProcedureStatusCode mstat = procedure_merge_step(sim, car, &merge_proc);

        if (mstat == PROCEDURE_STATUS_COMPLETED) {
            procedure->state[6] = PASS_PHASE_OVERTAKE; // successfully merged, so move on to overtake
        } else if (mstat != PROCEDURE_STATUS_IN_PROGRESS) {
            return mstat;   // Merge not possible, error...
        }
        return PROCEDURE_STATUS_IN_PROGRESS;
    }

    // ------- Phase-1 (Leave target car's lane, try left lane first)-------
    if (phase == PASS_PHASE_OVERTAKE) {
        SituationalAwareness* situation = sim_get_situational_awareness(sim, car->id);
        // TODO: Check that this does not break when the target crosses to another lane.
        // See if ego car passed target car (does not check that target car is in the same original lane during procedure)
        if (car_get_lane_progress_meters(car) > car_get_lane_progress_meters(target) + car_get_length(car) + merge_dist_buffer){
            if (should_merge_back) {
                // Set indicator toward original lane
                LaneId orig_lane = (LaneId) procedure->state[PASS_LOCALS + MERGE_STATE_WORDS - 2];

                const Lane *left  = situation->lane_left;
                const Lane *right = situation->lane_right;
                CarIndictor dir =
                    (left  && left->id  == orig_lane) ? INDICATOR_LEFT  :
                    (right && right->id == orig_lane) ? INDICATOR_RIGHT :
                    (orig_lane < car_get_lane_id(car)) ? INDICATOR_LEFT  :
                                                         INDICATOR_RIGHT;
                // Re‑initialise merge sub‑procedure
                double merge_args[6] = {
                    dir,
                    duration * 0.30,            // 30 % of whole pass
                    pass_speed_desired,
                    1,                          // adaptive cruise afterwards
                    2,                          // Follow disatance in seconds
                    use_preferred_acc_profile
                };
                ProcedureStatusCode mstat = procedure_merge_init(sim, car, &merge_proc, merge_args);

                if (mstat != PROCEDURE_STATUS_INITIALIZED)
                    return mstat;

                procedure->state[6] = PASS_PHASE_MERGE_BACK;
                return PROCEDURE_STATUS_IN_PROGRESS;
            }
            procedure->state[6] = PASS_PHASE_DONE;
            return PROCEDURE_STATUS_COMPLETED;
        }
        MetersPerSecond desired = pass_speed_desired;
        if (pass_speed_desired <= car_get_speed(target)){
            desired = car_get_speed(target) + mps(5); // If the desired speed is less than the target's speed we set a desired speed to ensure passing
        }
        // Use an adaptive cruise control to pass the target vehicle
        set_acceleration_cruise_control(car, situation, desired, true, 2, use_preferred_acc_profile);
        return PROCEDURE_STATUS_IN_PROGRESS;
    }
    // ------- Phase-2 (Merge back into original lane)-------
    if (phase == PASS_PHASE_MERGE_BACK) {
        ProcedureStatusCode mstat = procedure_merge_step(sim, car, &merge_proc);

        if (mstat == PROCEDURE_STATUS_COMPLETED) {
            procedure->state[6] = PASS_PHASE_DONE;
            return PROCEDURE_STATUS_COMPLETED;
        }
        return mstat; // In progress or some failure occured
    }
    LOG_ERROR("Car %d: Invalid pass phase %d.", car->id, phase);
    return PROCEDURE_STATUS_FAILED_REASON_NOW_IMPOSSIBLE; // Need to return something (as per the function signature) in all control paths, else it is a compiler warning (and a runtime error if `phase` actually ends up with an invalid value due to a bug).
}

void procedure_pass_cancel(Simulation* sim, Car* car, Procedure* procedure) {
    LOG_DEBUG("Cancelling PASS procedure for car %d.", car->id);
    car_reset_all_control_variables(car); // reset all control variables
    return;
}