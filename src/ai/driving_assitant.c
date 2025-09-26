#include "ai.h"
#include "logging.h"
#include <assert.h>
#include <math.h>

static const double MINIMUM_FOLLOW_MODE_BUFFER_FEET = 1;    // Minimum follow mode buffer to avoid unrealistic behavior
static const double DEFAULT_FOLLOW_MODE_BUFFER_FEET = 3;    // Default follow mode buffer in feet
static const Seconds DEFAULT_TIME_HEADWAY = 3.0;            // Default time headway


bool driving_assistant_reset_settings(DrivingAssistant* das, Car* car) {
    if (!car || !das) {
        LOG_ERROR("Invalid car or driving assistant provided to driving_assistant_reset_settings");
        return false; // Indicating error due to invalid input
    }

    // Resetting all settings to default values
    das->last_configured_at = 0;
    das->speed_target = 0.0;
    das->stop_mode = false; // Default to false
    das->stopping_distance_target = 0; // Default to 0 meters (which is not feasible) so this triggers full braking if stop_mode is engaged.
    das->merge_intent = INDICATOR_NONE;
    das->turn_intent = INDICATOR_NONE;
    das->follow_mode = false;
    das->follow_mode_buffer = from_feet(DEFAULT_FOLLOW_MODE_BUFFER_FEET);
    das->follow_mode_thw = DEFAULT_TIME_HEADWAY;      // Default time headway
    das->merge_assistance = false;   // Default to false
    das->aeb_assistance = false;     // Default to false
    das->aeb_in_progress = false;    // Reset AEB state
    das->aeb_manually_disengaged = false; // Reset manual disengagement state
    das->use_preferred_accel_profile = false; // Reset preferred acceleration profile state
    das->use_linear_speed_control = false; // Reset linear speed control state

    LOG_TRACE("Driving assistant settings reset for car %d", car->id);
    return true; // Indicating success
}


// Computes the best-case gap (minimum feasible distance) between the ego vehicle and the lead vehicle
// when the ego applies maximum braking to avoid a forward collision.
// The gap is the bumper-to-bumper distance, accounting for vehicle lengths.
// The function assumes constant accelerations until vehicles stop (speed reaches zero),
// after which they remain stationary. If a collision is inevitable, the gap is clipped to 0.
// Parameters:
//   car: Pointer to the ego vehicle's data (speed, capabilities, length).
//   sa: Pointer to situational awareness data (lead vehicle details, initial distance).
// Returns:
//   The best-case gap (in meters) under maximum braking, or INFINITY if inputs are invalid
//   or just the current distance to the lead vehicle if no forward collision risk exists (e.g., ego reversing).
static Meters compute_best_case_braking_gap(const Car* car, const SituationalAwareness* sa) {
    // Validate inputs
    if (!car || !sa) {
        LOG_ERROR("NULL car or situational awareness.");
        return INFINITY;
    }
    if (!sa->lead_vehicle) {
        return INFINITY;  // No lead vehicle, no collision risk
    }

    // Constants and numerical safeguards
    const double EPS = 1e-6;  // Small threshold for floating-point comparisons

    // Initial conditions
    const Meters initial_gap = sa->distance_to_lead_vehicle - (car_get_length(car) + car_get_length(sa->lead_vehicle)) / 2.0;  // Initial bumper-to-bumper gap
    if (initial_gap <= 0) {
        return 0;  // Already colliding or overlapping
    }
    const MetersPerSecond ego_speed = car->speed;  // Ego vehicle speed (positive forward)
    const MetersPerSecond lead_speed = sa->lead_vehicle->speed;  // Lead vehicle speed (positive forward, negative if reversing)
    const MetersPerSecond relative_speed = lead_speed - ego_speed;  // Relative speed (lead relative to ego)
    MetersPerSecondSquared ego_accel = -car->capabilities.accel_profile.max_deceleration;  // Ego acceleration under maximum braking (negative)
    const MetersPerSecondSquared lead_accel = sa->lead_vehicle->acceleration;  // Lead vehicle acceleration
    const MetersPerSecondSquared relative_accel = lead_accel - ego_accel;  // Relative acceleration

    // Ensure valid ego deceleration (must be negative)
    if (ego_accel >= -EPS) {
        LOG_ERROR("Invalid max_deceleration: non-positive value %f", -ego_accel);
        ego_accel = -EPS;  // Fallback to small negative value
    }

    Meters min_gap;  // Best-case gap under maximum braking
    Seconds t_min;   // Time at which the best-case gap occurs

    if (ego_speed >= 0 && lead_speed >= 0) {
        // Both vehicles moving forward or stopped. The best-case gap occurs at:
        // - t=0 (if gap is increasing or constant),
        // - when relative speed becomes zero (if trajectory has a minimum and ego hasn't stopped),
        // - or when ego stops (if gap decreases until ego halts).
        // The lead stopping first doesn't change the minimum, as the gap continues to close until ego stops.
        const Seconds ego_stop_time = -ego_speed / ego_accel;  // Time when ego speed reaches zero

        if (relative_speed >= 0 && relative_accel > EPS) {
            // Ego slower than lead, braking harder: gap increases (convex trajectory). Minimum at t=0.
            t_min = 0;
        } else if (relative_speed <= 0 && relative_accel > EPS) {
            // Ego faster than lead, braking harder: convex trajectory with minimum at relative speed zero.
            t_min = -relative_speed / relative_accel;  // Time when relative speed is zero
            if (ego_speed + ego_accel * t_min < -EPS) {
                // If ego would have stopped before t_min, use ego stopping time instead
                t_min = ego_stop_time;
            }
        } else if (relative_speed <= 0 && relative_accel < -EPS) {
            // Ego faster, braking softer: gap decreases (concave trajectory). Minimum at ego stop.
            t_min = ego_stop_time;
        } else if (relative_speed >= 0 && relative_accel < -EPS) {
            // Ego slower, lead braking harder: concave trajectory with a maximum at relative speed zero. Minimum either at t=0 or at ego stop (compared later).
            t_min = ego_stop_time;
        } else {
            // No relative acceleration: gap changes linearly
            if (relative_speed >= 0) {
                // Gap constant or increasing: minimum at t=0
                t_min = 0;
            } else {
                // Gap decreasing: minimum at ego stop
                t_min = ego_stop_time;
            }
        }

        // Compute distances traveled by each vehicle until t_min
        const Meters ego_distance = ego_speed * t_min + 0.5 * ego_accel * t_min * t_min;  // Ego displacement
        Meters lead_distance = lead_speed * t_min + 0.5 * lead_accel * t_min * t_min;  // Lead displacement
        if (lead_speed + lead_accel * t_min < -EPS) {
            // Lead would have negative speed: cap at its stopping distance (requires lead_accel < 0)
            if (fabs(lead_accel) > EPS) {
                lead_distance = -lead_speed * lead_speed / (2 * lead_accel);
            } else {
                lead_distance = 0;  // Avoid division by zero; treat as stopped
            }
        }
        min_gap = initial_gap - (ego_distance - lead_distance);

        // For concave trajectories with a maximum, check if t=0 gives a smaller gap
        if (initial_gap < min_gap) {
            t_min = 0;
            min_gap = initial_gap;
        }
    } else if (ego_speed >= 0 && lead_speed < -EPS) {
        // Ego moving forward, lead reversing. Best-case gap occurs when both stop,
        // assuming both decelerate; otherwise, collision is inevitable.
        if (lead_accel <= EPS) {
            // Lead not decelerating (accelerating backward or constant speed): collision inevitable
            min_gap = 0;
        } else {
            // Both decelerating: compute gap after both stop
            const Meters ego_distance = -ego_speed * ego_speed / (2 * ego_accel);  // Ego stopping distance (positive)
            Meters lead_distance;
            if (fabs(lead_accel) > EPS) {
                lead_distance = -lead_speed * lead_speed / (2 * lead_accel);  // Lead stopping distance (negative)
            } else {
                lead_distance = 0;  // Avoid division; treat as stopped
            }
            min_gap = initial_gap - (ego_distance - lead_distance);
        }
    } else {
        // Ego reversing: no forward collision risk (reversing collisions not handled here)
        min_gap = initial_gap;
    }

    // Ensure non-negative gap; clip to 0 if collision indicated
    min_gap = fmax(min_gap, 0.0);

    return min_gap;
}

bool driving_assistant_control_car(DrivingAssistant* das, Car* car, Simulation* sim) {
    if (!car || !sim || !das) {
        LOG_ERROR("Invalid car, simulation, or driving assistant provided to driving_assistant_control_car");
        return false; // Indicating error due to invalid input
    }

    // Car control variables to determine:

    MetersPerSecondSquared accel = 0;
    CarIndicator lane_indicator = INDICATOR_NONE;
    CarIndicator turn_indicator = INDICATOR_NONE;
    bool request_lane_change = false;
    bool request_turn = false;

    situational_awareness_build(sim, car->id);  // ensure up-to-date situational awareness
    SituationalAwareness* sa = sim_get_situational_awareness(sim, car_get_id(car));


    // ---------- Verify follow control and stop control requisites ----------------------

    if (das->follow_mode && das->stop_mode) {
        LOG_ERROR("Cannot enable both follow mode and stop mode simultaneously for car %d. Disabling follow mode.", car->id);
        das->follow_mode = false; // Disable follow mode if both are enabled
    }

    if (das->follow_mode) {
        if (das->speed_target < 0) {
            LOG_WARN("Disabling follow mode for car %d due to negative speed target.", car->id);
            das->follow_mode = false;
        }
    }
    
    if (das->stop_mode) {
        if (car_get_speed(car) < -1e-9) {
            LOG_WARN("Disabling stop mode for car %d since the car is reversing. Speed = %.2f mph", car->id, to_mph(car_get_speed(car)));
            das->stop_mode = false;
            das->stopping_distance_target = 0; // Reset stopping distance target
        }
        das->speed_target = 0; // Stop mode always implies speed target = 0
    }



    // ---- AEB logic overrides everything else ----

    // auto engagement conditions
    bool execute_aeb = false;
    Meters best_case_braking_gap = compute_best_case_braking_gap(car, sa);
    if (das->aeb_in_progress || (das->aeb_assistance && car->speed >= AEB_ENGAGE_MIN_SPEED_MPS && best_case_braking_gap < AEB_ENGAGE_BEST_CASE_BRAKING_GAP_M)) {
        execute_aeb = !das->aeb_manually_disengaged;
    }

    // execute AEB
    if (execute_aeb) {
        if (!das->aeb_in_progress) {
            das->aeb_in_progress = true; // Start AEB if not already in progress
            LOG_DEBUG("AEB engaged for car %d. Best-case gap: %.2f feet, speed: %.2f mph", car->id, to_feet(best_case_braking_gap), to_mph(car_get_speed(car)));
        }
        accel = car_compute_acceleration_stop_max_brake(car, das->use_preferred_accel_profile);
        LOG_TRACE("AEB in progress for car %d. Best-case gap: %.2f feet, speed: %.2f mph", car->id, to_feet(best_case_braking_gap), to_mph(car_get_speed(car)));
        // NOTE: Forward crash is still possible if the lead vehicle is backing into us, but we don't handle that case here since that is not about emergency braking.
    }





    // ----------- Follow mode ----------------------
    if (das->follow_mode && !das->aeb_in_progress) {
        const Car* lead_car = sa->lead_vehicle;
        if (lead_car) { 
            accel = car_compute_acceleration_adaptive_cruise(car, sa, das->speed_target, das->follow_mode_thw, das->follow_mode_buffer, das->use_preferred_accel_profile);
        } else {
            accel = car_compute_acceleration_cruise(car, das->speed_target, das->use_preferred_accel_profile);
        }
    }

    // ----------- Stop mode ------------------------
    if (das->stop_mode && !das->aeb_in_progress) {
        Meters position_target = car_get_lane_progress_meters(car) + das->stopping_distance_target;
        MetersPerSecond speed_target = das->speed_target;   // which is always 0 in stop mode
        Meters speed_limit = car_get_speed(car);            // In stop mode, we don't want to exceed current speed
        if (speed_limit < from_mph(5)) {
            speed_limit = from_mph(5); // Allow speeding up to 5 mph (if already below that speed) to enable creeping.
        }
        accel = car_compute_acceleration_chase_target(car, position_target, speed_target, 0, speed_limit, das->use_preferred_accel_profile);
    }

    // ---------- Standard speed/cruise control ----------------------
    if (!das->follow_mode && !das->stop_mode && !das->aeb_in_progress) {
        accel = das->use_linear_speed_control ? car_compute_acceleration_adjust_speed_linear(car, das->speed_target, das->use_preferred_accel_profile) : car_compute_acceleration_adjust_speed(car, das->speed_target, das->use_preferred_accel_profile);
    }


    // ----------- Merge ----------------------


    // first of all, if merge is not possible, cancel the merge intent and reset current indicator and requests
    if ((das->merge_intent == INDICATOR_LEFT && !sa->is_lane_change_left_possible) || (das->merge_intent == INDICATOR_RIGHT && !sa->is_lane_change_right_possible && !sa->is_exit_available_eventually)) {
        LOG_DEBUG("Disabling merge intent for car %d due to impossibility.", car->id);
        das->merge_intent = INDICATOR_NONE; // Reset merge intent if the lane change is impossible
        lane_indicator = INDICATOR_NONE; // Reset lane indicator
        request_lane_change = false; // Reset lane change request
    } else {
        // set it to what merge intent says
        lane_indicator = das->merge_intent;
        // if merge assistance is active, issue request only when it is safe, else issue right away
        if (das->merge_assistance && lane_indicator != INDICATOR_NONE) {
            if (car_is_lane_change_dangerous(car, sim, sa, lane_indicator, das->follow_mode_thw, das->follow_mode_buffer)) {
                LOG_TRACE("Waiting for safe merge for car %d with indicator %d", car->id, lane_indicator);
                request_lane_change = false;        // Keep indicating but don't execute
            } else {
                request_lane_change = true; // Safe to change lane
            }
        } else {
            // Unless we are indicating in advance for a highway exit, request right away even if unsafe as long as it is possible to do so.
            request_lane_change = (lane_indicator == INDICATOR_LEFT && sa->is_lane_change_right_possible) || (lane_indicator == INDICATOR_RIGHT && sa->is_lane_change_left_possible);
        }
    }


    // --------- Turn -------------------

    // First of all, if the turn intent is not feasible, cancel it and reset stuff
    if ((das->turn_intent == INDICATOR_LEFT && !sa->is_turn_possible[INDICATOR_LEFT]) || (das->turn_intent == INDICATOR_RIGHT && !sa->is_turn_possible[INDICATOR_RIGHT])) {
        LOG_DEBUG("Disabling turn intent for car %d due to no turn possibility.", car->id);
        das->turn_intent = INDICATOR_NONE;  // Reset turn intent if impossible
        turn_indicator = INDICATOR_NONE;    // Reset turn indicator
        request_turn = false; // Reset turn request
    } else {
        // Keep requesting turn in the direction of turn intent. Reset turn intent later once turn successful.
        turn_indicator = das->turn_intent;
        request_turn = (turn_indicator != INDICATOR_NONE);
    }

    // ---- Apply control commands ----

    car_set_acceleration(car, accel);
    car_set_indicator_lane(car, lane_indicator);
    car_set_indicator_turn(car, turn_indicator);
    car_set_request_indicated_lane(car, request_lane_change);
    car_set_request_indicated_turn(car, request_turn);
    car_set_auto_turn_off_indicators(car, true); // Needed to detect successful lane/turn changes

    return true; // Indicating successful control without error
}


void driving_assistant_post_sim_step(DrivingAssistant* das, Car* car, Simulation* sim) {
    if (!car || !sim || !das) {
        LOG_ERROR("Invalid car, simulation, or driving assistant provided to driving_assistant_post_step");
        return; // Early return if input validation fails
    }
    situational_awareness_build(sim, car_get_id(car)); // Ensure situational awareness is up-to-date

    if (das->aeb_in_progress && !das->follow_mode && !das->stop_mode) {
        das->speed_target = car_get_speed(car); // Update speed target to current speed if AEB is in progress, so that once AEB is disengaged (automatic or manual), the car will not accelerate or decelerate unexpectedly as it enters speed-only control mode.
    }

    if (das->stop_mode) {
        das->stopping_distance_target -= car_get_recent_forward_movement(car); // Update stopping distance target based on recent forward movement
        if (das->stopping_distance_target < 0) {
            das->stopping_distance_target = 0; // Clamp to 0 if overshot
        }
        if (fabs(car_get_speed(car)) < AEB_DISENGAGE_MAX_SPEED_MPS) {
            // Car is fully stopped
            das->stop_mode = false; // Auto disengage stop mode once fully stopped
            LOG_DEBUG("Stop mode disengaged for car %d after coming to a full stop. Speed = %.2f. Residual stopping distance target = %.2f feet", car->id, to_mph(car_get_speed(car)), to_feet(das->stopping_distance_target));
            das->stopping_distance_target = 0; // Reset stopping distance target
            das->speed_target = 0; // Ensure speed target is 0 so that the car continues to stop to 0.00 speed in the speed-only mode.
        }
    }

    Meters best_case_braking_gap = compute_best_case_braking_gap(car, sim_get_situational_awareness(sim, car_get_id(car)));

    // auto disengagement conditions
    bool disengagement_conditions_met = false;
    if (best_case_braking_gap > AEB_DISENGAGE_BEST_CASE_BRAKING_GAP_M || car_get_speed(car) < AEB_DISENGAGE_MAX_SPEED_MPS) {
        disengagement_conditions_met = true;
    }

    if (disengagement_conditions_met) {
        if (das->aeb_in_progress) {
            LOG_DEBUG("AEB disengaged for car %d. Best case braking gap: %.2f feet, speed: %.2f mph", car->id, to_feet(best_case_braking_gap), to_mph(car_get_speed(car)));
        }
        das->aeb_in_progress = false; // Reset AEB state if auto disengagement conditions are met
        das->aeb_manually_disengaged = false; // Reset manual disengagement state if auto disengagement occurs
    }

    // reset merge intent once merge successful. Detected by checking mismatch between merge_intent and current lane indicator, assuming car has auto-turn off indicator enabled
    if (das->merge_intent != car_get_indicator_lane(car)) {
        assert((car_get_indicator_lane(car) == INDICATOR_NONE && !car_get_request_indicated_lane(car)) && "How did this happen?");
        das->merge_intent = INDICATOR_NONE; // Reset merge intent
        LOG_DEBUG("Merge intent reset for car %d after successful merge.", car->id);
    }

    // reset turn intent once turn successful. Detected by checking mismatch between turn_intent and current turn indicator, assuming car has auto-turn off indicator enabled
    if (das->turn_intent != car_get_indicator_turn(car)) {
        assert((car_get_indicator_turn(car) == INDICATOR_NONE && !car_get_request_indicated_turn(car)) && "How did this happen?");
        das->turn_intent = INDICATOR_NONE; // Reset turn intent
        LOG_DEBUG("Turn intent reset for car %d after successful turn.", car->id);
    }
}



// Getters

Seconds driving_assistant_get_last_configured_at(const DrivingAssistant* das) { return das->last_configured_at; }
MetersPerSecond driving_assistant_get_speed_target(const DrivingAssistant* das) { return das->speed_target; }
bool driving_assistant_get_stop_mode(const DrivingAssistant* das) { return das->stop_mode; }
Meters driving_assistant_get_stopping_distance_target(const DrivingAssistant* das) { return das->stopping_distance_target; }
CarIndicator driving_assistant_get_merge_intent(const DrivingAssistant* das) { return das->merge_intent; }
CarIndicator driving_assistant_get_turn_intent(const DrivingAssistant* das) { return das->turn_intent; }
bool driving_assistant_get_follow_mode(const DrivingAssistant* das) { return das->follow_mode; }
Meters driving_assistant_get_follow_mode_buffer(const DrivingAssistant* das) { return das->follow_mode_buffer; }
Seconds driving_assistant_get_follow_mode_thw(const DrivingAssistant* das) { return das->follow_mode_thw; }
bool driving_assistant_get_merge_assistance(const DrivingAssistant* das) { return das->merge_assistance; }
bool driving_assistant_get_aeb_assistance(const DrivingAssistant* das) { return das->aeb_assistance; }
bool driving_assistant_get_aeb_in_progress(const DrivingAssistant* das) { return das->aeb_in_progress; }
bool driving_assistant_get_aeb_manually_disengaged(const DrivingAssistant* das) { return das->aeb_manually_disengaged; }
bool driving_assistant_get_use_preferred_accel_profile(const DrivingAssistant* das) { return das->use_preferred_accel_profile; }
bool driving_assistant_get_use_linear_speed_control(const DrivingAssistant* das) { return das->use_linear_speed_control; }


// Setters (will update the timestamp)

static bool validate_input(const Car* car, const DrivingAssistant* das, const Simulation* sim) {
    if (!car || !das || !sim) {
        LOG_ERROR("Invalid car, driving assistant, or simulation provided");
        return false;
    }
    return true;
}


void driving_assistant_configure_speed_target(DrivingAssistant* das, const Car* car, const Simulation* sim, MetersPerSecond speed_target) {
    if (!validate_input(car, das, sim)) {
        return; // Early return if input validation fails
    }
    if (das->speed_target > car->capabilities.top_speed) {
        LOG_WARN("Speed target %.2f m/s exceeds car's top speed %.2f m/s for car %d. Setting to top speed.", speed_target, car->capabilities.top_speed, car->id);
        speed_target = car->capabilities.top_speed; // Ensure speed target does not exceed car's top speed
    }
    das->speed_target = speed_target;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_stop_mode(DrivingAssistant* das, const Car* car, const Simulation* sim, bool stop_mode) {
    if (!validate_input(car, das, sim)) {
        return; // Early return if input validation fails
    }
    if (stop_mode) {
        if (car_get_speed(car) < -1e-9) {
            LOG_ERROR("Stop mode cannot be enabled for car %d while reversing. Instead, set speed target to 0 in speed-only control mode to try to stop the car. This function will now return without changing anything.", car->id);
            return; // Do not enable stop mode if the car is reversing
        }
        das->follow_mode = false;               // Disable follow mode when stop mode is enabled
        das->speed_target = 0;                  // Set speed target to 0 when stop mode is enabled
        LOG_DEBUG("Stop mode engaged for car %d.", car->id);
    }
    das->stop_mode = stop_mode;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_stopping_distance_target(DrivingAssistant* das, const Car* car, const Simulation* sim, Meters stopping_distance_target) {
    if (!validate_input(car, das, sim)) {
        return; // Early return if input validation fails
    }
    if (stopping_distance_target < 0) {
        LOG_WARN("Stopping distance target cannot be negative for car %d. Setting it to 0.", car->id);
        stopping_distance_target = 0;
    }
    das->stopping_distance_target = stopping_distance_target;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_merge_intent(DrivingAssistant* das, const Car* car, const Simulation* sim, CarIndicator merge_intent) {
    if (!validate_input(car, das, sim)) {
        return; // Early return if input validation fails
    }
    das->merge_intent = merge_intent;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_turn_intent(DrivingAssistant* das, const Car* car, const Simulation* sim, CarIndicator turn_intent) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    das->turn_intent = turn_intent;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_follow_mode(DrivingAssistant* das, const Car* car, const Simulation* sim, bool follow_mode) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (follow_mode && das->speed_target < 0) {
        LOG_ERROR("Follow mode cannot be enabled for car %d with negative speed target.", car->id);
        return;
    }
    if (follow_mode) {
        das->stop_mode = false;                 // Disable stop mode when follow mode is enabled
        LOG_DEBUG("Follow mode engaged for car %d.", car->id);
    }
    das->follow_mode = follow_mode;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_follow_mode_buffer(DrivingAssistant* das, const Car* car, const Simulation* sim, Meters follow_mode_buffer) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (follow_mode_buffer < from_feet(MINIMUM_FOLLOW_MODE_BUFFER_FEET)) {
        LOG_WARN("Follow mode buffer (tried = %1.f feet) too small for car %d. Setting it to a minimum of %.1f feet.", car->id, to_feet(follow_mode_buffer), MINIMUM_FOLLOW_MODE_BUFFER_FEET);
        follow_mode_buffer = from_feet(MINIMUM_FOLLOW_MODE_BUFFER_FEET);
    }
    das->follow_mode_buffer = follow_mode_buffer;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_follow_mode_thw(DrivingAssistant* das, const Car* car, const Simulation* sim, Seconds follow_mode_thw) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (follow_mode_thw < 0) {
        LOG_WARN("Follow mode THW cannot be negative for car %d. Setting it to 0.", car->id);
        follow_mode_thw = 0;
    }
    das->follow_mode_thw = follow_mode_thw;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_merge_assistance(DrivingAssistant* das, const Car* car, const Simulation* sim, bool merge_assistance) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    das->merge_assistance = merge_assistance;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_aeb_assistance(DrivingAssistant* das, const Car* car, const Simulation* sim, bool aeb_assistance) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (!aeb_assistance) {
        das->aeb_in_progress = false; // Reset AEB state if AEB assistance is disabled
        das->aeb_manually_disengaged = false; // Reset manual disengagement state if AEB assistance is disabled
    }
    das->aeb_assistance = aeb_assistance;
    das->last_configured_at = sim_get_time(sim);
}


void driving_assistant_configure_use_preferred_accel_profile(DrivingAssistant* das, const Car* car, const Simulation* sim, bool use_preferred_accel_profile) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    das->use_preferred_accel_profile = use_preferred_accel_profile;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_use_linear_speed_control(DrivingAssistant* das, const Car* car, const Simulation* sim, bool use_linear_speed_control) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    das->use_linear_speed_control = use_linear_speed_control;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_aeb_manually_disengage(DrivingAssistant* das, const Car* car, const Simulation* sim) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (das->aeb_in_progress) {
        LOG_DEBUG("Manually disengaging AEB for car %d", car->id);
        das->aeb_in_progress = false;
    } else {
        LOG_TRACE("%s for car %d. Nothing more to disengage.", das->aeb_manually_disengaged ? "AEB already manually disengaged" : "AEB is not in progress", car->id);
        return;
    }
    das->aeb_manually_disengaged = true;
    das->last_configured_at = sim_get_time(sim);
    LOG_DEBUG("AEB manually disengaged for car %d", car->id);
}
