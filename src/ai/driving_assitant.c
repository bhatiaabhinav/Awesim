#include "ai.h"
#include "logging.h"
#include <assert.h>

static const double MINIMUM_FOLLOW_MODE_BUFFER_FEET = 1; // Minimum follow mode buffer to avoid unrealistic behavior
static const Seconds MINIMUM_FOLLOW_MODE_THW = 0.2; // Minimum follow mode time headway to avoid unrealistic behavior
static const Seconds MINIMUM_MERGE_MIN_THW = 0.1; // Minimum merge minimum time headway to avoid unrealistic behavior
static const Seconds MINIMUM_AEB_MIN_THW = 0.1; // Minimum AEB minimum time headway to avoid unrealistic behavior
static const Seconds MAXIMUM_AEB_MIN_THW = 1.0; // Maximum AEB minimum time headway to avoid unrealistic behavior
static const double DEFAULT_FOLLOW_MODE_BUFFER_FEET = 2; // Default follow mode buffer in feet
static const Seconds DEFAULT_TIME_HEADWAY = 3.0; // Default time headway for merging and AEB
static const Seconds DEFAULT_AEB_MIN_THW = 0.5; // Default minimum time headway for AEB


bool driving_assistant_reset_settings(DrivingAssistant* das, Car* car) {
    if (!car || !das) {
        LOG_ERROR("Invalid car or driving assistant provided to driving_assistant_reset_settings");
        return false; // Indicating error due to invalid input
    }

    // Resetting all settings to default values
    das->last_configured_at = 0;
    das->speed_target = 0.0;
    das->PD_mode = false;
    das->position_error = 0.0;
    das->merge_intent = INDICATOR_NONE;
    das->turn_intent = INDICATOR_NONE;
    das->follow_mode = false;
    das->follow_mode_buffer = from_feet(DEFAULT_FOLLOW_MODE_BUFFER_FEET);
    das->follow_mode_thw = DEFAULT_TIME_HEADWAY;      // Default time headway
    das->merge_assistance = false;   // Default to false
    das->aeb_assistance = false;     // Default to false
    das->merge_min_thw = DEFAULT_TIME_HEADWAY;        // Default minimum THW for merging
    das->aeb_min_thw = DEFAULT_AEB_MIN_THW;          // Default minimum THW for AEB
    das->feasible_thw = INFINITY;    // Reset feasible THW
    das->aeb_in_progress = false;    // Reset AEB state
    das->aeb_manually_disengaged = false; // Reset manual disengagement state
    das->use_preferred_accel_profile = false; // Reset preferred acceleration profile state

    LOG_TRACE("Driving assistant settings reset for car %d", car->id);
    return true; // Indicating success
}


static Seconds _compute_feasible_thw(const Car* car, const SituationalAwareness* sa) {
    if (!car || !sa) {
        LOG_ERROR("Invalid car or situational awareness provided to _compute_feasible_thw");
        return INFINITY; // Return infinity if inputs are invalid
    }

    Seconds feasible_thw = INFINITY;

    if (sa->lead_vehicle && car->speed > sa->lead_vehicle->speed && car->speed > 0) {
        MetersPerSecond v_relative = car->speed - sa->lead_vehicle->speed;
        if (v_relative > 0) {
            // suppose we break hard, how far will we go before stopping in the lead's frame of reference?
            MetersPerSecondSquared decel_max = car->capabilities.accel_profile.max_deceleration;
            Meters braking_distance_rel = (v_relative * v_relative) / (2 * decel_max); // TODO: incorporate the lead car's acceleration into this calculation.
            // Once we match the lead's speed, how far would we be from it?
            Meters distance_to_lead_vehicle_feasible = sa->distance_to_lead_vehicle - braking_distance_rel;
            // Subtract sum of car half lengths
            distance_to_lead_vehicle_feasible -= (car_get_length(car) + car_get_length(sa->lead_vehicle)) / 2.0;
            // how much is this in headway seconds?
            feasible_thw = distance_to_lead_vehicle_feasible / sa->lead_vehicle->speed;
        }
    }

    return feasible_thw;
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


    // ---------- Verify follow control and PD control requisites ----------------------

    if (das->follow_mode) {
        if (das->speed_target < 0) {
            LOG_DEBUG("Disabling follow mode for car %d due to negative speed target.", car->id);
            das->follow_mode = false;
        }
    }
    
    if (das->PD_mode) {
        if (das->follow_mode) {
            das->PD_mode = false; // Disable PD mode if follow mode is active. The code should never reach here since it was already ensured in setters that the modes are mutually exclusive.
        }
        if (fabs(das->position_error) < from_centimeters(1)) {
            LOG_DEBUG("Position error is negligible for car %d. Switching to standard speed control.", car->id);
            das->PD_mode = false; // Disable PD mode if position error is negligible
            das->position_error = 0.0; // Reset position error
        }
    }



    // ---- AEB logic overrides everything else ----

    das->feasible_thw = _compute_feasible_thw(car, sa);

    // auto engagement conditions
    bool execute_aeb = false;
    if (das->aeb_assistance && (das->feasible_thw < das->aeb_min_thw || das->aeb_in_progress)) {
        // AEB should be engaged if feasible THW is below the threshold or if AEB is already in progress, unless manually disengaged.
        execute_aeb = !das->aeb_manually_disengaged;
    }

    // execute AEB
    if (execute_aeb) {
        if (!das->aeb_in_progress) {
            das->aeb_in_progress = true; // Start AEB if not already in progress
            LOG_DEBUG("AEB engaged for car %d. Feasible THW: %.2f seconds", car->id, das->feasible_thw);
        }
        accel = car_compute_acceleration_stop_max_brake(car, das->use_preferred_accel_profile);
        LOG_TRACE("AEB in progress for car %d. Feasible THW: %.2f seconds", car->id, das->feasible_thw);
        // NOTE: Forward crash is still possible if the lead vehicle is backing into us, but we don't handle that case here since that is not about emergency braking.

        // If AEB is on, PD mode should be turned off
        das->PD_mode = false;
        das->position_error = 0.0; // Reset position error since PD mode is off
    }





    // ----------- Follow mode ----------------------
    if (das->follow_mode) {
        const Car* lead_car = sa->lead_vehicle;
        if (lead_car) { 
            accel = car_compute_acceleration_adaptive_cruise(car, sa, das->speed_target, das->follow_mode_thw, das->follow_mode_buffer, das->use_preferred_accel_profile);
        } else {
            accel = car_compute_acceleration_cruise(car, das->speed_target, das->use_preferred_accel_profile);
        }
    }

    // ----------- PD control ------------------------
    if (das->PD_mode) {
        // Use PD control to adjust both speed and position
        Meters position_target = car_get_lane_progress_meters(car) - das->position_error; // Target position is current position minus the position error (which is the error in target position's frame of reference)
        Meters overshoot_buffer = from_feet(0);
        MetersPerSecond speed_target = das->speed_target;
        Meters speed_limit = fabs(speed_target);
        accel = car_compute_acceleration_chase_target(car, position_target, speed_target, overshoot_buffer, speed_limit, das->use_preferred_accel_profile);
    }

    // ---------- Standard speed/cruise control ----------------------
    if (!das->follow_mode && !das->PD_mode && !das->aeb_in_progress) {
        accel = car_compute_acceleration_adjust_speed(car, das->speed_target, das->use_preferred_accel_profile);
    }


    // ----------- Merge ----------------------


    // first of all, if merge is not possible, cancel the merge intent and reset current indicator and requests
    if ((das->merge_intent == INDICATOR_LEFT && !sa->is_lane_change_left_possible) || (das->merge_intent == INDICATOR_RIGHT && !sa->is_lane_change_right_possible)) {
        LOG_DEBUG("Disabling merge intent for car %d due to impossibility.", car->id);
        das->merge_intent = INDICATOR_NONE; // Reset merge intent if the lane change is impossible
        lane_indicator = INDICATOR_NONE; // Reset lane indicator
        request_lane_change = false; // Reset lane change request
    } else {
        // set it to what merge intent says
        lane_indicator = das->merge_intent;
        // if merge assistance is active, issue request only when it is safe, else issue right away
        if (das->merge_assistance && lane_indicator != INDICATOR_NONE) {
            if (car_is_lane_change_dangerous(car, sim, sa, lane_indicator, das->merge_min_thw)) {
                LOG_TRACE("Waiting for safe merge for car %d with indicator %d", car->id, lane_indicator);
                lane_indicator = INDICATOR_NONE; // Disengage lane change if it's dangerous
            } else {
                request_lane_change = true; // Safe to change lane
            }
        } else {
            // Request right away even if unsafe
            request_lane_change = (lane_indicator != INDICATOR_NONE);
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

    if (das->aeb_in_progress && !das->follow_mode) {
        das->speed_target = car_get_speed(car); // Update speed target to current speed if AEB is in progress, so that once AEB is disengaged (automatic or manual), the car will not accelerate or decelerate unexpectedly.
    }

    if (das->PD_mode) {
        das->position_error += car_get_recent_forward_movement(car); // Update position error based on recent forward movement
        if (fabs(das->position_error) < from_centimeters(1)) {
            // If position error is negligible, we can use standard speed control
            LOG_DEBUG("Position error is negligible for car %d. Switching to standard speed control.", car->id);
            das->PD_mode = false; // Disable PD mode if position error is negligible
            das->position_error = 0.0; // Reset position error
        }
    }

    das->feasible_thw = _compute_feasible_thw(car, sim_get_situational_awareness(sim, car_get_id(car)));

    // auto disengagement conditions
    bool disengagement_conditions_met = false;
    if (das->feasible_thw > 1.5 * das->aeb_min_thw) {
        disengagement_conditions_met = true; // AEB should be disengaged if feasible THW is 50% above the trigger threshold.
    }

    if (disengagement_conditions_met) {
        if (das->aeb_in_progress) {
            LOG_DEBUG("AEB disengaged for car %d. Feasible THW: %.2f seconds", car->id, das->feasible_thw);
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
bool driving_assistant_get_PD_mode(const DrivingAssistant* das) { return das->PD_mode; }
Meters driving_assistant_get_position_error(const DrivingAssistant* das) { return das->position_error; }
CarIndicator driving_assistant_get_merge_intent(const DrivingAssistant* das) { return das->merge_intent; }
CarIndicator driving_assistant_get_turn_intent(const DrivingAssistant* das) { return das->turn_intent; }
bool driving_assistant_get_follow_mode(const DrivingAssistant* das) { return das->follow_mode; }
Meters driving_assistant_get_follow_mode_buffer(const DrivingAssistant* das) { return das->follow_mode_buffer; }
Seconds driving_assistant_get_follow_mode_thw(const DrivingAssistant* das) { return das->follow_mode_thw; }
bool driving_assistant_get_merge_assistance(const DrivingAssistant* das) { return das->merge_assistance; }
bool driving_assistant_get_aeb_assistance(const DrivingAssistant* das) { return das->aeb_assistance; }
Seconds driving_assistant_get_merge_min_thw(const DrivingAssistant* das) { return das->merge_min_thw; }
Seconds driving_assistant_get_aeb_min_thw(const DrivingAssistant* das) { return das->aeb_min_thw; }
Seconds driving_assistant_get_feasible_thw(const DrivingAssistant* das) { return das->feasible_thw; }
bool driving_assistant_get_aeb_in_progress(const DrivingAssistant* das) { return das->aeb_in_progress; }
bool driving_assistant_get_aeb_manually_disengaged(const DrivingAssistant* das) { return das->aeb_manually_disengaged; }
bool driving_assistant_get_use_preferred_accel_profile(const DrivingAssistant* das) { return das->use_preferred_accel_profile; }


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

void driving_assistant_configure_PD_mode(DrivingAssistant* das, const Car* car, const Simulation* sim, bool PD_mode) {
    if (!validate_input(car, das, sim)) {
        return; // Early return if input validation fails
    }
    das->PD_mode = PD_mode;
    if (PD_mode) {
        das->follow_mode = false; // Disable follow mode when PD mode is enabled
    } else {
        das->position_error = 0.0; // Reset position error when PD mode is disabled
    }
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_position_error(DrivingAssistant* das, const Car* car, const Simulation* sim, Meters position_error) {
    if (!validate_input(car, das, sim)) {
        return; // Early return if input validation fails
    }
    if (das->PD_mode) {
        das->position_error = position_error;
    } else {
        LOG_WARN("Position error can only be configured when PD mode is enabled for car %d", car->id);
        das->position_error = 0.0; // Reset position error if PD mode is not enabled. Just for good measure.
        return; // Position error can only be set in PD mode
    }
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
        das->PD_mode = false;       // Disable PD mode when follow mode is enabled
        das->position_error = 0.0;  // Reset position error when follow mode is enabled
    }
    das->follow_mode = follow_mode;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_follow_mode_buffer(DrivingAssistant* das, const Car* car, const Simulation* sim, Meters follow_mode_buffer) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (follow_mode_buffer < from_feet(MINIMUM_FOLLOW_MODE_BUFFER_FEET)) {
        LOG_WARN("Follow mode buffer too small for car %d. Setting it to a minimum of %.1f feet.", car->id, MINIMUM_FOLLOW_MODE_BUFFER_FEET);
        follow_mode_buffer = from_feet(MINIMUM_FOLLOW_MODE_BUFFER_FEET);
    }
    das->follow_mode_buffer = follow_mode_buffer;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_follow_mode_thw(DrivingAssistant* das, const Car* car, const Simulation* sim, Seconds follow_mode_thw) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (follow_mode_thw < MINIMUM_FOLLOW_MODE_THW) {
        LOG_WARN("Follow mode THW too small for car %d. Setting it to a minimum of %.1f seconds.", car->id, MINIMUM_FOLLOW_MODE_THW);
        follow_mode_thw = MINIMUM_FOLLOW_MODE_THW;
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

void driving_assistant_configure_merge_min_thw(DrivingAssistant* das, const Car* car, const Simulation* sim, Seconds merge_min_thw) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (merge_min_thw < MINIMUM_MERGE_MIN_THW) {
        LOG_WARN("Merge minimum THW too small for car %d. Setting it to a minimum of %.1f seconds.", car->id, MINIMUM_MERGE_MIN_THW);
        merge_min_thw = MINIMUM_MERGE_MIN_THW; // Set a minimum THW to avoid unrealistic behavior
    }
    das->merge_min_thw = merge_min_thw;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_aeb_min_thw(DrivingAssistant* das, const Car* car, const Simulation* sim, Seconds aeb_min_thw) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (aeb_min_thw < MINIMUM_AEB_MIN_THW) {
        LOG_WARN("AEB minimum THW too small for car %d. Setting it to a minimum of %.1f seconds.", car->id, MINIMUM_AEB_MIN_THW);
        aeb_min_thw = MINIMUM_AEB_MIN_THW; // Set a minimum THW to avoid unrealistic behavior
    }
    if (aeb_min_thw > MAXIMUM_AEB_MIN_THW) {
        LOG_WARN("AEB minimum THW too large for car %d. Setting it to a maximum of %.1f seconds.", car->id, MAXIMUM_AEB_MIN_THW);
        aeb_min_thw = MAXIMUM_AEB_MIN_THW; // Set a maximum THW to avoid unrealistic behavior
    }
    das->aeb_min_thw = aeb_min_thw;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_use_preferred_accel_profile(DrivingAssistant* das, const Car* car, const Simulation* sim, bool use_preferred_accel_profile) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    das->use_preferred_accel_profile = use_preferred_accel_profile;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_aeb_manually_disengage(DrivingAssistant* das, const Car* car, const Simulation* sim) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (das->aeb_in_progress) {
        LOG_DEBUG("Manually disengaging AEB for car %d", car->id);
    } else {
        LOG_TRACE("%s for car %d. Nothing more to disengage.", das->aeb_manually_disengaged ? "AEB already manually disengaged" : "AEB is not in progress", car->id);
        return;
    }
    das->aeb_manually_disengaged = true;
    das->last_configured_at = sim_get_time(sim);
    LOG_DEBUG("AEB manually disengaged for car %d", car->id);
}
