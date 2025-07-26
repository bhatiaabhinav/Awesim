#include "ai.h"
#include "logging.h"


bool driving_assistant_control_car(DrivingAssistant das, Car* car, Simulation* sim) {
    if (!car || !sim) {
        LOG_ERROR("Invalid car or simulation provided to driving_assistant_control_car");
        return false; // Indicating error due to invalid input
    }


    // Car control variables to determine:

    MetersPerSecondSquared accel;
    CarIndictor lane_indicator;
    CarIndictor turn_indicator;
    bool request_lane_change;
    bool request_turn;

    situational_awareness_build(sim, car->id);  // ensure up-to-date situational awareness
    SituationalAwareness* sa = sim_get_situational_awareness(sim, car_get_id(car));

    if (das.cruise_mode) {
        const Car* lead_car = sa->lead_vehicle;
        if (das.follow_mode && lead_car) {
            accel = car_compute_acceleration_adaptive_cruise(car, sa, das.cruise_speed, das.follow_mode_thw, das.follow_mode_buffer, das.use_preferred_accel_profile);
        } else {
            accel = car_compute_acceleration_cruise(car, das.cruise_speed, das.use_preferred_accel_profile);
        }
    } else {
        // If not in cruise mode, use PD control.
        if (das.PD_mode) {
            // Use PD control to adjust both speed and position
            Meters position_target = das.position_target;
            Meters overshoot_buffer = from_feet(0);
            MetersPerSecond speed_target = das.speed_target;
            Meters speed_limit = fmax(das.cruise_speed, fabs(speed_target));
            accel = car_compute_acceleration_chase_target(car, position_target, speed_target, overshoot_buffer, speed_limit, das.use_preferred_accel_profile);
        } else {
            // Use standard speed control
            accel = car_compute_acceleration_adjust_speed(car, das.speed_target, das.use_preferred_accel_profile);
        }
    }

    lane_indicator = das.merge_intent;
    request_lane_change = false;
    if (das.merge_assistance && lane_indicator != INDICATOR_NONE) {
        if (car_is_lane_change_dangerous(car, sim, sa, lane_indicator, das.merge_min_thw)) {
            LOG_TRACE("Waiting for safe lane change for car %d with indicator %d", car->id, lane_indicator);
            lane_indicator = INDICATOR_NONE; // Disengage lane change if it's dangerous
        } else {
            request_lane_change = true; // Safe to change lane
        }
    } else {
        request_lane_change = (lane_indicator != INDICATOR_NONE);
    }

    turn_indicator = das.turn_intent;
    request_turn = (turn_indicator != INDICATOR_NONE);

    if (request_lane_change) {
        das.merge_intent = INDICATOR_NONE; // Reset intent after issuing command
    }
    if (request_turn) {
        das.turn_intent = INDICATOR_NONE; // Reset intent after issuing command
    }


    // ---- AEB logic ----

    das.feasible_thw = INFINITY;
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
            das.feasible_thw = distance_to_lead_vehicle_feasible / sa->lead_vehicle->speed;
        }
    }

    // auto engagement conditions
    bool enagement_conditions_met;
    if (das.feasible_thw < das.aeb_min_thw || das.aeb_in_progress) {
        enagement_conditions_met = true; // AEB should be engaged if feasible THW is below the threshold or if AEB is already in progress.
    }
    // auto disengagement conditions
    bool disengagement_conditions_met;
    if (das.feasible_thw > 1.0) {
        disengagement_conditions_met = false; // AEB should be disengaged if feasible THW is greater than 1 second.
    }

    bool execute_aeb = false;
    if (enagement_conditions_met) {
        // unless manually disengaged, apply AEB
        execute_aeb = !das.aeb_manually_disengaged;
    }
    if (disengagement_conditions_met) {
        if (das.aeb_in_progress) {
            LOG_DEBUG("AEB disengaged for car %d. Feasible THW: %.2f seconds", car->id, das.feasible_thw);
        }
        das.aeb_in_progress = false; // Reset AEB state if auto disengagement conditions are met
        das.aeb_manually_disengaged = false; // Reset manual disengagement state if auto disengagement occurs
    }

    // execute AEB if in progress
    if (execute_aeb) {
        if (!das.aeb_in_progress) {
            das.aeb_in_progress = true; // Start AEB if not already in progress
            LOG_DEBUG("AEB engaged for car %d. Feasible THW: %.2f seconds", car->id, das.feasible_thw);
        }
        accel = car_compute_acceleration_stop_max_brake(car, das.use_preferred_accel_profile);
        LOG_TRACE("AEB in progress for car %d. Feasible THW: %.2f seconds", car->id, das.feasible_thw);
    }

    car_set_acceleration(car, accel);
    car_set_indicator_lane(car, lane_indicator);
    car_set_indicator_turn(car, turn_indicator);
    car_set_request_indicated_lane(car, request_lane_change);
    car_set_request_indicated_turn(car, request_turn);

    return true; // Indicating successful control without error
}