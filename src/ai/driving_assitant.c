#include "ai.h"
#include "logging.h"
#include "sim.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>


bool driving_assistant_reset_settings(DrivingAssistant* das, Car* car) {
    if (!car || !das) {
        LOG_ERROR("Invalid car or driving assistant provided to driving_assistant_reset_settings");
        return false; // Indicating error due to invalid input
    }

    // Resetting all settings to default values
    das->last_configured_at = 0;
    das->speed_target = 0.0;
    das->merge_intent = INDICATOR_NONE;
    das->turn_intent = INDICATOR_NONE;
    das->thw = DRIVING_ASSISTANT_DEFAULT_TIME_HEADWAY;      // Default time headway
    das->buffer = DRIVING_ASSISTANT_DEFAULT_CAR_DISTANCE_BUFFER; // Default distance buffer
    das->follow_assistance = true;  // Default to true
    das->merge_assistance = true;   // Default to true
    das->aeb_assistance = false;     // Default to false
    das->smart_das = true;          // Default to true
    das->smart_das_driving_style = SMART_DAS_DRIVING_STYLE_NORMAL; // Default driving style
    das->aeb_in_progress = false;    // Reset AEB state
    das->aeb_manually_disengaged = false; // Reset manual disengagement state
    das->smart_das_intersection_approach_engaged = false; // Reset smart DAS intersection approach state
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
static Meters compute_best_case_braking_gap(Simulation* sim, const Car* car, const SituationalAwareness* sa) {
    // Validate inputs
    if (!car || !sa) {
        LOG_ERROR("NULL car or situational awareness.");
        return INFINITY;
    }
    if (!sa->nearby_vehicles.lead) {
        return INFINITY;  // No lead vehicle, no collision risk
    }

    // Constants and numerical safeguards
    const double EPS = 1e-6;  // Small threshold for floating-point comparisons

    // Initial conditions
    Meters distance_to_lead;
    MetersPerSecond speed_lead;
    MetersPerSecondSquared accel_lead;
    perceive_lead_vehicle(car, sim, sa, &distance_to_lead, &speed_lead, &accel_lead);

    const Meters initial_gap = distance_to_lead  - (car_get_length(car) + car_get_length(sa->nearby_vehicles.lead)) / 2.0;  // Initial bumper-to-bumper gap
    if (initial_gap <= 0) {
        return 0;  // Already colliding or overlapping
    }
    const MetersPerSecond ego_speed = car->speed;  // Ego vehicle speed (positive forward)
    // Add perception noise to lead vehicle speed
    const MetersPerSecond lead_speed = fmax(0, speed_lead);  // Lead vehicle speed (positive forward, negative if reversing)
    const MetersPerSecond relative_speed = lead_speed - ego_speed;  // Relative speed (lead relative to ego)
    MetersPerSecondSquared ego_accel = -car->capabilities.accel_profile.max_deceleration;  // Ego acceleration under maximum braking (negative)
    const MetersPerSecondSquared lead_accel = accel_lead;  // Lead vehicle acceleration
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

static bool driving_assistant_smart_das_should_engage_intersection_approach(DrivingAssistant* das, Car* car, Simulation* sim, SituationalAwareness* situation) {
    if (!das->smart_das) {
        return false;
    }
    if (das->smart_das_intersection_approach_engaged) {
        return true; // already engaged
    }
    if (situation->is_an_intersection_upcoming) {
        Meters distance_to_stop_line = situation->distance_to_end_of_lane_from_leading_edge - STOP_LINE_BUFFER_METERS;
        // check if we need to engage now. Are we within the appropriate distance to start slowing down for stopping at intersection? braking distance situation->braking_distance.preferred_smooth, situation->braking_distance.preferred, situation->braking_distance.capable depending on SmartDASDrivingStyle.
        Meters braking_distances[SMART_DAS_DRIVING_STYLES_COUNT] = {
            [SMART_DAS_DRIVING_STYLE_RECKLESS] = situation->braking_distance.capable,
            [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = situation->braking_distance.preferred,
            [SMART_DAS_DRIVING_STYLE_NORMAL] = situation->braking_distance.preferred_smooth,
            [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = situation->braking_distance.preferred_smooth
        };
        Meters braking_distance_to_use = braking_distances[das->smart_das_driving_style];
        if (situation->speed < 0) {
            braking_distance_to_use = 0; // if reversing, no braking distance needed, which is originally positive even if speed is negative
        }
        return distance_to_stop_line <= 1.01 * braking_distance_to_use || situation->is_approaching_end_of_lane;
    }
    return false;
}

static MetersPerSecondSquared driving_assistant_smart_das_handle_intersection(DrivingAssistant* das, Car* car, Simulation* sim, SituationalAwareness* situation) {
    Map* map = sim_get_map(sim);
    MetersPerSecondSquared accel = car->capabilities.accel_profile.max_acceleration;
    if (situation->is_an_intersection_upcoming) {
        if (das->smart_das_intersection_approach_engaged) {
            Meters distance_to_stop_line = situation->distance_to_end_of_lane_from_leading_edge - STOP_LINE_BUFFER_METERS;
            TrafficLight light = situation->light_for_turn[das->turn_intent];
            bool is_sudden_emergency_braking_possible = (situation->braking_distance.capable < situation->distance_to_end_of_lane);
            bool is_sudden_semi_hard_braking_possible = (situation->braking_distance.preferred < situation->distance_to_end_of_lane);
            bool is_smooth_braking_possible = (situation->braking_distance.preferred_smooth < situation->distance_to_end_of_lane);
            if (situation->stopped || situation->reversing) {
                // if reversing, it's all safe
                is_sudden_emergency_braking_possible = true;
                is_sudden_semi_hard_braking_possible = true;
                is_smooth_braking_possible = true;
            }
            const Lane* next_lane = situation->lane_next_after_turn[das->turn_intent];
            MetersPerSecond next_lane_speed_limit = lane_get_speed_limit(next_lane);
            Meters next_lane_length = next_lane->length;
            Seconds yield_thw = das->thw + next_lane_length / next_lane_speed_limit + situation->distance_to_end_of_lane / situation->speed; // time headway to consider for yielding at intersection, with some buffer for traversing the intersection and for reaching the intersection

            bool should_brake_for_full_stop = false;
            bool should_brake_for_rolling_stop = false;
            bool should_brake_for_yield = false;
            bool should_speed_up_to_make_light = false;

            bool is_ok_with_yieldings[SMART_DAS_DRIVING_STYLES_COUNT] = {
                [SMART_DAS_DRIVING_STYLE_NO_RULES] = false,
                [SMART_DAS_DRIVING_STYLE_RECKLESS] = is_smooth_braking_possible,
                [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = is_sudden_semi_hard_braking_possible,
                [SMART_DAS_DRIVING_STYLE_NORMAL] = is_sudden_emergency_braking_possible,
                [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = is_sudden_emergency_braking_possible
            };
            bool should_brake_for_yellows[SMART_DAS_DRIVING_STYLES_COUNT] = {
                [SMART_DAS_DRIVING_STYLE_NO_RULES] = false,
                [SMART_DAS_DRIVING_STYLE_RECKLESS] = false,
                [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = is_smooth_braking_possible,
                [SMART_DAS_DRIVING_STYLE_NORMAL] = is_smooth_braking_possible,
                [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = is_smooth_braking_possible
            };
            bool like_to_speed_up_for_yellows[SMART_DAS_DRIVING_STYLES_COUNT] = {
                [SMART_DAS_DRIVING_STYLE_NO_RULES] = false,
                [SMART_DAS_DRIVING_STYLE_RECKLESS] = true,
                [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = true,
                [SMART_DAS_DRIVING_STYLE_NORMAL] = false,
                [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = false
            };

            switch (light)
            {
            case TRAFFIC_LIGHT_GREEN:
                // all false
                should_brake_for_full_stop = false;
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = false;
                should_speed_up_to_make_light = false;
                break;
            case TRAFFIC_LIGHT_GREEN_YIELD:
            if (car->id == 131) {
                // printf("Car %d: At intersection with GREEN-YIELD light. Distance to stop line: %.2f meters\n", car->id, distance_to_stop_line);
            }
                should_brake_for_full_stop = false;
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = is_ok_with_yieldings[das->smart_das_driving_style] && car_has_something_to_yield_to_at_intersection(car, sim, situation, das->turn_intent, yield_thw, sim_get_dt(sim));
                should_speed_up_to_make_light = false;
                break;
            case TRAFFIC_LIGHT_YELLOW:
                // note: we are not lawfully expected to hard brake on yellows in general. No driving style should do that.
                should_brake_for_full_stop = should_brake_for_yellows[das->smart_das_driving_style];
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = false;
                should_speed_up_to_make_light = !should_brake_for_full_stop && like_to_speed_up_for_yellows[das->smart_das_driving_style];
                break;
            case TRAFFIC_LIGHT_YELLOW_YIELD:
                // note: we are not lawfully expected to hard brake on yellows in general. No driving style should do that. Also, we may need to yield here in case we decide to go through the yellow. Also, we won't be speeding up to make the light here, because when combined with yielding, it messes up things.
                should_brake_for_full_stop = should_brake_for_yellows[das->smart_das_driving_style];
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = !should_brake_for_full_stop && is_ok_with_yieldings[das->smart_das_driving_style] && car_has_something_to_yield_to_at_intersection(car, sim, situation, das->turn_intent, yield_thw, sim_get_dt(sim));
                should_speed_up_to_make_light = false;
                break;
            case TRAFFIC_LIGHT_RED:
                should_brake_for_full_stop = true;
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = false;
                should_speed_up_to_make_light = false;
                break;
            case TRAFFIC_LIGHT_RED_YIELD: {
                // e.g., free right. first stop at stop line, then yield once you are beyond it
                bool yield_mode = distance_to_stop_line <= meters(0.1); // within 10 cm of stop line
                if (das->smart_das_driving_style == SMART_DAS_DRIVING_STYLE_DEFENSIVE) {
                    yield_mode = false; // defensive does not exercise red-yield and considers it as simple red
                }
                if (yield_mode) {
                    should_brake_for_full_stop = false;
                    should_brake_for_rolling_stop = false;
                    should_brake_for_yield = car_has_something_to_yield_to_at_intersection(car, sim, situation, das->turn_intent, yield_thw, sim_get_dt(sim));
                } else {
                    should_brake_for_full_stop = true;
                    should_brake_for_rolling_stop = false; 
                    should_brake_for_yield = false;
                    if (das->smart_das_driving_style == SMART_DAS_DRIVING_STYLE_RECKLESS) {
                        should_brake_for_full_stop = false;
                        should_brake_for_rolling_stop = true; // reckless does a rolling stop on red-yield
                    }
                }
                should_speed_up_to_make_light = false;
                break;
            }
            case TRAFFIC_LIGHT_STOP_FCFS: {
                // first come first serve stop sign
                // first, stop at stop line, then proceed when it's my turn
                bool fcfs_yield_mode = distance_to_stop_line <= meters(0.1); // within 10 cm of stop line
                bool all_clear = !intersection_is_any_car_on_intersection(situation->intersection, sim);
                if (fcfs_yield_mode) {
                    should_brake_for_rolling_stop = false;
                    if (!all_clear) {
                        // even those doing rolling stops should come to full stop when it's not clear
                        should_brake_for_full_stop = true;
                    } else {
                        if (das->smart_das_driving_style == SMART_DAS_DRIVING_STYLE_RECKLESS) {
                            // reckless did rolling stop at stop line, now, may stop fully if not clear and go if clear even if not their turn
                            should_brake_for_full_stop = false;
                        } else {
                            if (situation->is_my_turn_at_stop_sign_fcfs) {
                                should_brake_for_full_stop = false;
                            } else {
                                // even those doing rolling stops should come to full stop when it's not their turn or not clear
                                should_brake_for_full_stop = true;
                                
                            }
                        }
                    }
                } else {
                    // come to full stop or rolling stop at stop line
                    should_brake_for_full_stop = das->smart_das_driving_style == SMART_DAS_DRIVING_STYLE_NORMAL || das->smart_das_driving_style == SMART_DAS_DRIVING_STYLE_DEFENSIVE;
                    should_brake_for_rolling_stop = das->smart_das_driving_style == SMART_DAS_DRIVING_STYLE_RECKLESS || das->smart_das_driving_style == SMART_DAS_DRIVING_STYLE_AGGRESSIVE;
                    // for reckless, if we are approaching faster than other cars, do not even do rolling stop, just go through
                    if (das->smart_das_driving_style == SMART_DAS_DRIVING_STYLE_RECKLESS && should_brake_for_rolling_stop) {
                        bool is_approaching_faster_than_others = true;
                        if (!all_clear) {
                            // someone is already on intersection
                            is_approaching_faster_than_others = false;
                        } else {
                            Car* foremost_vehicle = intersection_get_foremost_vehicle(situation->intersection, sim);
                            Meters perceived_progress;
                            MetersPerSecond perceived_speed;
                            // Add perception dropout for the other vehicle
                            bool perceived = car_noisy_perceive_other(car, foremost_vehicle, sim, NULL, &perceived_progress, &perceived_speed, NULL, NULL, NULL, NULL, NULL, NULL, false, sim_get_dt(sim));
                            if (!perceived) {
                                foremost_vehicle = NULL;
                            }

                            if (foremost_vehicle && foremost_vehicle->id != car->id) {
                                // how far is the foremost vehicle from the intersection?
                                Lane* foremost_vehicle_lane = car_get_lane(foremost_vehicle, map);
                                
                                Meters distance_of_foremost_vehicle_to_stop_line = lane_get_length(foremost_vehicle_lane) - perceived_progress - STOP_LINE_BUFFER_METERS - car_get_length(foremost_vehicle) / 2.0;

                                Seconds time_for_foremost_vehicle_to_stop_line = (perceived_speed > STOP_SPEED_THRESHOLD) ? distance_of_foremost_vehicle_to_stop_line / perceived_speed : INFINITY;
                                Seconds time_for_ego_to_stop_line = (situation->speed > STOP_SPEED_THRESHOLD) ? distance_to_stop_line / situation->speed : INFINITY;
                                if (time_for_ego_to_stop_line >= time_for_foremost_vehicle_to_stop_line) {
                                    is_approaching_faster_than_others = false;
                                }
                            }
                        }
                        if (is_approaching_faster_than_others) {
                            should_brake_for_rolling_stop = false;
                        }
                    }
                }
                should_brake_for_yield = false;
                should_speed_up_to_make_light = false;
                break;
            }
            default:
                should_brake_for_full_stop = true;  // default to safe option for unknown lights
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = false;
                should_speed_up_to_make_light = false;
                break;
            }


            // Finally, as you enter the intersection (e.g., going straight), and someone else in front of you just went to a different lane (e.g., turned left), we should brake a bit to let them make some distance. Basically, check all outgoing connections and see if there is a car very close to the intersection in front of us but in a different lane.
            for (int turn_dir = 0; turn_dir < 3; turn_dir++) {
                const Lane* next_lane = situation->lane_next_after_turn[turn_dir];
                if (next_lane != NULL && next_lane != situation->lane_next_after_turn[das->turn_intent] && lane_get_num_cars(next_lane) > 0) {
                    Car* closest_car = lane_get_car(next_lane, sim, lane_get_num_cars(next_lane) - 1); // last car in the lane is the closest to us
                    // no noise here because we are checking for very close cars
                    if (closest_car) {
                        Meters closest_car_progress = closest_car->lane_progress_meters;
                        if (closest_car_progress < car_get_length(closest_car) + car_get_length(car)) {
                            should_brake_for_yield = true;
                            break;
                        }
                    }
                }
            }

            if (das->smart_das_driving_style == SMART_DAS_DRIVING_STYLE_NO_RULES) {
                // in this mode, we don't brake for anything at intersections
                should_brake_for_full_stop = false;
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = false;
                should_speed_up_to_make_light = false;
            }

            if (should_brake_for_full_stop && !is_sudden_emergency_braking_possible) {
                // cannot brake in time even with max capable braking, so gotta run through (other than reckless, which should speed instead even if red)
                if (das->smart_das_driving_style == SMART_DAS_DRIVING_STYLE_RECKLESS) {
                    should_brake_for_full_stop = false;
                    should_speed_up_to_make_light = true;
                } else {
                    should_brake_for_full_stop = false;
                    should_speed_up_to_make_light = false;
                }
            }

            if (should_brake_for_yield && !is_sudden_emergency_braking_possible) {
                // cannot brake in time even with max capable braking, so gotta run through
                // printf("Car %d cannot brake for yield in time, so not yielding\n", car->id);
                should_brake_for_yield = false;
            }


            bool should_brake = should_brake_for_full_stop || should_brake_for_rolling_stop || should_brake_for_yield;
            if (should_brake) {
                // compute position target delta. If full stop or rolling stop, target is stop line - buffer. If yield, target is end of lane.
                Meters car_position = situation->lane_progress_m;
                Meters position_target_delta = (should_brake_for_full_stop || should_brake_for_rolling_stop) ? distance_to_stop_line : situation->distance_to_end_of_lane_from_leading_edge;
                // Brake smoothly to target stop at the end of the lane
                Meters position_target = car_position + fmax(position_target_delta, meters(0)); // the fmax is needed to prevent reversing if the car has already crossed the stop line (could happen if light turned red while crossing)
                // compute speed at target. If full stop or yield, target speed is 0. If rolling stop, target speed is CREEP_SPEED.
                MetersPerSecond speed_at_target = (should_brake_for_rolling_stop) ? CREEP_SPEED : 0;
                // compute speed limit. Post stop line, limit speed to CREEP_SPEED.
                MetersPerSecond speed_limit = distance_to_stop_line > 0 ? das->speed_target : fmin(CREEP_SPEED, das->speed_target);
                MetersPerSecondSquared accel_to_stop = car_compute_acceleration_chase_target(car, position_target, speed_at_target, fmax(situation->distance_to_end_of_lane_from_leading_edge - position_target, meters(0)), speed_limit, das->use_preferred_accel_profile);
                // accel_to_stop = 0.5 * accel_to_stop + 0.5 * car_get_acceleration(car); // smooth it a bit
                bool stop_right_away = (situation->distance_to_end_of_lane_from_leading_edge < 0.2 && situation->going_forward) || (accel_to_stop < 0 && (situation->stopped || situation->reversing));
                if (stop_right_away) {
                    accel_to_stop = car_compute_acceleration_stop_max_brake(car, das->use_preferred_accel_profile);
                }
                accel = accel_to_stop; // take the minimum of the two accelerations
            } else if (should_speed_up_to_make_light) {
                // compute acceleration to speed up to make the light
                das->speed_target = das->speed_target + from_mph(10);  // speed up by 10 mph to try to make the light
            } else {
                // we are approaching, but no need to stop or yield. Best to start adjusting speed to the speed limit of the intersection-lane you are going into if you are approaching end of lane.
                if (situation->is_approaching_end_of_lane) {
                    MetersPerSecond next_lane_speed_limit = lane_get_speed_limit(situation->lane_next_after_turn[das->turn_intent]);
                    MetersPerSecond next_lane_speed_targets[SMART_DAS_DRIVING_STYLES_COUNT] = {
                        [SMART_DAS_DRIVING_STYLE_NO_RULES] = 1.5 * next_lane_speed_limit,
                        [SMART_DAS_DRIVING_STYLE_RECKLESS] = 1.25 * next_lane_speed_limit,
                        [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = 1.1 * next_lane_speed_limit,
                        [SMART_DAS_DRIVING_STYLE_NORMAL] = next_lane_speed_limit,
                        [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = 0.9 * next_lane_speed_limit
                    };
                    das->speed_target = next_lane_speed_targets[das->smart_das_driving_style];
                }
            }
        }
    } else {
        das->smart_das_intersection_approach_engaged = false; // reset the flag
    }

    return accel;
}


void driving_assistant_smart_update_das_variables(DrivingAssistant* das, Car* car, Simulation* sim, SituationalAwareness* sa) {
    if (das->smart_das) {

        Seconds thws[] = {
            [SMART_DAS_DRIVING_STYLE_NO_RULES] = 0.5,
            [SMART_DAS_DRIVING_STYLE_RECKLESS] = 1.0,
            [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = 2.0,
            [SMART_DAS_DRIVING_STYLE_NORMAL] = 3.0,
            [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = 4.0
        };

        Meters buffers[] = {
            [SMART_DAS_DRIVING_STYLE_NO_RULES] = 0.5,
            [SMART_DAS_DRIVING_STYLE_RECKLESS] = 1.0,
            [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = 2.0,
            [SMART_DAS_DRIVING_STYLE_NORMAL] = 3.0,
            [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = 4.0
        };

        MetersPerSecond lane_speed_limit = lane_get_speed_limit(sa->lane);

        if (sa->road) {
            int num_lanes = road_get_num_lanes(sa->road);
            // on two or three lane roads, increase speed limit of passing lane by 5 mph
            if ((num_lanes == 2 || num_lanes == 3) && sa->is_on_leftmost_lane) {
                lane_speed_limit += from_mph(5);
            }
            // on three lanes road, decrease speed limit of rightmost lane by 5 mph
            if (num_lanes == 3 && sa->is_on_rightmost_lane) {
                lane_speed_limit -= from_mph(5);
            }

            // on 4 lane roads, increase speed limit of leftmost by 10 mph, 2nd from left by 5 mph, and decrease of rightmost lane by 5.
            if (num_lanes == 4) {
                if (sa->is_on_leftmost_lane) {
                    lane_speed_limit += from_mph(10);
                } else if (road_find_index_of_lane(sa->road, sa->lane->id) == 2) {
                    lane_speed_limit += from_mph(5);
                } else if (sa->is_on_rightmost_lane) {
                    lane_speed_limit -= from_mph(5);
                }
            }

            // for more than 4 lanes, no adjustments for now. raise warning
            if (num_lanes > 4) {
                LOG_WARN("Smart DAS: No speed limit adjustments for roads with more than 4 lanes. Road ID: %d, Num Lanes: %d", sa->road->id, num_lanes);
            }
        }

        MetersPerSecond speed_limits[] = {
            [SMART_DAS_DRIVING_STYLE_NO_RULES] = 1.5 * lane_speed_limit,
            [SMART_DAS_DRIVING_STYLE_RECKLESS] = 1.25 * lane_speed_limit,
            [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = 1.1 * lane_speed_limit,
            [SMART_DAS_DRIVING_STYLE_NORMAL] = lane_speed_limit,
            [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = 0.9 * lane_speed_limit
        };

        // set the THW and speed target based on driving style
        das->thw = thws[das->smart_das_driving_style];
        das->buffer = buffers[das->smart_das_driving_style];
        das->speed_target = speed_limits[das->smart_das_driving_style];
        das->speed_target = fmin(das->speed_target, car->capabilities.top_speed); // cap speed target to car's max speed
        das->use_linear_speed_control = das->smart_das_driving_style <= SMART_DAS_DRIVING_STYLE_RECKLESS; // reckless uses linear speed control instead of exponential error decay control as it likes more abrupt changes
        das->use_preferred_accel_profile = das->smart_das_driving_style >= SMART_DAS_DRIVING_STYLE_NORMAL; // normal and defensive use preferred accel profile (which is smoother than capable), while reckless and aggressive use full capable profile
        das->merge_assistance = true; // smart DAS always has merge assistance on
        das->follow_assistance = true; // smart DAS always has follow assistance on

        das->smart_das_intersection_approach_engaged = driving_assistant_smart_das_should_engage_intersection_approach(das, car, sim, sa);

        if (das->smart_das_intersection_approach_engaged) {
            // disable lane change and freeze turn intent during intersection approach
            das->merge_intent = INDICATOR_NONE;     // disable merge intent (no lane changes during intersection approach)
            CarIndicator old_turn_intent = car_get_indicator_turn(car);
            if (sa->is_turn_possible[old_turn_intent]) {
                das->turn_intent = old_turn_intent; // freeze turn intent to current turn indicator if valid
            } else {
                // it may not be possible to do current turn intent
                if (!sa->is_turn_possible[das->turn_intent]) {
                    // if current turn intent is also not possible, just go straight
                    if (sa->is_turn_possible[INDICATOR_NONE]) {
                        das->turn_intent = INDICATOR_NONE;
                    } else if (sa->is_turn_possible[INDICATOR_RIGHT]) {
                        das->turn_intent = INDICATOR_RIGHT;
                    } else if (sa->is_turn_possible[INDICATOR_LEFT]) {
                        das->turn_intent = INDICATOR_LEFT;
                    } else {
                        LOG_ERROR("Smart DAS: No valid turn intents possible at intersection for car ID %d", car->id);
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }
}

bool driving_assistant_control_car(DrivingAssistant* das, Car* car, Simulation* sim) {
    if (!car || !sim || !das) {
        LOG_ERROR("Invalid car, simulation, or driving assistant provided to driving_assistant_control_car");
        return false; // Indicating error due to invalid input
    }

    // Car control variables to determine:

    MetersPerSecondSquared accel = car->capabilities.accel_profile.max_acceleration; // Overall acc will be minimum of all the following acceleration logics
    CarIndicator lane_indicator = INDICATOR_NONE;
    CarIndicator turn_indicator = INDICATOR_NONE;
    bool request_lane_change = false;
    bool request_turn = false;

    situational_awareness_build(sim, car->id);  // ensure up-to-date situational awareness
    SituationalAwareness* sa = sim_get_situational_awareness(sim, car_get_id(car));

    if (das->smart_das) {

        driving_assistant_smart_update_das_variables(das, car, sim, sa);

        // ------------- Intersection handling (it may adjust speed target to smoothly transition to the intersection lane when not stopping/yielding or briefly speeding to jump yellow) ------------------------
        MetersPerSecondSquared accel_intersection = driving_assistant_smart_das_handle_intersection(das, car, sim, sa);
        accel = fmin(accel, accel_intersection);
    }

    das->speed_target = fmin(das->speed_target, car->capabilities.top_speed); // cap speed target to car's max speed

    MetersPerSecondSquared accel_prev = car_get_acceleration(car);
    Seconds dt = sim_get_dt(sim);

    // ------------- Follow lead vehicle ------------------------
    if (das->follow_assistance) {
        Meters buffer = das->buffer;
        MetersPerSecondSquared accel_cruise = car_compute_acceleration_adaptive_cruise(car, sim, sa, fmax(das->speed_target, 0), das->thw, buffer, das->use_preferred_accel_profile); // follow next car or adjust speed. Doesn't work for negative speed target.
        double step_size = 0.99 * dt; // smoothing step size
        accel_cruise = step_size * accel_cruise + (1 - step_size) * accel_prev; // smooth it a bit. As a bonus, it simulates driver reaction time delay.
        // and once you stop, no reversal unless lead car is reversing too
        if (sa->stopped && accel_cruise < 0 && (!sa->nearby_vehicles.lead || sa->nearby_vehicles.lead->speed > -STOP_SPEED_THRESHOLD)) {
            accel_cruise = 0; // prevent reversing
        } 
        accel = fmin(accel, accel_cruise);
    }

    // ----------- Speed adjustment ------------------------
    MetersPerSecondSquared accel_speed_adjust = das->use_linear_speed_control ? car_compute_acceleration_adjust_speed_linear(car, das->speed_target, das->use_preferred_accel_profile) : car_compute_acceleration_adjust_speed(car, das->speed_target, das->use_preferred_accel_profile); // works for negative speed targets as well
    accel = fmin(accel, accel_speed_adjust);

    // ----------- Handle stop at dead end at the last moment with full brake ------------------------
    if (sa->speed > 0 && sa->is_approaching_dead_end && sa->braking_distance.capable >= sa->distance_to_end_of_lane_from_leading_edge - STOP_LINE_BUFFER_METERS) {
        MetersPerSecondSquared accel_emergency_stop = -car->capabilities.accel_profile.max_deceleration;
        accel = fmin(accel, accel_emergency_stop);
    }

    // ----------- Handle stop at intersection (if smart DAS not enabled)  ------------------------
    if (!das->smart_das && sa->speed > 0 && das->should_stop_at_intersection && sa->is_an_intersection_upcoming) {
        Meters distance_to_stop_line = sa->distance_to_end_of_lane_from_leading_edge - STOP_LINE_BUFFER_METERS;
        MetersPerSecondSquared accel_stop;
        Meters position_target = sa->lane_progress_m + fmax(distance_to_stop_line, 0);
        accel_stop = car_compute_acceleration_chase_target(car, position_target, 0, fmax(sa->distance_to_end_of_lane_from_leading_edge - position_target, meters(0)), das->speed_target, das->use_preferred_accel_profile);
        accel_stop = 0.9 * accel_stop + 0.1 * car_get_acceleration(car); // smooth it a bit
        accel = fmin(accel, accel_stop);
    }

    // Seconds dt = sim_get_dt(sim);




    // smooth acceleration changes to prevent jerky behavior
    // MetersPerSecondSquared accel_prev = car_get_acceleration(car);
    // // if (accel * accel_prev < 0) {
    // //     accel = 0; // changing pedals, go through zero first
    // // } else {
    // //     // same direction, limit change rate
        // MetersPerSecondSquared accel_change_mag = fabs(accel - accel_prev);
        
        // if (accel_change_mag / dt > car->capabilities.accel_profile.max_acceleration) {
        //     // smooth and clip it:
        //     if (accel > accel_prev) {
        //         accel = accel_prev + car->capabilities.accel_profile.max_acceleration * dt;
        //     } else {
        //         accel = accel_prev - car->capabilities.accel_profile.max_acceleration * dt;
        //     }
        //     // accel = 0.5 * (accel + accel_prev); // average to smoothen further
        // }
    // }
    // accel = 0.5 * (accel + accel_prev);

    // once you stop, never reverse, unless speed target is negative, and don't do micro adjustments either.
    // if (car_get_speed(car) <= 1e-9 && das->speed_target >= 0) {
    //     if (accel < 0) {
    //         accel = 0; // prevent reversing
    //     } else if (accel < 0.1 * car->capabilities.accel_profile.max_acceleration) {
    //         accel = 0; // prevent micro adjustments
    //     }
    // }



    // ---- AEB logic overrides everything else ----

    // auto engagement conditions
    bool execute_aeb = false;
    Meters best_case_braking_gap = compute_best_case_braking_gap(sim, car, sa); // todo: fix this fn
    if (das->aeb_in_progress || (das->aeb_assistance && sa->nearby_vehicles.lead && sa->speed >= AEB_ENGAGE_MIN_SPEED_MPS && best_case_braking_gap < AEB_ENGAGE_BEST_CASE_BRAKING_GAP_M)) {
        execute_aeb = !das->aeb_manually_disengaged;
    }

    // execute AEB
    if (execute_aeb) {
        if (!das->aeb_in_progress) {
            das->aeb_in_progress = true; // Start AEB if not already in progress
            // LOG_DEBUG("AEB engaged for car %d. Best-case gap: %.2f feet, speed: %.2f mph", car->id, to_feet(best_case_braking_gap), to_mph(car_get_speed(car)));
        }
        MetersPerSecondSquared accel_aeb = car_compute_acceleration_stop_max_brake(car, das->use_preferred_accel_profile);
        accel = fmin(accel, accel_aeb);
        // accel = accel_aeb; // AEB overrides all other acceleration commands
        LOG_TRACE("AEB in progress for car %d. Best-case gap: %.2f feet, speed: %.2f mph", car->id, to_feet(best_case_braking_gap), to_mph(car_get_speed(car)));
        // NOTE: Forward crash is still possible if the lead vehicle is backing into us, but we don't handle that case here since that is not about emergency braking.
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
            Meters buffer = das->buffer;
            if (car_is_lane_change_dangerous(car, sim, sa, lane_indicator, das->thw, buffer, sim_get_dt(sim))) {
                LOG_TRACE("Waiting for safe merge for car %d with indicator %d", car->id, lane_indicator);
                request_lane_change = false;        // Keep indicating but don't execute
            } else {
                request_lane_change = true; // Safe to change lane
            }
        } else {
            // Unless we are indicating in advance for a highway exit, request right away even if unsafe as long as it is possible to do so.
            request_lane_change = (lane_indicator == INDICATOR_LEFT && sa->is_lane_change_left_possible) || (lane_indicator == INDICATOR_RIGHT && sa->is_lane_change_right_possible);
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
    SituationalAwareness* sa = sim_get_situational_awareness(sim, car_get_id(car));

    if (das->aeb_in_progress) {
        das->speed_target = car_get_speed(car); // Update speed target to current speed if AEB is in progress, so that once AEB is disengaged (automatic or manual), the car will not accelerate or decelerate unexpectedly
    }

    Meters best_case_braking_gap = compute_best_case_braking_gap(sim, car, sim_get_situational_awareness(sim, car_get_id(car)));

    // auto disengagement conditions
    bool disengagement_conditions_met = false;
    if (best_case_braking_gap > AEB_DISENGAGE_BEST_CASE_BRAKING_GAP_M || sa->speed < AEB_DISENGAGE_MAX_SPEED_MPS) {
        disengagement_conditions_met = true;
    }

    if (disengagement_conditions_met) {
        if (das->aeb_in_progress) {
            // LOG_DEBUG("AEB disengaged for car %d. Best case braking gap: %.2f feet, speed: %.2f mph", car->id, to_feet(best_case_braking_gap), to_mph(car_get_speed(car)));
        }
        das->aeb_in_progress = false; // Reset AEB state if auto disengagement conditions are met
        das->aeb_manually_disengaged = false; // Reset manual disengagement state if auto disengagement occurs
    }

    // update intersection approach engagement flag
    if (das->smart_das) {
        das->smart_das_intersection_approach_engaged = driving_assistant_smart_das_should_engage_intersection_approach(das, car, sim, sa);
    }

    // reset merge intent once merge successful. Detected by checking mismatch between merge_intent and current lane indicator, assuming car has auto-turn off indicator enabled
    if (das->merge_intent != car_get_indicator_lane(car)) {
        assert((car_get_indicator_lane(car) == INDICATOR_NONE && !car_get_request_indicated_lane(car)) && "How did this happen?");
        das->merge_intent = INDICATOR_NONE; // Reset merge intent
        LOG_TRACE("Merge intent reset for car %d after successful merge.", car->id);
    }

    // reset turn intent once turn successful. Detected by checking mismatch between turn_intent and current turn indicator, assuming car has auto-turn off indicator enabled
    if (das->turn_intent != car_get_indicator_turn(car)) {
        assert((car_get_indicator_turn(car) == INDICATOR_NONE && !car_get_request_indicated_turn(car)) && "How did this happen?");
        das->turn_intent = INDICATOR_NONE; // Reset turn intent
        LOG_TRACE("Turn intent reset for car %d after successful turn.", car->id);
    }
}



// Getters

Seconds driving_assistant_get_last_configured_at(const DrivingAssistant* das) { return das->last_configured_at; }
MetersPerSecond driving_assistant_get_speed_target(const DrivingAssistant* das) { return das->speed_target; }
bool driving_assistant_get_should_stop_at_intersection(const DrivingAssistant* das) { return das->should_stop_at_intersection; }
CarIndicator driving_assistant_get_merge_intent(const DrivingAssistant* das) { return das->merge_intent; }
CarIndicator driving_assistant_get_turn_intent(const DrivingAssistant* das) { return das->turn_intent; }
Seconds driving_assistant_get_thw(const DrivingAssistant* das) { return das->thw; }
Meters driving_assistant_get_buffer(const DrivingAssistant* das) { return das->buffer; }
bool driving_assistant_get_follow_assistance(const DrivingAssistant* das) { return das->follow_assistance; }
bool driving_assistant_get_merge_assistance(const DrivingAssistant* das) { return das->merge_assistance; }
bool driving_assistant_get_aeb_assistance(const DrivingAssistant* das) { return das->aeb_assistance; }
bool driving_assistant_get_aeb_in_progress(const DrivingAssistant* das) { return das->aeb_in_progress; }
bool driving_assistant_get_aeb_manually_disengaged(const DrivingAssistant* das) { return das->aeb_manually_disengaged; }
bool driving_assistant_get_use_preferred_accel_profile(const DrivingAssistant* das) { return das->use_preferred_accel_profile; }
bool driving_assistant_get_use_linear_speed_control(const DrivingAssistant* das) { return das->use_linear_speed_control; }
bool driving_assistant_get_smart_das(const DrivingAssistant* das) { return das->smart_das; }
SmartDASDrivingStyle driving_assistant_get_smart_das_driving_style(const DrivingAssistant* das) { return das->smart_das_driving_style; }


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

void driving_assistant_configure_should_stop_at_intersection(DrivingAssistant* das, const Car* car, const Simulation* sim, bool should_stop_at_intersection) {
    if (!validate_input(car, das, sim)) {
        return; // Early return if input validation fails
    }
    das->should_stop_at_intersection = should_stop_at_intersection;
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

void driving_assistant_configure_thw(DrivingAssistant* das, const Car* car, const Simulation* sim, Seconds thw) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (thw < 0) {
        LOG_WARN("THW cannot be negative for car %d. Setting it to 0.", car->id);
        thw = 0;
    }
    das->thw = thw;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_buffer(DrivingAssistant* das, const Car* car, const Simulation* sim, Meters buffer) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    if (buffer < 0) {
        LOG_WARN("Buffer cannot be negative for car %d. Setting it to 0.", car->id);
        buffer = 0;
    }
    das->buffer = buffer;
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_follow_assistance(DrivingAssistant* das, const Car* car, const Simulation* sim, bool follow_assistance) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    das->follow_assistance = follow_assistance;
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

void driving_assistant_configure_smart_das(DrivingAssistant* das, const Car* car, const Simulation* sim, bool smart_das) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    das->smart_das = smart_das;
    if (!smart_das) {
        das->smart_das_intersection_approach_engaged = false; // reset the flag
    }
    das->last_configured_at = sim_get_time(sim);
}

void driving_assistant_configure_smart_das_driving_style(DrivingAssistant* das, const Car* car, const Simulation* sim, SmartDASDrivingStyle driving_style) {
    if (!validate_input(car, das, sim)) {
        return;
    }
    das->smart_das_driving_style = driving_style;
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
