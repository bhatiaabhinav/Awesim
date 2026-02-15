#include "ai.h"
#include "logging.h"
#include "sim.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

// #define BENCHMARK_DAS_CONTROL
// ^^ Uncomment to enable per-section timing of driving_assistant_control_car
// #define BENCHMARK_DAS_INTERSECTION
// ^^ Uncomment to enable per-section timing of handle_intersection
// #define BENCHMARK_AEB
// ^^ Uncomment to enable per-section timing of compute_best_case_braking_gap
// #define BENCHMARK_DAS_POST_STEP
// ^^ Uncomment to enable per-section timing of driving_assistant_post_sim_step

// Shared timer utility for any benchmark in this file
#if defined(BENCHMARK_DAS_CONTROL) || defined(BENCHMARK_DAS_INTERSECTION) || defined(BENCHMARK_AEB) || defined(BENCHMARK_DAS_POST_STEP)
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
static double _bdas_get_time_us(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double _bdas_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}
#endif
#endif

#ifdef BENCHMARK_DAS_CONTROL
static double _bdas_sa_build_us = 0;
static double _bdas_smart_das_us = 0;
static double _bdas_speed_ctrl_us = 0;
static double _bdas_pedal_sim_us = 0;
static double _bdas_aeb_us = 0;
static double _bdas_merge_turn_us = 0;
static double _bdas_apply_us = 0;
static int _bdas_call_count = 0;
#define BDAS_INTERVAL 500000
#endif

#ifdef BENCHMARK_DAS_INTERSECTION
static double _bdas_intx_setup_us = 0;
static double _bdas_intx_lightswitch_us = 0;
static double _bdas_intx_exitcheck_us = 0;
static double _bdas_intx_defyield_us = 0;
static double _bdas_intx_brakecomp_us = 0;
static int _bdas_intx_count = 0;
#define BDAS_INTX_INTERVAL 500000
#endif

#ifdef BENCHMARK_AEB
static double _baeb_perceive_us = 0;
static double _baeb_physics_us = 0;
static double _baeb_total_us = 0;
static int _baeb_call_count = 0;
#define BAEB_INTERVAL 500000
#endif

#ifdef BENCHMARK_DAS_POST_STEP
static double _bpost_sa_build_us = 0;
static double _bpost_aeb_disengage_us = 0;
static double _bpost_intx_update_us = 0;
static double _bpost_cleanup_us = 0;
static int _bpost_call_count = 0;
#define BPOST_INTERVAL 500000
#endif


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
static Meters compute_best_case_braking_gap(Simulation* sim, const Car* car, const SituationalAwareness* sa, Meters* distance_to_lead_cached, MetersPerSecond* speed_lead_cached, MetersPerSecondSquared* accel_lead_cached) {
    #ifdef BENCHMARK_AEB
    double _baeb_t0 = _bdas_get_time_us();
    #endif
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
    bool cached = (distance_to_lead_cached && speed_lead_cached && accel_lead_cached);
    Meters distance_to_lead;
    MetersPerSecond speed_lead;
    MetersPerSecondSquared accel_lead;
    if (cached) {
        distance_to_lead = *distance_to_lead_cached;
        speed_lead = *speed_lead_cached;
        accel_lead = *accel_lead_cached;
    } else {
        perceive_lead_vehicle(car, sim, sa, &distance_to_lead, &speed_lead, &accel_lead);
    }
    #ifdef BENCHMARK_AEB
    double _baeb_t1 = _bdas_get_time_us();
    #endif

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

    #ifdef BENCHMARK_AEB
    double _baeb_t2 = _bdas_get_time_us();
    _baeb_perceive_us += _baeb_t1 - _baeb_t0;
    _baeb_physics_us  += _baeb_t2 - _baeb_t1;
    _baeb_total_us    += _baeb_t2 - _baeb_t0;
    _baeb_call_count++;
    if (_baeb_call_count % BAEB_INTERVAL == 0) {
        double n = (double)_baeb_call_count;
        printf("[BENCH aeb_gap] Avg over %d calls: Perceive=%.3f us | Physics=%.3f us | Total=%.3f us\n",
            _baeb_call_count,
            _baeb_perceive_us / n,
            _baeb_physics_us / n,
            _baeb_total_us / n);
    }
    #endif

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
        // Compute braking distance based on driving style directly
        Meters braking_distance_to_use;
        switch (das->smart_das_driving_style) {
            case SMART_DAS_DRIVING_STYLE_RECKLESS:    braking_distance_to_use = situation->braking_distance.capable; break;
            case SMART_DAS_DRIVING_STYLE_AGGRESSIVE:  braking_distance_to_use = situation->braking_distance.preferred; break;
            default:                                  braking_distance_to_use = situation->braking_distance.preferred_smooth; break;
        }
        if (situation->speed < 0) {
            braking_distance_to_use = 0; // if reversing, no braking distance needed, which is originally positive even if speed is negative
        }
        return distance_to_stop_line <= 1.1 * braking_distance_to_use || situation->is_approaching_end_of_lane; // the 10% extra buffer is to account for reaction/actuation delays
    }
    return false;
}

static const bool _like_to_speed_up_for_yellows[SMART_DAS_DRIVING_STYLES_COUNT] = {
    [SMART_DAS_DRIVING_STYLE_NO_RULES] = false,
    [SMART_DAS_DRIVING_STYLE_RECKLESS] = true,
    [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = true,
    [SMART_DAS_DRIVING_STYLE_NORMAL] = false,
    [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = false
};
static const bool _defensive_even_if_right_of_way[SMART_DAS_DRIVING_STYLES_COUNT] = {
    [SMART_DAS_DRIVING_STYLE_NO_RULES] = false,
    [SMART_DAS_DRIVING_STYLE_RECKLESS] = false,
    [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = false,
    [SMART_DAS_DRIVING_STYLE_NORMAL] = true,
    [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = true
};

static MetersPerSecondSquared driving_assistant_smart_das_handle_intersection(DrivingAssistant* das, Car* car, Simulation* sim, SituationalAwareness* situation) {
    Map* map = sim_get_map(sim);
    Seconds dt = sim_get_dt(sim);
    MetersPerSecondSquared accel = car->capabilities.accel_profile.max_acceleration;
    if (situation->is_an_intersection_upcoming) {
        if (das->smart_das_intersection_approach_engaged) {
            #ifdef BENCHMARK_DAS_INTERSECTION
            double _bi_t0 = _bdas_get_time_us();
            #endif
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

            // Compute is_ok_with_yielding directly instead of building array
            SmartDASDrivingStyle style = das->smart_das_driving_style;
            bool is_ok_with_yielding;
            switch (style) {
                case SMART_DAS_DRIVING_STYLE_NO_RULES:    is_ok_with_yielding = false; break;
                case SMART_DAS_DRIVING_STYLE_RECKLESS:    is_ok_with_yielding = is_smooth_braking_possible; break;
                case SMART_DAS_DRIVING_STYLE_AGGRESSIVE:  is_ok_with_yielding = is_sudden_semi_hard_braking_possible; break;
                default:                                  is_ok_with_yielding = is_sudden_emergency_braking_possible; break;
            }
            // Compute should_brake_for_yellow directly
            bool should_brake_for_yellow = (style >= SMART_DAS_DRIVING_STYLE_AGGRESSIVE) && is_smooth_braking_possible;
            #ifdef BENCHMARK_DAS_INTERSECTION
            double _bi_t1 = _bdas_get_time_us();
            #endif

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
                should_brake_for_full_stop = false;
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = is_ok_with_yielding && car_has_something_to_yield_to_at_intersection(car, sim, situation, das->turn_intent, yield_thw, dt);
                should_speed_up_to_make_light = false;
                break;
            case TRAFFIC_LIGHT_YELLOW:
                // note: we are not lawfully expected to hard brake on yellows in general. No driving style should do that.
                should_brake_for_full_stop = should_brake_for_yellow;
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = false;
                should_speed_up_to_make_light = !should_brake_for_full_stop && _like_to_speed_up_for_yellows[style];
                break;
            case TRAFFIC_LIGHT_YELLOW_YIELD:
                // note: we are not lawfully expected to hard brake on yellows in general. No driving style should do that. Also, we may need to yield here in case we decide to go through the yellow. Also, we won't be speeding up to make the light here, because when combined with yielding, it messes up things.
                should_brake_for_full_stop = should_brake_for_yellow;
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = !should_brake_for_full_stop && is_ok_with_yielding && car_has_something_to_yield_to_at_intersection(car, sim, situation, das->turn_intent, yield_thw, dt);
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
                if (style == SMART_DAS_DRIVING_STYLE_DEFENSIVE) {
                    yield_mode = false; // defensive treats red-yield as red
                }
                if (style == SMART_DAS_DRIVING_STYLE_RECKLESS) {
                    yield_mode = true; // reckless treats red-yield as yield
                }
                if (yield_mode) {
                    should_brake_for_full_stop = false;
                    should_brake_for_rolling_stop = false;
                    should_brake_for_yield = car_has_something_to_yield_to_at_intersection(car, sim, situation, das->turn_intent, yield_thw, dt);
                } else {
                    should_brake_for_full_stop = true;
                    should_brake_for_rolling_stop = false; 
                    should_brake_for_yield = false;
                    if (style <= SMART_DAS_DRIVING_STYLE_AGGRESSIVE) {
                        should_brake_for_full_stop = false;
                        should_brake_for_rolling_stop = true; // rolling stop for aggressive
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
                        if (situation->is_my_turn_at_stop_sign_fcfs) {
                            should_brake_for_full_stop = false;
                        } else {
                            // even those doing rolling stops should now come to full stop when it's not their turn or not clear
                            should_brake_for_full_stop = true;
                        }
                    }
                } else {
                    // come to full stop or rolling stop at stop line
                    should_brake_for_full_stop = style == SMART_DAS_DRIVING_STYLE_NORMAL || style == SMART_DAS_DRIVING_STYLE_DEFENSIVE;
                    should_brake_for_rolling_stop = style == SMART_DAS_DRIVING_STYLE_RECKLESS || style == SMART_DAS_DRIVING_STYLE_AGGRESSIVE;
                    // for reckless, if we are approaching faster than other cars, do not even do rolling stop, just go through
                    if (style == SMART_DAS_DRIVING_STYLE_RECKLESS && should_brake_for_rolling_stop) {
                        bool is_approaching_faster_than_others = true;
                        if (!all_clear) {
                            // someone is already on intersection
                            is_approaching_faster_than_others = false;
                        } else {
                            Car* foremost_vehicle = intersection_get_foremost_vehicle(situation->intersection, sim);
                            Meters perceived_progress;
                            MetersPerSecond perceived_speed;
                            // Add perception dropout for the other vehicle
                            bool perceived = car_noisy_perceive_other(car, foremost_vehicle, sim, NULL, &perceived_progress, &perceived_speed, NULL, NULL, NULL, NULL, NULL, NULL, false, dt);
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
                should_brake_for_full_stop = true;  // default to safe option for unknown/not-yet-implemented lights
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = false;
                should_speed_up_to_make_light = false;
                break;
            }


            #ifdef BENCHMARK_DAS_INTERSECTION
            double _bi_t2 = _bdas_get_time_us();
            #endif
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

            #ifdef BENCHMARK_DAS_INTERSECTION
            double _bi_t3 = _bdas_get_time_us();
            #endif
            bool should_brake = should_brake_for_full_stop || should_brake_for_rolling_stop || should_brake_for_yield;

            if (!should_brake && _defensive_even_if_right_of_way[style] && defensive_yield_despite_right_of_way(car, sim, situation, das->turn_intent, yield_thw, dt)) {
                // even if we have the right of way, if we are defensive and there is something to yield to with a short time headway, we should yield
                should_brake_for_yield = true;
                should_speed_up_to_make_light = false; // if we were going to speed up for yellow, now we should not do that because we are going to yield instead
                should_brake = true;
            }

            if (style == SMART_DAS_DRIVING_STYLE_NO_RULES) {
                // in this mode, we don't brake for anything at intersections
                should_brake_for_full_stop = false;
                should_brake_for_rolling_stop = false;
                should_brake_for_yield = false;
                should_speed_up_to_make_light = false;
            }

            if (should_brake_for_full_stop && !is_sudden_emergency_braking_possible) {
                // cannot brake in time even with max capable braking, so gotta run through (other than reckless, which should speed instead even if red)
                if (style == SMART_DAS_DRIVING_STYLE_RECKLESS) {
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


            #ifdef BENCHMARK_DAS_INTERSECTION
            double _bi_t4 = _bdas_get_time_us();
            #endif
            should_brake = should_brake_for_full_stop || should_brake_for_rolling_stop || should_brake_for_yield;   // update should_brake after adjustments based on feasibility and style
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
                bool stop_right_away = (situation->distance_to_end_of_lane_from_leading_edge < 0 && situation->going_forward) || (accel_to_stop < 0 && (situation->stopped || situation->reversing));
                if (stop_right_away) {
                    accel_to_stop = -situation->speed / dt; // just enough to stop right away
                    accel_to_stop = fclamp(accel_to_stop, -car->capabilities.accel_profile.max_deceleration, car->capabilities.accel_profile.max_acceleration);
                }
                accel = accel_to_stop; // take the minimum of the two accelerations
            } else if (should_speed_up_to_make_light && das->turn_intent != INDICATOR_RIGHT) {
                // compute acceleration to speed up to make the light
                das->speed_target = das->speed_target + from_mph(10);
            }
            #ifdef BENCHMARK_DAS_INTERSECTION
            double _bi_t5 = _bdas_get_time_us();
            _bdas_intx_setup_us       += _bi_t1 - _bi_t0;
            _bdas_intx_lightswitch_us += _bi_t2 - _bi_t1;
            _bdas_intx_exitcheck_us   += _bi_t3 - _bi_t2;
            _bdas_intx_defyield_us    += _bi_t4 - _bi_t3;
            _bdas_intx_brakecomp_us   += _bi_t5 - _bi_t4;
            _bdas_intx_count++;
            if (_bdas_intx_count % BDAS_INTX_INTERVAL == 0) {
                double n = (double)_bdas_intx_count;
                double total = (_bdas_intx_setup_us + _bdas_intx_lightswitch_us + _bdas_intx_exitcheck_us + _bdas_intx_defyield_us + _bdas_intx_brakecomp_us) / n;
                printf("  [BENCH intx_handle] Avg over %d calls: Setup=%.3f us | LightSwitch=%.3f us | ExitCheck=%.3f us | DefYield=%.3f us | BrakeComp=%.3f us | Total=%.3f us\n",
                    _bdas_intx_count,
                    _bdas_intx_setup_us / n,
                    _bdas_intx_lightswitch_us / n,
                    _bdas_intx_exitcheck_us / n,
                    _bdas_intx_defyield_us / n,
                    _bdas_intx_brakecomp_us / n,
                    total);
            }
            #endif
        }
    } else {
        das->smart_das_intersection_approach_engaged = false; // reset the flag
    }

    return accel;
}

static const double _speed_limit_multipliers[SMART_DAS_DRIVING_STYLES_COUNT] = {
    [SMART_DAS_DRIVING_STYLE_NO_RULES] = 1.5,
    [SMART_DAS_DRIVING_STYLE_RECKLESS] = 1.25,
    [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = 1.1,
    [SMART_DAS_DRIVING_STYLE_NORMAL] = 1.0,
    [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = 0.9
};

// Precomputed mph-to-mps constants for speed limit adjustments
static const MetersPerSecond _mph5  = 5.0  * (1609.34 / 3600.0);
static const MetersPerSecond _mph10 = 10.0 * (1609.34 / 3600.0);

static MetersPerSecond determine_preferred_speed_limit(Map* map, const Lane* lane, SmartDASDrivingStyle style) {
    MetersPerSecond lane_speed_limit = lane->speed_limit;

    if (!lane->is_at_intersection) {
        Road* road = &map->roads[lane->road_id];
        int num_lanes = road->num_lanes;
        int idx = lane->index_in_road;

        if (num_lanes == 2 || num_lanes == 3) {
            if (idx == 0) lane_speed_limit += _mph5;              // leftmost = passing lane
            if (num_lanes == 3 && idx == num_lanes - 1) lane_speed_limit -= _mph5; // rightmost
        } else if (num_lanes == 4) {
            if (idx == 0) lane_speed_limit += _mph10;             // leftmost
            else if (idx == 2) lane_speed_limit += _mph5;         // 2nd from right (matches original index==2)
            else if (idx == 3) lane_speed_limit -= _mph5;         // rightmost
        }
        // for more than 4 lanes, no adjustments
    }

    return _speed_limit_multipliers[style] * lane_speed_limit;
}


static const Seconds _smart_das_thws[SMART_DAS_DRIVING_STYLES_COUNT] = {
    [SMART_DAS_DRIVING_STYLE_NO_RULES] = 0.5,
    [SMART_DAS_DRIVING_STYLE_RECKLESS] = 1.0,
    [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = 2.0,
    [SMART_DAS_DRIVING_STYLE_NORMAL] = 3.0,
    [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = 4.0
};

static const Meters _smart_das_buffers[SMART_DAS_DRIVING_STYLES_COUNT] = {
    [SMART_DAS_DRIVING_STYLE_NO_RULES] = 0.5,
    [SMART_DAS_DRIVING_STYLE_RECKLESS] = 1.0,
    [SMART_DAS_DRIVING_STYLE_AGGRESSIVE] = 2.0,
    [SMART_DAS_DRIVING_STYLE_NORMAL] = 3.0,
    [SMART_DAS_DRIVING_STYLE_DEFENSIVE] = 4.0
};

void driving_assistant_smart_update_das_variables(DrivingAssistant* das, Car* car, Simulation* sim, SituationalAwareness* sa) {
    if (das->smart_das) {

        // set the THW and speed target based on driving style
        das->speed_target = determine_preferred_speed_limit(sim_get_map(sim), sa->lane, das->smart_das_driving_style);
        das->speed_target = fmin(das->speed_target, car->capabilities.top_speed); // cap speed target to car's max speed
        das->thw = _smart_das_thws[das->smart_das_driving_style];
        das->buffer = _smart_das_buffers[das->smart_das_driving_style];
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

    #ifdef BENCHMARK_DAS_CONTROL
    double _bdas_t0 = _bdas_get_time_us();
    #endif
    situational_awareness_build(sim, car->id);  // ensure up-to-date situational awareness
    SituationalAwareness* sa = sim_get_situational_awareness(sim, car_get_id(car));
    #ifdef BENCHMARK_DAS_CONTROL
    double _bdas_t1 = _bdas_get_time_us();
    #endif

    if (das->smart_das) {

        driving_assistant_smart_update_das_variables(das, car, sim, sa);

        // ------------- Intersection handling (it may adjust speed target to smoothly transition to the intersection lane when not stopping/yielding or briefly speeding to jump yellow) ------------------------
        MetersPerSecondSquared accel_intersection = driving_assistant_smart_das_handle_intersection(das, car, sim, sa);
        accel = fmin(accel, accel_intersection);
    }

    das->speed_target = fmin(das->speed_target, car->capabilities.top_speed); // cap speed target to car's max speed
    #ifdef BENCHMARK_DAS_CONTROL
    double _bdas_t2 = _bdas_get_time_us();
    #endif

    // ----------- Speed adjustment (general) ------------------------
    MetersPerSecondSquared accel_speed_adjust = das->use_linear_speed_control ? car_compute_acceleration_adjust_speed_linear(car, das->speed_target, das->use_preferred_accel_profile) : car_compute_acceleration_adjust_speed(car, das->speed_target, das->use_preferred_accel_profile); // works for negative speed targets as well
    accel = fmin(accel, accel_speed_adjust);

    // ----------- Speed adjustment (approaching end of lane to a lower speed limit lane, unless going straight into an intersection -- where we may speed up to make light and the speed limit is usually anyway the same) ------------
    const Lane* next_lane = sa->lane_next_after_turn[das->turn_intent];
    if (next_lane) {
        MetersPerSecond next_speed_target = determine_preferred_speed_limit(sim_get_map(sim), next_lane, das->smart_das_driving_style);
        if (next_speed_target < sa->speed && !(sa->is_an_intersection_upcoming && das->turn_intent == INDICATOR_NONE)) {
            Meters position_target = sa->lane_progress_m + sa->distance_to_end_of_lane;
            MetersPerSecond next_speed_target = determine_preferred_speed_limit(sim_get_map(sim), next_lane, das->smart_das_driving_style);
            MetersPerSecondSquared accel_approach_end_of_lane = car_compute_acceleration_chase_target(car, position_target, next_speed_target, meters(1), sa->lane->speed_limit, das->use_preferred_accel_profile);
            accel = fmin(accel, accel_approach_end_of_lane);
        }
    }

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

    // ------------- Follow lead vehicle ------------------------
    Meters distance_to_lead;
    MetersPerSecond speed_lead;
    MetersPerSecondSquared accel_lead;
    bool lead_perception_cached = false;
    if (das->follow_assistance && sa->nearby_vehicles.lead) {
        perceive_lead_vehicle(car, sim, sa, &distance_to_lead, &speed_lead, &accel_lead);
        lead_perception_cached = true;
        Meters distance_to_lead_bumper_to_bumper = distance_to_lead - (car_get_length(car) + car_get_length(sa->nearby_vehicles.lead)) / 2.0; // bumper to bumper distance
        Meters target_distance_bumper_to_bumper = das->buffer + fmax(sa->speed, 0) * das->thw;
        Meters position_delta = distance_to_lead_bumper_to_bumper - target_distance_bumper_to_bumper;
        MetersPerSecond speed_target = speed_lead; // we want to match the lead vehicle's speed
        if (sa->almost_stopped) {
            if (sa->speed >= 0) {
                if (from_feet(1) < distance_to_lead_bumper_to_bumper && distance_to_lead_bumper_to_bumper < 1.5 * das->buffer) {
                    position_delta = 0; // if we are almost stopped and within a reasonable range of the target distance, don't adjust position target to prevent microadjustments
                    speed_target = 0;  // and stop or stay stopped
                }
            } else {
                if (distance_to_lead_bumper_to_bumper > das->buffer) {
                    position_delta = 0; // if we are almost stopped after reversing and the lead is not too close, don't adjust position target to prevent microadjustments
                    speed_target = 0;   // and stop or stay stopped
                }
            }
        }
        Meters position_target = sa->lane_progress_m + position_delta;
        Meters overshoot_buffer = sa->speed >= 0 ? distance_to_lead_bumper_to_bumper - from_feet(1) : from_feet(1);
        MetersPerSecondSquared accel_follow = car_compute_acceleration_chase_target(car, position_target, speed_target, overshoot_buffer, das->speed_target, das->use_preferred_accel_profile);
        accel = fmin(accel, accel_follow);
    }

    // Account for human-like delay in adjusting acceleration (gas/brake pedal presses):
    #ifdef BENCHMARK_DAS_CONTROL
    double _bdas_t3 = _bdas_get_time_us();
    #endif

    Seconds dt = sim_get_dt(sim);
    MetersPerSecondSquared accel_prev = car_get_acceleration(car);

    bool in_forward_gear = sa->going_forward;
    bool in_reverse_gear = sa->reversing;
    bool stopped = sa->stopped;
    bool gas_pedal_pressed = ((in_forward_gear || stopped) && accel_prev > 1e-6) || (in_reverse_gear && accel_prev < -1e-6);
    bool brake_pedal_pressed = ((in_forward_gear || stopped) && accel_prev < -1e-6) || (in_reverse_gear && accel_prev > 1e-6);
    bool no_pedals_pressed = !gas_pedal_pressed && !brake_pedal_pressed;
    bool trying_gas_pedal = ((in_forward_gear || stopped) && accel > 1e-6) || ((in_reverse_gear || stopped) && accel < -1e-6);
    bool trying_brake_pedal = (in_forward_gear && accel < -1e-6) || (in_reverse_gear && accel > 1e-6);
    bool trying_no_pedals = !trying_gas_pedal && !trying_brake_pedal;

    MetersPerSecondSquared pressing_gas_harder_max_change = car->capabilities.accel_profile.max_acceleration * dt / HUMAN_TIME_TO_PRESS_FULL_PEDAL;
    MetersPerSecondSquared releasing_gas_max_change = car->capabilities.accel_profile.max_acceleration * dt / HUMAN_TIME_TO_RELEASE_FULL_PEDAL;
    MetersPerSecondSquared pressing_brake_harder_max_change = car->capabilities.accel_profile.max_deceleration * dt / HUMAN_TIME_TO_PRESS_FULL_PEDAL;
    MetersPerSecondSquared releasing_brake_max_change = car->capabilities.accel_profile.max_deceleration * dt / HUMAN_TIME_TO_RELEASE_FULL_PEDAL;

    // consider all combinations
    if (in_forward_gear) {
        if (gas_pedal_pressed) {
            if (trying_gas_pedal) {
                double max_change = accel > accel_prev ? pressing_gas_harder_max_change : releasing_gas_max_change;
                accel = accel_prev + fclamp(accel - accel_prev, -max_change, max_change);
            } else if (trying_brake_pedal || trying_no_pedals) {
                // first, we need to go through no pedals and set accel to zero
                MetersPerSecondSquared max_change_gas_pedal = releasing_gas_max_change;
                accel = accel_prev - max_change_gas_pedal;
                if (accel < 0) {
                    accel = 0;  // release gas pedal fully
                }
            }
        } else if (brake_pedal_pressed) {
            if (trying_brake_pedal) {
                double max_change = accel < accel_prev ? pressing_brake_harder_max_change : releasing_brake_max_change;
                accel = accel_prev + fclamp(accel - accel_prev, -max_change, max_change);
            } else if (trying_gas_pedal || trying_no_pedals) {
                // first, we need to go through no pedals and set accel to zero
                MetersPerSecondSquared max_change_brake_pedal = releasing_brake_max_change;
                accel = accel_prev + max_change_brake_pedal;
                if (accel > 0) {
                    accel = 0;  // release brake pedal fully
                }
            }
        } else if (no_pedals_pressed) {    // no pedals pressed
            if (trying_no_pedals) {
                accel = 0;      // already no pedals, stay there
            } else {
                double foot_still_in_air_probability = 1 - dt / HUMAN_AVERAGE_DELAY_SWITCHING_PEDALS;
                bool foot_switched = rand_0_to_1() > foot_still_in_air_probability;
                if (foot_switched) {
                    if (trying_gas_pedal) {
                        accel = fmin(pressing_gas_harder_max_change, accel);
                    } else if (trying_brake_pedal) {
                        accel = fmax(-pressing_brake_harder_max_change, accel);
                    }
                } else {
                    accel = 0; // remain with no pedals
                }
            }
        }
    } else if (in_reverse_gear) {
        pressing_gas_harder_max_change = car->capabilities.accel_profile.max_acceleration_reverse_gear * dt / HUMAN_TIME_TO_PRESS_FULL_PEDAL;
        releasing_gas_max_change = car->capabilities.accel_profile.max_acceleration_reverse_gear * dt / HUMAN_TIME_TO_RELEASE_FULL_PEDAL;
        if (gas_pedal_pressed) {
            if (trying_gas_pedal) {
                double max_change = accel < accel_prev ? pressing_gas_harder_max_change : releasing_gas_max_change;
                accel = accel_prev + fclamp(accel - accel_prev, -max_change, max_change);
            } else if (trying_brake_pedal || trying_no_pedals) {
                // first, we need to go through no pedals and set accel to zero
                MetersPerSecondSquared max_change_gas_pedal = releasing_gas_max_change;
                accel = accel_prev + max_change_gas_pedal;
                if (accel > 0) {
                    accel = 0;  // release gas pedal fully
                }
            }
        } else if (brake_pedal_pressed) {
            if (trying_brake_pedal) {
                double max_change = accel > accel_prev ? pressing_brake_harder_max_change : releasing_brake_max_change;
                accel = accel_prev + fclamp(accel - accel_prev, -max_change, max_change);
            } else if (trying_gas_pedal || trying_no_pedals) {
                // first, we need to go through no pedals and set accel to zero
                MetersPerSecondSquared max_change_brake_pedal = releasing_brake_max_change;
                accel = accel_prev - max_change_brake_pedal;
                if (accel < 0) {
                    accel = 0;  // release brake pedal fully
                }
            }
        } else if (no_pedals_pressed) {
            if (trying_no_pedals) {
                accel = 0;      // already no pedals, stay there
            } else {
                double foot_still_in_air_probability = 1 - dt / HUMAN_AVERAGE_DELAY_SWITCHING_PEDALS;
                bool foot_switched = rand_0_to_1() > foot_still_in_air_probability;
                if (foot_switched) {
                    if (trying_gas_pedal) {
                        accel = fmax(-pressing_gas_harder_max_change, accel);
                    } else if (trying_brake_pedal) {
                        accel = fmin(pressing_brake_harder_max_change, accel);
                    }
                } else {
                    accel = 0; // remain with no pedals
                }
            }
        }
    } else { // fully stopped right now.
        if (accel < 0) {
            pressing_gas_harder_max_change = car->capabilities.accel_profile.max_acceleration_reverse_gear * dt / HUMAN_TIME_TO_PRESS_FULL_PEDAL;   // adjust for reverse gear
        }
        if (trying_gas_pedal) {
            accel = accel_prev + fclamp(accel - accel_prev, -pressing_gas_harder_max_change, pressing_gas_harder_max_change);
        } else if (trying_brake_pedal) {
            accel = accel_prev + fclamp(accel - accel_prev, -pressing_brake_harder_max_change, pressing_brake_harder_max_change);
        } else {
            accel = 0; // no input
        }
    }



    // ---- AEB logic overrides everything else ----
    #ifdef BENCHMARK_DAS_CONTROL
    double _bdas_t4 = _bdas_get_time_us();
    #endif

    // auto engagement conditions
    bool execute_aeb = false;
    Meters best_case_braking_gap;
    if (das->aeb_assistance && sa->nearby_vehicles.lead) {
        if (das->aeb_in_progress) {
            execute_aeb = true; // once AEB is engaged, stay engaged until lead is gone or we are stopped to prevent oscillation
        } else if (sa->speed >= AEB_ENGAGE_MIN_SPEED_MPS) {
            best_case_braking_gap = compute_best_case_braking_gap(sim, car, sa, lead_perception_cached ? &distance_to_lead : NULL, lead_perception_cached ? &speed_lead : NULL, lead_perception_cached ? &accel_lead : NULL); // todo: fix this fn
            execute_aeb = best_case_braking_gap < AEB_ENGAGE_BEST_CASE_BRAKING_GAP_M;
        }
    }

    // execute AEB
    if (execute_aeb) {
        if (!das->aeb_in_progress) {
            das->aeb_in_progress = true; // Start AEB if not already in progress
            // LOG_DEBUG("AEB engaged for car %d. Best-case gap: %.2f feet, speed: %.2f mph", car->id, to_feet(best_case_braking_gap), to_mph(car_get_speed(car)));
        }
        MetersPerSecondSquared accel_aeb = car_compute_acceleration_stop_max_brake(car, das->use_preferred_accel_profile);
        accel = accel_aeb; // AEB overrides all other acceleration commands
        LOG_TRACE("AEB in progress for car %d. Best-case gap: %.2f feet, speed: %.2f mph", car->id, to_feet(best_case_braking_gap), to_mph(car_get_speed(car)));
        // NOTE: Forward crash is still possible if the lead vehicle is backing into us, but we don't handle that case here since that is not about emergency braking.
    }

    

    // ----------- Merge ----------------------
    #ifdef BENCHMARK_DAS_CONTROL
    double _bdas_t5 = _bdas_get_time_us();
    #endif


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
    #ifdef BENCHMARK_DAS_CONTROL
    double _bdas_t6 = _bdas_get_time_us();
    #endif

    car_set_acceleration(car, accel);
    car_set_indicator_lane(car, lane_indicator);
    car_set_indicator_turn(car, turn_indicator);
    car_set_request_indicated_lane(car, request_lane_change);
    car_set_request_indicated_turn(car, request_turn);
    car_set_auto_turn_off_indicators(car, true); // Needed to detect successful lane/turn changes

    #ifdef BENCHMARK_DAS_CONTROL
    double _bdas_t7 = _bdas_get_time_us();
    _bdas_sa_build_us   += _bdas_t1 - _bdas_t0;
    _bdas_smart_das_us  += _bdas_t2 - _bdas_t1;
    _bdas_speed_ctrl_us += _bdas_t3 - _bdas_t2;
    _bdas_pedal_sim_us  += _bdas_t4 - _bdas_t3;
    _bdas_aeb_us        += _bdas_t5 - _bdas_t4;
    _bdas_merge_turn_us += _bdas_t6 - _bdas_t5;
    _bdas_apply_us      += _bdas_t7 - _bdas_t6;
    _bdas_call_count++;
    if (_bdas_call_count % BDAS_INTERVAL == 0) {
        double n = (double)_bdas_call_count;
        double total = (_bdas_sa_build_us + _bdas_smart_das_us + _bdas_speed_ctrl_us + _bdas_pedal_sim_us + _bdas_aeb_us + _bdas_merge_turn_us + _bdas_apply_us) / n;
        printf("[BENCH das_control] Avg over %d calls: SA_Build=%.3f us | SmartDAS=%.3f us | SpeedCtrl+Follow=%.3f us | PedalSim=%.3f us | AEB=%.3f us | MergeTurn=%.3f us | Apply=%.3f us | Total=%.3f us\n",
            _bdas_call_count,
            _bdas_sa_build_us / n,
            _bdas_smart_das_us / n,
            _bdas_speed_ctrl_us / n,
            _bdas_pedal_sim_us / n,
            _bdas_aeb_us / n,
            _bdas_merge_turn_us / n,
            _bdas_apply_us / n,
            total);
    }
    #endif

    return true; // Indicating successful control without error
}


void driving_assistant_post_sim_step(DrivingAssistant* das, Car* car, Simulation* sim) {
    if (!car || !sim || !das) {
        LOG_ERROR("Invalid car, simulation, or driving assistant provided to driving_assistant_post_step");
        return; // Early return if input validation fails
    }
    #ifdef BENCHMARK_DAS_POST_STEP
    double _bpost_t0 = _bdas_get_time_us();
    #endif
    situational_awareness_build(sim, car_get_id(car)); // Ensure situational awareness is up-to-date
    SituationalAwareness* sa = sim_get_situational_awareness(sim, car_get_id(car));
    #ifdef BENCHMARK_DAS_POST_STEP
    double _bpost_t1 = _bdas_get_time_us();
    #endif

    if (das->aeb_in_progress) {
        das->speed_target = car_get_speed(car); // Update speed target to current speed if AEB is in progress, so that once AEB is disengaged (automatic or manual), the car will not accelerate or decelerate unexpectedly
    }

    if (das->aeb_assistance && das->aeb_in_progress) {
        bool disengagement_conditions_met = false;
        Meters best_case_braking_gap = -INFINITY;  // Initialize to invalid value
        if (sa->nearby_vehicles.lead == NULL) {
            disengagement_conditions_met = true; // If lead vehicle is gone, disengage AEB
        } else if (sa->speed < AEB_DISENGAGE_MAX_SPEED_MPS) {
            disengagement_conditions_met = true;
        } else {
            best_case_braking_gap = compute_best_case_braking_gap(sim, car, sim_get_situational_awareness(sim, car_get_id(car)), NULL, NULL, NULL); // recompute with up-to-date SA after the sim step
            if (best_case_braking_gap > AEB_DISENGAGE_BEST_CASE_BRAKING_GAP_M) {
                disengagement_conditions_met = true;
            }
        }
        if (disengagement_conditions_met) {
            LOG_DEBUG("AEB disengaged for car %d. Best case braking gap: %.2f feet, speed: %.2f mph", car->id, to_feet(best_case_braking_gap), to_mph(car_get_speed(car)));
            das->aeb_in_progress = false; // Reset AEB state if auto disengagement conditions are met
        }
    }
    #ifdef BENCHMARK_DAS_POST_STEP
    double _bpost_t2 = _bdas_get_time_us();
    #endif

    // update intersection approach engagement flag
    if (das->smart_das) {
        das->smart_das_intersection_approach_engaged = driving_assistant_smart_das_should_engage_intersection_approach(das, car, sim, sa);
    }
    #ifdef BENCHMARK_DAS_POST_STEP
    double _bpost_t3 = _bdas_get_time_us();
    #endif

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

    #ifdef BENCHMARK_DAS_POST_STEP
    double _bpost_t4 = _bdas_get_time_us();
    _bpost_sa_build_us      += _bpost_t1 - _bpost_t0;
    _bpost_aeb_disengage_us += _bpost_t2 - _bpost_t1;
    _bpost_intx_update_us   += _bpost_t3 - _bpost_t2;
    _bpost_cleanup_us       += _bpost_t4 - _bpost_t3;
    _bpost_call_count++;
    if (_bpost_call_count % BPOST_INTERVAL == 0) {
        double n = (double)_bpost_call_count;
        double total = (_bpost_sa_build_us + _bpost_aeb_disengage_us + _bpost_intx_update_us + _bpost_cleanup_us) / n;
        printf("[BENCH das_post_step] Avg over %d calls: SA_Build=%.3f us | AEB_Disengage=%.3f us | Intx_Update=%.3f us | Cleanup=%.3f us | Total=%.3f us\n",
            _bpost_call_count,
            _bpost_sa_build_us / n,
            _bpost_aeb_disengage_us / n,
            _bpost_intx_update_us / n,
            _bpost_cleanup_us / n,
            total);
    }
    #endif
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