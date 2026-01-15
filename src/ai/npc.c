#include "ai.h"
#include "map.h"
#include "math.h"
#include "utils.h"
#include "logging.h"
#include "sim.h"
#include <stdio.h>
#include <stdlib.h>

#define MAX_CARS 1024
void npc_car_make_decisions(Car* self, Simulation* sim) {
    Seconds dt = sim_get_dt(sim);
    SituationalAwareness* situation = sim_get_situational_awareness(sim, self->id);
    const Lane* lane = situation->lane;
    // printf("\nOn lane: %p\n", lane);
    Meters car_speed = car_get_speed(self);
    Meters car_length = car_get_length(self);
    Meters car_half_length = car_length / 2;
    // Meters lane_length = lane_get_length(lane);
    // Meters lane_progress = situation->lane_progress;
    Meters lane_progress_m = situation->lane_progress_m;
    Meters car_position = lane_progress_m;          // alias for lane_progress_m
    Meters distance_to_stop_line = situation->distance_to_end_of_lane_from_leading_edge - STOP_LINE_BUFFER_METERS; // if we had to stop at the end of the lane, how far would we be from the stop line?
    MetersPerSecond speed_cruise = lane_get_speed_limit(lane) + self->preferences.average_speed_offset;  // incorporate preference into speed limit
    speed_cruise = fmax(speed_cruise, CREEP_SPEED); // ensure at least 1 mph speed

    // the following parameters will determine the outputs
    Meters position_target;
    Meters position_target_overshoot_buffer;    // car will try its best to not overshoot target + this buffer, by applying max capable braking.
    MetersPerSecond speed_at_target;
    CarIndicator turn_indicator;
    CarIndicator lane_change_indicator;

    // Some random calls. It is always good to sample same number of randoms in each decision-making cycle and make them independent of branching (which depends on floating point comparisons), to make decisions consistent across platforms / compilers, as different platforms / compilers may have different floating point comparison results due to precision issues.
    double r1 = rand_0_to_1();
    double r2 = rand_0_to_1();
    double r3 = rand_0_to_1();
    CarIndicator rand_turn = turn_sample_possible(situation);
    CarIndicator rand_lane_change = lane_change_sample_possible(situation);

    bool indicate_only = false; // whether to indicate only, without requesting the sim to execute the lane change or turn.

    if (situation->is_an_intersection_upcoming) {
        // Whe approaching an intersection, turn and lane change decision will depend on distance to intersection. If we are far off, we can randomly change lanes or change turn intent. Else, we should not change lanes or change turn intent.
        if (situation->is_approaching_end_of_lane || situation->braking_distance.preferred_smooth >= 1.1 * distance_to_stop_line) {
            // Should not change lanes or change turn intent.
            // printf("0. Not changing lanes or turn intent\n");
            turn_indicator = car_get_indicator_turn(self);  // whatever it is
            if (!situation->is_turn_possible[turn_indicator]) {
                turn_indicator = rand_turn;
            }
            lane_change_indicator = INDICATOR_NONE;
        } else {
            // We are not close to the end of the lane.
            turn_indicator = (r1 < dt / 4) ? rand_turn : car_get_indicator_turn(self);
            if (!situation->is_turn_possible[turn_indicator]) {
                turn_indicator = rand_turn;
            }
            // printf("Turn indicator: %d\n", turn_indicator);

            CarIndicator current_lane_indicator = car_get_indicator_lane(self);
            if (current_lane_indicator == INDICATOR_NONE) {
                lane_change_indicator = (r2 < dt / 5) ? rand_lane_change : INDICATOR_NONE;  // about every 5 seconds we have been travelling straight without indicating, decide to change lanes randomly
            } else {
                lane_change_indicator = (r2 < dt / 5) ? INDICATOR_NONE : current_lane_indicator; // about every 5 seconds we have been indicating unsuccessfully, decide to cancel the lane change indicator
            }

            // printf("Lane change indicator: %d\n", lane_change_indicator);
            // printf("0. Intersection upcoming: sampled turn indicator and lane change indicator: %d, %d\n", turn_indicator, lane_change_indicator);
        }
    } else if (situation->is_approaching_dead_end) {
        // printf("NPC P1: C2\n");
        // We are not approaching an intersection, but rather a dead end. We should try to move to left most lane if this road merges into something. If not, maybe there is an exit available and or there is an upcoming exit? Then we should try to move to right most lane. If none of this is possible, the we should just stop at the right most lane.
        turn_indicator = INDICATOR_NONE;
        if (situation->is_on_entry_ramp) {
            lane_change_indicator = INDICATOR_LEFT;
        } else if (situation->is_exit_available || situation->is_exit_available_eventually) {
            lane_change_indicator = INDICATOR_RIGHT;
        } else {
            lane_change_indicator = INDICATOR_RIGHT;
        }
        // printf("0. Dead end: lane change indicator: %d\n", lane_change_indicator);
    } else {
        // We are not approaching an intersection or a dead end. We can randomly change lanes (and turn intent is not relevant).
        turn_indicator = INDICATOR_NONE;

        CarIndicator current_lane_indicator = car_get_indicator_lane(self);
        if (current_lane_indicator == INDICATOR_NONE) {
            lane_change_indicator = (r3 < dt / 5) ? rand_lane_change : INDICATOR_NONE;  // about every 5 seconds we have been travelling straight without indicating, decide to change lanes randomly
        } else {
            lane_change_indicator = (r3 < dt / 5) ? INDICATOR_NONE : current_lane_indicator; // about every 5 seconds we have been indicating unsuccessfully, decide to cancel the lane change indicator
        }

        if (lane_change_indicator != INDICATOR_NONE) {
            // printf("0. Not an intersection or dead end: lane change indicator: %d\n", lane_change_indicator);
        }
        if (situation->is_exit_possible) {
            // printf("Exit is possible. Indicating right.\n");
            lane_change_indicator = INDICATOR_RIGHT;
        }
    }

    // cancel lane change if it is dangerous
    if (car_is_lane_change_dangerous(self, sim, situation, lane_change_indicator, self->preferences.time_headway, from_feet(3))) {
        indicate_only = true; // we will just indicate the lane change, but not request it
        // keep the indicator as it is, but do not request the lane change
        // printf("Lane change is dangerous.\n");
    }

    // Next up is to determine the acceleration. If there is a vehicle ahead, then we just need to follow it with its speed. If nothing ahead, we just cruise. But we should override to a slower acceleration if we are close to an intersection (and approaching yellow / red), or approaching a dead end. It is possible that the lead vehicle is going faster than it should be!
    MetersPerSecondSquared accel = 0.0;
    
    if (situation->is_vehicle_ahead) {
        // printf("1. Lead vehicle detected\n");
        // Decide how much distance to maintain. Depends on preferred time headway.
        // TODO: For worse conditions, it should be higher.
        Meters target_distance_from_next_car = fmax(car_speed, 0) * self->preferences.time_headway + from_feet(3);
        Meters car_lengths_offset = car_half_length + car_get_length(situation->lead_vehicle) / 2;
        target_distance_from_next_car += car_lengths_offset;
        position_target = car_position + situation->distance_to_lead_vehicle - target_distance_from_next_car;
        position_target_overshoot_buffer = target_distance_from_next_car - car_lengths_offset - from_feet(1.0); // let's call being within 1.0 feet of the next car as a crash.
        speed_at_target = car_get_speed(situation->lead_vehicle);
        accel = car_compute_acceleration_chase_target(self, position_target, speed_at_target, position_target_overshoot_buffer, speed_cruise, true);

        // If are already within ttc_threshold of the lead vehicle, we should suddenly change the lane to avoid rear-ending it.
        Meters bumper_to_bumper_distance = situation->distance_to_lead_vehicle - car_lengths_offset;
        MetersPerSecond relative_speed = car_speed - car_get_speed(situation->lead_vehicle);
        Seconds lead_ttc = bumper_to_bumper_distance / relative_speed;
        if (lead_ttc < 0) {
            lead_ttc = INFINITY;    // we are receding relative to the lead vehicle
        }
        Seconds ttc_threshold = 1.0; // 1 second time to collision threshold
        if (lead_ttc < ttc_threshold) {
            // printf("Emergency lane change maneuver to avoid rear-ending the next car.\n");
            // go to the side where time-to-collision is higher
            bool can_go_left = situation->is_lane_change_left_possible && situation->nearby_vehicles.colliding[INDICATOR_LEFT].car == NULL && situation->nearby_vehicles.ahead[INDICATOR_LEFT].time_to_collision > ttc_threshold;
            bool can_go_right = situation->is_lane_change_right_possible && situation->nearby_vehicles.colliding[INDICATOR_RIGHT].car == NULL && situation->nearby_vehicles.ahead[INDICATOR_RIGHT].time_to_collision > ttc_threshold;
            if (can_go_left && can_go_right) {
                lane_change_indicator = (situation->nearby_vehicles.ahead[INDICATOR_LEFT].time_to_collision > situation->nearby_vehicles.ahead[INDICATOR_RIGHT].time_to_collision) ? INDICATOR_LEFT : INDICATOR_RIGHT;
                indicate_only = false; // we will request the lane change asap
            } else if (can_go_left) {
                lane_change_indicator = INDICATOR_LEFT;
                indicate_only = false; // we will request the lane change asap
            } else if (can_go_right) {
                lane_change_indicator = INDICATOR_RIGHT;
                indicate_only = false; // we will request the lane change asap
            } else {
                // no better lane change possible, just brake hard. Keep the lane change indicator & request as they are.
                accel = car_compute_acceleration_stop_max_brake(self, false);
            }
        }
    } else {
        if (!situation->is_approaching_dead_end && (situation->is_approaching_end_of_lane || situation->braking_distance.preferred_smooth >= distance_to_stop_line)) {
            // start changing speed to match speed limit of the next lane
            accel = car_compute_acceleration_cruise(self, fmax(lane_get_speed_limit(situation->lane_next_after_turn[turn_indicator]) + self->preferences.average_speed_offset, CREEP_SPEED), true);
        } else {
            accel = car_compute_acceleration_cruise(self, speed_cruise, true);
        }
    }

    bool should_brake_for_full_stop = false;    // e.g., approaching red light or stop sign. Target the stop line.
    bool should_brake_for_yield = false;        // e.g., yielding at intersection. Stop by end of lane if needed (kinda creeping beyond stop line).
    
    if (situation->is_approaching_dead_end) {
        should_brake_for_full_stop = true;
    }

    if (situation->is_an_intersection_upcoming) {
        TrafficLight light = situation->light_for_turn[turn_indicator];
        bool is_emergency_braking_possible = (situation->braking_distance.capable < situation->distance_to_end_of_lane_from_leading_edge);
        bool is_comfy_braking_possible = (situation->braking_distance.preferred_smooth < distance_to_stop_line);

        switch (light) {
        case TRAFFIC_LIGHT_STOP_FCFS:
            // Must first stop or be stopping. After that, we should check if it is our turn based on first-come-first-serve.
            if (is_emergency_braking_possible && (situation->distance_to_end_of_lane_from_leading_edge > STOP_LINE_BUFFER_METERS - meters(0.1)) && car_get_speed(self) > STOP_SPEED_THRESHOLD) { // can stop before lane end, but it still moving and not yet at the stop line (which it will really reach in infinite time (if comfy braking possible) due to asymptotic approach of PD control, so we use a small threshold)
                should_brake_for_full_stop = true;
            } else if (fabs(car_get_speed(self)) <= STOP_SPEED_THRESHOLD) { // If stopped or barely moving
                bool all_clear = !intersection_is_any_car_on_intersection(situation->intersection, sim);
                if (situation->is_my_turn_at_stop_sign_fcfs && all_clear) {
                    should_brake_for_full_stop = false; // Our turn
                } else {
                    should_brake_for_full_stop = true; // Not permitted, remain stopped.
                }
            } else { // Approaching but cannot stop in time or already past line and moving.
                 should_brake_for_full_stop = false;
            }
            break;
        case TRAFFIC_LIGHT_RED:
            should_brake_for_full_stop = true;
            if (!is_emergency_braking_possible) {
                should_brake_for_full_stop = false;
            }
            break;
        case TRAFFIC_LIGHT_YELLOW:
            if (is_comfy_braking_possible) {
                should_brake_for_full_stop = true;
            } else {
                should_brake_for_full_stop = !self->preferences.run_yellow_light;
                if (should_brake_for_full_stop && !is_emergency_braking_possible) {
                    should_brake_for_full_stop = false;
                }
            }
            break;

        case TRAFFIC_LIGHT_GREEN_YIELD:
            should_brake_for_full_stop = false;
            if (car_has_something_to_yield_to_at_intersection(self, sim, situation, turn_indicator)) {
                should_brake_for_yield = true;
            } else {
                should_brake_for_yield = false;
            }
            break;
        case TRAFFIC_LIGHT_YELLOW_YIELD:
            if (is_comfy_braking_possible) {
                should_brake_for_full_stop = true;
            } else {
                should_brake_for_full_stop = !self->preferences.run_yellow_light;
                if (should_brake_for_full_stop && !is_emergency_braking_possible) {
                    should_brake_for_full_stop = false;
                }
            }
            if (!should_brake_for_full_stop) {  // planning to go through
                if (is_emergency_braking_possible && car_has_something_to_yield_to_at_intersection(self, sim, situation, turn_indicator)) {
                    should_brake_for_yield = true;  // yield if possible and needed
                } else {
                    should_brake_for_yield = false; 
                }
            }
            break;
        case TRAFFIC_LIGHT_RED_YIELD:
            if (is_emergency_braking_possible) {
                bool not_stopped_yet = (situation->distance_to_end_of_lane_from_leading_edge > STOP_LINE_BUFFER_METERS - meters(0.1)) && car_get_speed(self) > STOP_SPEED_THRESHOLD;
                if (not_stopped_yet) {
                    should_brake_for_full_stop = true;
                } else {
                    should_brake_for_full_stop = false;
                }
                if (car_has_something_to_yield_to_at_intersection(self, sim, situation, turn_indicator)) {
                    should_brake_for_yield = true;
                } else {
                    should_brake_for_yield = false;
                }
            } else {
                should_brake_for_full_stop = false;
                should_brake_for_yield = false;
            }
            break;
        
        case TRAFFIC_LIGHT_YIELD: // General yield sign
            if (car_has_something_to_yield_to_at_intersection(self, sim, situation, turn_indicator)) {
                should_brake_for_yield = true;
            } else {
                should_brake_for_yield = false;
            }
            break;
        case TRAFFIC_LIGHT_GREEN:
        case TRAFFIC_LIGHT_NONE:
        default:
            should_brake_for_full_stop = false;
            should_brake_for_yield = false;
            break;
        }

        // Finally, as you enter the intersection (e.g., going straight), and someone else in front of you just went to a different lane (e.g., turned left), we should brake a bit to let them make some distance. Basically, check all outgoing connections and see if there is a car very close to the intersection in front of us but in a different lane.
        for (int turn_dir = 0; turn_dir < 3; turn_dir++) {
            const Lane* next_lane = situation->lane_next_after_turn[turn_dir];
            if (next_lane != NULL && next_lane != situation->lane_next_after_turn[turn_indicator] && lane_get_num_cars(next_lane) > 0) {
                Car* closest_car = lane_get_car(next_lane, sim, lane_get_num_cars(next_lane) - 1); // last car in the lane is the closest to us
                Meters closest_car_progress = closest_car->lane_progress_meters;
                if (closest_car_progress < car_get_length(closest_car) + car_get_length(self)) {
                    should_brake_for_yield = true;
                }
            }
        }

        if (sim->npc_rogue_factor > 0.0) {
            double rogue_roll = rand_0_to_1();
            if (rogue_roll < sim_get_npc_rogue_factor(sim)) {
                // ignore traffic light
                should_brake_for_full_stop = false;
                should_brake_for_yield = false;
            }
        }
    }

    if (should_brake_for_full_stop || should_brake_for_yield) {
        if (should_brake_for_full_stop) {   // takes precedence
            // Brake smoothly to target stop at the end of the lane
            position_target = car_position + fmax(distance_to_stop_line, meters(0)); // the fmax is needed to prevent reversing if the car has already crossed the stop line (could happen if light turned red while crossing)
            speed_at_target = 0;
            MetersPerSecondSquared accel_to_stop_at_lane_end = car_compute_acceleration_chase_target(self, position_target, speed_at_target, fmax(situation->distance_to_end_of_lane_from_leading_edge - position_target, meters(0)), speed_cruise, true);
            accel = fmin(accel, accel_to_stop_at_lane_end); // take the minimum of the two accelerations
        } else {
            // Brake smoothly to target stop by the end of the lane (i.e., can creep beyond stop line)
            position_target = car_position + fmax(situation->distance_to_end_of_lane_from_leading_edge, meters(0)); // the fmax is needed in case the leading edge is already beyond the lane end once the car moved but the car center is not. If that happened, and now suddenly yield is true due to new traffic, we want to stop right where we are and not reverse.
            speed_at_target = 0;
            MetersPerSecond speed_cruise_for_yield = situation->distance_to_end_of_lane_from_leading_edge <= STOP_LINE_BUFFER_METERS ? fmin(CREEP_SPEED, speed_cruise) : speed_cruise; // limit speed after crossing stop line to 5 mph (creep speed)
            MetersPerSecondSquared accel_to_yield_at_lane_end = car_compute_acceleration_chase_target(self, position_target, speed_at_target, 0, speed_cruise_for_yield, true);
            accel = fmin(accel, accel_to_yield_at_lane_end); // take the minimum of the two accelerations
        }
    }

    // Set the decision variables
    accel = 0.95 * accel + 0.05 * car_get_acceleration(self); // smooth the acceleration
    car_set_acceleration(self, accel);
    car_set_indicator_turn_and_request(self, turn_indicator);
    car_set_indicator_lane(self, lane_change_indicator);
    car_set_request_indicated_lane(self, !indicate_only && lane_change_indicator != INDICATOR_NONE);
}
