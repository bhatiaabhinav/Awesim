#include "ai.h"
#include "map.h"
#include "math.h"
#include "utils.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>


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
    Meters distance_to_stop_line = situation->distance_to_end_of_lane - car_half_length - meters(2); // if we had to stop at the end of the lane, how far would we be from the stop line?
    MetersPerSecond speed_cruise = lane_get_speed_limit(lane) + self->preferences.average_speed_offset;  // incorporate preference into speed limit

    // the following parameters will determine the outputs
    Meters position_target;
    Meters position_target_overshoot_buffer;    // car will try its best to not overshoot target + this buffer, by applying max capable braking.
    MetersPerSecond speed_at_target;
    CarIndictor turn_indicator;
    CarIndictor lane_change_indicator;

    // Some random calls. It is always good to sample same number of randoms in each decision-making cycle and make them independent of branching (which depends on floating point comparisons), to make decisions consistent across platforms / compilers, as different platforms / compilers may have different floating point comparison results due to precision issues.
    double r1 = rand_0_to_1();
    double r2 = rand_0_to_1();
    double r3 = rand_0_to_1();
    CarIndictor rand_turn = turn_sample_possible(situation);
    CarIndictor rand_lane_change = lane_change_sample_possible(situation);

    if (situation->is_approaching_intersection) {
        // Whe approaching an intersection, turn and lane change decision will depend on distance to intersection. If we are far off, we can randomly change lanes or change turn intent. Else, we should not change lanes or change turn intent.
        if (situation->is_close_to_end_of_lane || situation->braking_distance.preferred_smooth >= 1.1 * distance_to_stop_line) {
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
            lane_change_indicator = (r2 < dt / 5) ? rand_lane_change : INDICATOR_NONE;
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
        lane_change_indicator = (r3 < dt) ? rand_lane_change : INDICATOR_NONE;
        if (lane_change_indicator != INDICATOR_NONE) {
            // printf("0. Not an intersection or dead end: lane change indicator: %d\n", lane_change_indicator);
        }
        if (situation->is_exit_possible) {
            // printf("Exit is possible. Indicating right.\n");
            lane_change_indicator = INDICATOR_RIGHT;
        }
    }

    // cancel lane change if it is dangerous
    if (car_is_lane_change_dangerous(self, sim, situation, lane_change_indicator)) {
        lane_change_indicator = INDICATOR_NONE;
        // printf("Lane change is dangerous. Canceling lane change.\n");
    }

    // Next up is to determine the acceleration. If there is a vehicle ahead, then we just need to follow it with its speed. If nothing ahead, we just cruise. But we should override to a slower acceleration if we are close to an intersection (and approaching yellow / red), or approaching a dead end. It is possible that the lead vehicle is going faster than it should be!
    MetersPerSecondSquared accel;
    
    if (situation->is_vehicle_ahead) {
        // printf("1. Lead vehicle detected\n");
        // Decide how much distance to maintain. Let's make it "3-second rule" for now.
        // TODO: For worse conditions, it should be higher.
        // TODO: Incorporate preferences.
        Meters target_distance_from_next_car = fmax(car_speed, 0) * 3 + from_feet(3);
        Meters car_lengths_offset = car_half_length + car_get_length(situation->lead_vehicle) / 2;
        target_distance_from_next_car += car_lengths_offset;
        position_target = car_position + situation->distance_to_lead_vehicle - target_distance_from_next_car;
        position_target_overshoot_buffer = target_distance_from_next_car - car_lengths_offset - from_feet(1.0); // let's call being within 1.0 feet of the next car as a crash.
        speed_at_target = car_get_speed(situation->lead_vehicle);
        accel = car_compute_acceleration_chase_target(self, position_target, speed_at_target, position_target_overshoot_buffer, speed_cruise);
        // accel += situation->lead_vehicle->acceleration; // add the lead vehicle's acceleration to our acceleration to account for the fact that we are following it and it may be accelerating or braking.

        // If are already within 2.0 feet of the lead vehicle, let's try to change lanes to avoid rear-ending it.
        rand_lane_change = lane_change_sample_possible(situation);
        if (situation->distance_to_lead_vehicle - (car_half_length + car_get_length(situation->lead_vehicle) / 2) < from_feet(2.0)) { // TODO: make it depend on more variables
            // printf("Emergency lane change maneuver to avoid rear-ending the next car.\n");
            if (lane_change_indicator == INDICATOR_NONE) {
                lane_change_indicator = rand_lane_change;
                if (lane_change_indicator == INDICATOR_NONE) {
                    // printf("Oh no! No lane change possible. We might rear-end the lead vehicle!\n");
                }
                // TODO: check if we will crash into the side car if we change lanes. If so, then we should not change lanes. Or maybe a side crash is better than a rear-end crash?
            }
        }
    } else {
        if (!situation->is_approaching_dead_end && (situation->is_close_to_end_of_lane || situation->braking_distance.preferred_smooth >= distance_to_stop_line)) {
            // start changing speed to match speed limit of the next lane
            accel = car_compute_acceleration_cruise(self, lane_get_speed_limit(situation->lane_next_after_turn[turn_indicator]) + self->preferences.average_speed_offset);
        } else {
            accel = car_compute_acceleration_cruise(self, speed_cruise);
        }
    }

    bool should_brake = false;
    
    if (situation->is_approaching_dead_end) {
        // printf("3. But, approaching dead end\n");
        // printf("4. Distance to stop line: %f\n", distance_to_stop_line + meters(1));
        should_brake = true;
    }

    if (situation->is_approaching_intersection) {
        TrafficLight light = situation->light_for_turn[turn_indicator];
        bool is_emergency_braking_possible = (situation->braking_distance.capable <= distance_to_stop_line + meters(1) + 1e-6);
        bool is_comfy_braking_possible = (situation->braking_distance.preferred_smooth <= distance_to_stop_line + 1e-6);

        switch (light) {
        case TRAFFIC_LIGHT_STOP:
            should_brake = true;
            if (!is_emergency_braking_possible) {
                should_brake = false;
            }
            break;
        case TRAFFIC_LIGHT_RED:
            should_brake = true;
            if (!is_emergency_braking_possible) {
                should_brake = false;
            }
            break;
        case TRAFFIC_LIGHT_YELLOW:
            if (is_comfy_braking_possible) {
                should_brake = true;
            } else {
                should_brake = !self->preferences.run_yellow_light;
                if (should_brake && !is_emergency_braking_possible) {
                    should_brake = false;
                }
            }
            break;

        case TRAFFIC_LIGHT_GREEN_YIELD:
        case TRAFFIC_LIGHT_YELLOW_YIELD:
            if (light == TRAFFIC_LIGHT_YELLOW_YIELD) {
                bool try_to_stop_for_yellow = true;
                if (is_comfy_braking_possible) {
                    try_to_stop_for_yellow = true;
                } else {
                    try_to_stop_for_yellow = !self->preferences.run_yellow_light;
                    if (try_to_stop_for_yellow && !is_emergency_braking_possible) {
                        try_to_stop_for_yellow = false; 
                    }
                }
                if (try_to_stop_for_yellow) {
                    should_brake = true; 
                    break; 
                }
            }
            if (car_should_yield_at_intersection(self, sim, situation, turn_indicator)) {
                should_brake = true;
            } else {
                should_brake = false; 
            }
            break;

        case TRAFFIC_LIGHT_RED_YIELD:
            // Must first obey the RED aspect (stop or be stopping).
            if (fabs(car_get_speed(self)) > meters(0.5) && // If moving with some speed
                (is_comfy_braking_possible || is_emergency_braking_possible) && // and can stop
                (situation->distance_to_end_of_lane - car_half_length > meters(0.5))) { // and not yet at the line
                should_brake = true; // Enforce stop for the RED part.
            } else if (fabs(car_get_speed(self)) <= meters(0.5)) { // If stopped or barely moving at the line
                // Check if it's a permitted turn (usually right) and if it's clear
                if (situation->is_turn_possible[turn_indicator] &&
                    (turn_indicator == INDICATOR_RIGHT /* Add other conditions if exits, e.g. specific ramp merges */) &&
                    !car_should_yield_at_intersection(self, sim, situation, turn_indicator)) {
                    should_brake = false; // Safe to proceed with the yield turn.
                } else {
                    should_brake = true; // Not safe or not a permitted turn, remain stopped.
                }
            } else { // Approaching but cannot stop in time or already past line - this is a red light running scenario
                 should_brake = false; // Run the red (already determined by lack of braking possibility for RED)
            }
            break;
        
        case TRAFFIC_LIGHT_YIELD: // General yield sign
            if (car_should_yield_at_intersection(self, sim, situation, turn_indicator)) {
                should_brake = true;
            } else {
                should_brake = false;
            }
            break;

        case TRAFFIC_LIGHT_GREEN:
        case TRAFFIC_LIGHT_NONE:
        default:
            should_brake = false;
            break;
        }
    }

    // If we are at an intersection and making a left turn, then we should stop immediately (not necessarily at the stop line). Same case for right turn from right lane.
    if (should_brake && car_should_yield_at_intersection(self, sim, situation, turn_indicator)) {
        // printf("Breaking on Left Yield!\n");
        position_target = car_position;
        position_target_overshoot_buffer = meters(1);
        speed_at_target = 0;
        MetersPerSecondSquared accel_to_stop_at_lane_end = car_compute_acceleration_chase_target(self, position_target, speed_at_target, position_target_overshoot_buffer, speed_cruise);
        accel = fmin(accel, accel_to_stop_at_lane_end); // take the minimum of the two accelerations
    }
    else if (should_brake) {
        // printf("10. Potentially braking\n");
        position_target = car_position + distance_to_stop_line;
        position_target_overshoot_buffer = meters(1);
        speed_at_target = 0;
        MetersPerSecondSquared accel_to_stop_at_lane_end = car_compute_acceleration_chase_target(self, position_target, speed_at_target, position_target_overshoot_buffer, speed_cruise);
        accel = fmin(accel, accel_to_stop_at_lane_end); // take the minimum of the two accelerations
    }

    // Set the decision variables
    // printf("11. Setting acceleration to %f\n\n", accel);
    accel = 0.95 * accel + 0.05 * car_get_acceleration(self); // smooth the acceleration
    car_set_acceleration(self, accel);
    car_set_indicator_turn(self, turn_indicator);
    car_set_indicator_lane(self, lane_change_indicator);
}
