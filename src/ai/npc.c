#include "ai.h"
#include "map.h"
#include "math.h"
#include "utils.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>


void npc_car_make_decisions(Car* self, Simulation* sim) {
    Seconds dt = sim_get_dt(sim);
    SituationalAwareness situation = situational_awareness_build(self, sim);
    const Lane* lane = situation.lane;
    // printf("\nOn lane: %p\n", lane);
    Meters car_speed = car_get_speed(self);
    Meters car_length = car_get_length(self);
    Meters car_half_length = car_length / 2;
    // Meters lane_length = lane_get_length(lane);
    // Meters lane_progress = situation.lane_progress;
    Meters lane_progress_m = situation.lane_progress_m;
    Meters car_position = lane_progress_m;          // alias for lane_progress_m
    Meters distance_to_stop_line = situation.distance_to_end_of_lane - car_half_length - meters(2); // if we had to stop at the end of the lane, how far would we be from the stop line?
    MetersPerSecond speed_cruise = lane_get_speed_limit(lane) + self->preferences.average_speed_offset;  // incorporate preference into speed limit

    // the following parameters will determine the outputs
    Meters position_target;
    Meters position_target_overshoot_buffer;    // car will try its best to not overshoot target + this buffer, by applying max capable braking.
    MetersPerSecond speed_at_target;
    CarIndictor turn_indicator;
    CarIndictor lane_change_indicator;

    if (situation.is_approaching_intersection) {
        // Whe approaching an intersection, turn and lane change decision will depend on distance to intersection. If we are far off, we can randomly change lanes or change turn intent. Else, we should not change lanes or change turn intent.
        if (situation.is_close_to_end_of_lane || situation.braking_distance.preferred_smooth >= 1.1 * distance_to_stop_line) {
            // Should not change lanes or change turn intent.
            // printf("0. Not changing lanes or turn intent\n");
            turn_indicator = car_get_indicator_turn(self);  // whatever it is
            if (!situation.is_turn_possible[turn_indicator]) {
                turn_indicator = turn_sample_possible(&situation);
            }
            lane_change_indicator = INDICATOR_NONE;
        } else {
            // We are not close to the end of the lane.
            turn_indicator = (rand_0_to_1() < dt / 4) ? turn_sample_possible(&situation) : car_get_indicator_turn(self);
            if (!situation.is_turn_possible[turn_indicator]) {
                turn_indicator = turn_sample_possible(&situation);
            }
            // printf("Turn indicator: %d\n", turn_indicator);
            lane_change_indicator = (rand_0_to_1() < dt / 5) ? lane_change_sample_possible(&situation) : INDICATOR_NONE;
            // printf("Lane change indicator: %d\n", lane_change_indicator);
            // printf("0. Intersection upcoming: sampled turn indicator and lane change indicator: %d, %d\n", turn_indicator, lane_change_indicator);
        }
    } else if (situation.is_approaching_dead_end) {
        // printf("NPC P1: C2\n");
        // We are not approaching an intersection, but rather a dead end. We should try to move to left most lane if this road merges into something. If not, maybe there is an exit available and or there is an upcoming exit? Then we should try to move to right most lane. If none of this is possible, the we should just stop at the right most lane.
        turn_indicator = INDICATOR_NONE;
        if (situation.is_on_entry_ramp) {
            lane_change_indicator = INDICATOR_LEFT;
        } else if (situation.is_exit_available || situation.is_exit_available_eventually) {
            lane_change_indicator = INDICATOR_RIGHT;
        } else {
            lane_change_indicator = INDICATOR_RIGHT;
        }
        // printf("0. Dead end: lane change indicator: %d\n", lane_change_indicator);
    } else {
        // We are not approaching an intersection or a dead end. We can randomly change lanes (and turn intent is not relevant).
        turn_indicator = INDICATOR_NONE;
        lane_change_indicator = (rand_0_to_1() < dt) ? lane_change_sample_possible(&situation) : INDICATOR_NONE;
        if (lane_change_indicator != INDICATOR_NONE) {
            // printf("0. Not an intersection or dead end: lane change indicator: %d\n", lane_change_indicator);
        }
        if (situation.is_exit_possible) {
            // printf("Exit is possible. Indicating right.\n");
            lane_change_indicator = INDICATOR_RIGHT;
        }
    }

    // cancel lane change if it is dangerous
    if (car_is_lane_change_dangerous(self, sim, &situation, lane_change_indicator)) {
        lane_change_indicator = INDICATOR_NONE;
        // printf("Lane change is dangerous. Canceling lane change.\n");
    }

    // Next up is to determine the acceleration. If there is a vehicle ahead, then we just need to follow it with its speed. If nothing ahead, we just cruise. But we should override to a slower acceleration if we are close to an intersection (and approaching yellow / red), or approaching a dead end. It is possible that the lead vehicle is going faster than it should be!
    MetersPerSecondSquared accel;
    
    if (situation.is_vehicle_ahead) {
        // printf("1. Lead vehicle detected\n");
        // Decide how much distance to maintain. Let's make it "3-second rule" for now.
        // TODO: For worse conditions, it should be higher.
        // TODO: Incorporate preferences.
        Meters target_distance_from_next_car = fmax(car_speed, 0) * 3 + from_feet(3);
        Meters car_lengths_offset = car_half_length + car_get_length(situation.lead_vehicle) / 2;
        target_distance_from_next_car += car_lengths_offset;
        position_target = car_position + situation.distance_to_lead_vehicle - target_distance_from_next_car;
        position_target_overshoot_buffer = target_distance_from_next_car - car_lengths_offset - from_feet(1.0); // let's call being within 1.0 feet of the next car as a crash.
        speed_at_target = car_get_speed(situation.lead_vehicle);
        accel = car_compute_acceleration_chase_target(self, position_target, speed_at_target, position_target_overshoot_buffer, speed_cruise);
        // accel += situation.lead_vehicle->acceleration; // add the lead vehicle's acceleration to our acceleration to account for the fact that we are following it and it may be accelerating or braking.

        // If are already within 2.0 feet of the lead vehicle, let's try to change lanes to avoid rear-ending it.
        if (situation.distance_to_lead_vehicle - (car_half_length + car_get_length(situation.lead_vehicle) / 2) < from_feet(2.0)) { // TODO: make it depend on more variables
            // printf("Emergency lane change maneuver to avoid rear-ending the next car.\n");
            if (lane_change_indicator == INDICATOR_NONE) {
                lane_change_indicator = lane_change_sample_possible(&situation);
                if (lane_change_indicator == INDICATOR_NONE) {
                    // printf("Oh no! No lane change possible. We might rear-end the lead vehicle!\n");
                }
                // TODO: check if we will crash into the side car if we change lanes. If so, then we should not change lanes. Or maybe a side crash is better than a rear-end crash?
            }
        }
    } else {
        if (!situation.is_approaching_dead_end && (situation.is_close_to_end_of_lane || situation.braking_distance.preferred_smooth >= distance_to_stop_line)) {
            // start changing speed to match speed limit of the next lane
            accel = car_compute_acceleration_cruise(self, lane_get_speed_limit(situation.lane_next_after_turn[turn_indicator]) + self->preferences.average_speed_offset);
        } else {
            accel = car_compute_acceleration_cruise(self, speed_cruise);
        }
    }

    bool should_brake = false;
    
    if (situation.is_approaching_dead_end) {
        // printf("3. But, approaching dead end\n");
        // printf("4. Distance to stop line: %f\n", distance_to_stop_line + meters(1));
        should_brake = true;
    }

    if (situation.is_approaching_intersection) {
        TrafficLight light = situation.light_for_turn[turn_indicator];
        bool is_emergency_braking_possible = (situation.braking_distance.capable <= distance_to_stop_line + meters(1) + 1e-6);
        bool is_comfy_braking_possible = (situation.braking_distance.preferred_smooth <= distance_to_stop_line + 1e-6);

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
            if (car_should_yield_at_intersection(self, sim, &situation, turn_indicator)) {
                should_brake = true;
            } else {
                should_brake = false; 
            }
            break;

        case TRAFFIC_LIGHT_RED_YIELD:
            // Must first obey the RED aspect (stop or be stopping).
            if (fabs(car_get_speed(self)) > meters(0.5) && // If moving with some speed
                (is_comfy_braking_possible || is_emergency_braking_possible) && // and can stop
                (situation.distance_to_end_of_lane - car_half_length > meters(0.5))) { // and not yet at the line
                should_brake = true; // Enforce stop for the RED part.
            } else if (fabs(car_get_speed(self)) <= meters(0.5)) { // If stopped or barely moving at the line
                // Check if it's a permitted turn (usually right) and if it's clear
                if (situation.is_turn_possible[turn_indicator] &&
                    (turn_indicator == INDICATOR_RIGHT /* Add other conditions if exits, e.g. specific ramp merges */) &&
                    !car_should_yield_at_intersection(self, sim, &situation, turn_indicator)) {
                    should_brake = false; // Safe to proceed with the yield turn.
                } else {
                    should_brake = true; // Not safe or not a permitted turn, remain stopped.
                }
            } else { // Approaching but cannot stop in time or already past line - this is a red light running scenario
                 should_brake = false; // Run the red (already determined by lack of braking possibility for RED)
            }
            break;
        
        case TRAFFIC_LIGHT_YIELD: // General yield sign
            if (car_should_yield_at_intersection(self, sim, &situation, turn_indicator)) {
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
    if (should_brake && car_should_yield_at_intersection(self, sim, &situation, turn_indicator)) {
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


SituationalAwareness situational_awareness_build(const Car* car, Simulation* sim) {
    LOG_TRACE("\nBuilding situational awareness for car %d:", car->id);
    Map* map = sim_get_map(sim);
    SituationalAwareness situation;
    const Lane* lane = car_get_lane(car, map);
    const Road* road = lane_get_road(lane, map);    // Road we are on. May be NULL if we are already on an intersection.
    const Intersection* intersection = lane_get_intersection(lane, map); // Intersection we are on. May be NULL if we are not on an intersection.
    double lane_progress = car_get_lane_progress(car);
    Meters lane_progress_m = car_get_lane_progress_meters(car);
    MetersPerSecond car_speed = car_get_speed(car);
    Meters lane_length = lane_get_length(lane);
    Direction lane_dir = lane->direction;
    Meters distance_to_end_of_lane = lane_length - lane_progress_m;
    LOG_TRACE("Car %d is %.2f meters (progress = %.2f), travelling at %.2f mph along lane %d (dir = %d) on %s", car->id, lane_progress_m, lane_progress, to_mph(car_speed), lane->id, lane_dir, road ? road_get_name(road) : intersection_get_name(intersection));
    const Intersection* intersection_upcoming = road ? road_leads_to_intersection(road, map) : NULL;   // may be null if we are already on an intersection or this road does not lead to an intersection.

    situation.is_approaching_dead_end = lane_get_num_connections(lane) == 0;
    if (lane->is_at_intersection) {
        situation.intersection = intersection;
        situation.is_on_intersection = true;
        situation.is_stopped_at_intersection = false;
    } else {
        situation.is_on_intersection = false;
        situation.intersection = intersection_upcoming; // may be NULL
        situation.is_approaching_intersection = intersection_upcoming != NULL;
        situation.is_stopped_at_intersection = intersection_upcoming && (fabs(to_mph(car_get_speed(car))) < 1) && (distance_to_end_of_lane < 50.0);
        if (intersection_upcoming) LOG_TRACE("Car %d is approaching intersection %s in %.2f meters. Stopped for it = %d", car->id, intersection_get_name(intersection_upcoming), distance_to_end_of_lane, situation.is_stopped_at_intersection);
    }
    situation.road = road;
    situation.lane = lane;
    Lane* adjacent_left = lane_get_adjacent_left(lane, map);
    Lane* adjacent_right = lane_get_adjacent_right(lane, map);
    situation.lane_left = (adjacent_left && adjacent_left->direction == lane->direction) ? adjacent_left : NULL;
    situation.lane_right = adjacent_right;
    situation.distance_to_intersection = intersection_upcoming ? distance_to_end_of_lane : 1e6;
    situation.distance_to_end_of_lane = distance_to_end_of_lane;
    situation.is_close_to_end_of_lane = distance_to_end_of_lane < 50.0;
    situation.is_on_leftmost_lane = road ? lane == road_get_leftmost_lane(road, map) : false;
    situation.is_on_rightmost_lane = road ? lane == road_get_rightmost_lane(road, map) : false;
    situation.lane_progress = lane_progress;
    situation.lane_progress_m = lane_progress_m;
    LOG_TRACE("Car %d (current lane %d (is leftmost=%d, rightmost=%d)) adjacent left lane: %d, right lane: %d." , car->id, lane->id, situation.is_on_leftmost_lane, situation.is_on_rightmost_lane, situation.lane_left ? situation.lane_left->id : -1, situation.lane_right ? situation.lane_right->id : -1);

    situation.is_turn_possible[INDICATOR_LEFT] = lane->connections[INDICATOR_LEFT] != ID_NULL;
    situation.is_turn_possible[INDICATOR_NONE] = lane->connections[INDICATOR_NONE] != ID_NULL;
    situation.is_turn_possible[INDICATOR_RIGHT] = lane->connections[INDICATOR_RIGHT] != ID_NULL;
    situation.lane_next_after_turn[INDICATOR_LEFT] = lane_get_connection_left(lane, map);
    situation.lane_next_after_turn[INDICATOR_NONE] = lane_get_connection_straight(lane, map);
    situation.lane_next_after_turn[INDICATOR_RIGHT] = lane_get_connection_right(lane, map);
    LOG_TRACE("Car %d: Turn indicator outcomes: left=%d, straight=%d, right=%d", car->id, lane->connections[INDICATOR_LEFT], lane->connections[INDICATOR_NONE], lane->connections[INDICATOR_RIGHT]);
    if (intersection_upcoming) {
        bool free_right = true;
        bool is_NS = (lane_dir == DIRECTION_NORTH || lane_dir == DIRECTION_SOUTH);
        switch (intersection_upcoming->state) {
            case FOUR_WAY_STOP:
                situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_STOP;
                situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_STOP;
                situation.light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_STOP;
                break;
            case NS_GREEN_EW_RED:
                if (is_NS) {
                    situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_GREEN_YIELD;
                    situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_GREEN;
                    situation.light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_GREEN;
                } else {
                    situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation.light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                }
                break;
            case NS_YELLOW_EW_RED:
                if (is_NS) {
                    situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_YELLOW_YIELD;
                    situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_YELLOW;
                    situation.light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_YELLOW;
                } else {
                    situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation.light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                }
                break;
            case ALL_RED_BEFORE_EW_GREEN:
                situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                situation.light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                break;
            case NS_RED_EW_GREEN:
                if (is_NS) {
                    situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation.light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                } else {
                    situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_GREEN_YIELD;
                    situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_GREEN;
                    situation.light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_GREEN;
                }
                break;
            case EW_YELLOW_NS_RED:
                if (is_NS) {
                    situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation.light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                } else {
                    situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_YELLOW_YIELD;
                    situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_YELLOW;
                    situation.light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_YELLOW;
                }
                break;
            case ALL_RED_BEFORE_NS_GREEN:
                situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                situation.light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                break;
            default:
                situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                situation.light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_RED;
                break;
        }
        LOG_TRACE("Car %d: Traffic lights at upcoming intersection %s (free right = %d): left=%d, straight=%d, right=%d", car->id, intersection_get_name(intersection_upcoming), free_right, situation.light_for_turn[INDICATOR_LEFT], situation.light_for_turn[INDICATOR_NONE], situation.light_for_turn[INDICATOR_RIGHT]);
    } else {
        situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_NONE;
        situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_NONE;
        situation.light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_NONE;
        LOG_TRACE("Car %d: No upcoming traffic lights", car->id);
    }
    situation.is_lane_change_left_possible = situation.lane_left != NULL;
    situation.is_lane_change_right_possible = situation.lane_right != NULL;
    situation.is_merge_possible = lane_is_merge_available(lane);
    situation.is_exit_possible = lane_is_exit_lane_available(lane, lane_progress);
    situation.is_on_entry_ramp = road ? road_is_merge_available(road, map) : false;
    situation.is_exit_available = road ? road_is_exit_road_available(road, map, lane_progress) : false;
    situation.is_exit_available_eventually = road ? road_is_exit_road_eventually_available(road, map, lane_progress) : false;
    LOG_TRACE("Car %d: Lane change and merge/exit possibilities: left=%d, right=%d, merge=%d, exit=%d, entry ramp=%d, exit available now=%d, exit available eventually=%d", car->id, situation.is_lane_change_left_possible, situation.is_lane_change_right_possible, situation.is_merge_possible, situation.is_exit_possible, situation.is_on_entry_ramp, situation.is_exit_available, situation.is_exit_available_eventually);

    situation.merges_into_lane = lane_get_merge_into(lane, map);
    situation.exit_lane = (situation.is_exit_available || situation.is_exit_available_eventually) ? lane_get_exit_lane(lane, map) : NULL;
    situation.lane_target_for_indicator[INDICATOR_LEFT] = situation.merges_into_lane ? situation.merges_into_lane : situation.lane_left;
    situation.lane_target_for_indicator[INDICATOR_NONE] = situation.lane;
    situation.lane_target_for_indicator[INDICATOR_RIGHT] = situation.exit_lane ? situation.exit_lane : situation.lane_right;
    LaneId left_outcome_id = situation.lane_target_for_indicator[INDICATOR_LEFT] ? situation.lane_target_for_indicator[INDICATOR_LEFT]->id : ID_NULL;
    LaneId straight_outcome_id = situation.lane_target_for_indicator[INDICATOR_NONE] ? situation.lane_target_for_indicator[INDICATOR_NONE]->id : ID_NULL;
    LaneId right_outcome_id = situation.lane_target_for_indicator[INDICATOR_RIGHT] ? situation.lane_target_for_indicator[INDICATOR_RIGHT]->id : ID_NULL;
    LOG_TRACE("Car %d: Lane indicator outcomes: left=%d (is a merge = %d), none=%d, right=%d (is an exit = %d)", car->id, left_outcome_id, situation.merges_into_lane != NULL, straight_outcome_id, right_outcome_id, situation.exit_lane != NULL);

    const Car* closest_forward = NULL;
    Meters closest_forward_distance = 1e6;
    const Car* closest_behind = NULL;
    Meters closest_behind_distance = 1e6;
    for (int i = 0; i < lane_get_num_cars(lane); i++) { // TODO: make the cars sorted by progress in the cars array on lane.
        const Car* car_i = lane_get_car(lane, sim, i);
        if (car_i == car) continue;
        Meters distance = car_get_lane_progress_meters(car_i) - lane_progress_m;
        if (distance > 0 && distance < closest_forward_distance) {
            closest_forward_distance = distance;
            closest_forward = car_i;
        } else if (distance < 0 && -distance < closest_behind_distance) {
            closest_behind_distance = -distance;
            closest_behind = car_i;
        }
    }
    // TODO: if no vehicle found ahead on this lane, pick the closest vehicle on the next lane.
    // TODO: if no vehicle found behind on this lane, pick the closest vehicle on the previous lane.
    situation.is_vehicle_ahead = closest_forward != NULL;
    situation.is_vehicle_behind = closest_behind != NULL;
    situation.lead_vehicle = closest_forward;
    situation.following_vehicle = closest_behind;
    situation.distance_to_lead_vehicle = closest_forward_distance;
    situation.distance_to_following_vehicle = closest_behind_distance;
    // TODO: situational awareness of this vehicle regarding these things can be propagated to the situational awareness of the lead vehicle and the following vehicle.
    LOG_TRACE("Car %d: Lead vehicle: %d (distance = %.2f m), following vehicle: %d (distance = %.2f m)", car->id, situation.is_vehicle_ahead ? situation.lead_vehicle->id : -1, situation.distance_to_lead_vehicle, situation.is_vehicle_behind ? situation.following_vehicle->id : -1,  situation.distance_to_following_vehicle);

    situation.braking_distance = car_compute_braking_distance(car);
    LOG_TRACE("Car %d: Braking distance: capable=%.2f m, preferred=%.2f m, preferred smooth=%.2f m", car->id, situation.braking_distance.capable, situation.braking_distance.preferred, situation.braking_distance.preferred_smooth);

    LOG_TRACE("Situational awareness built for car %d\n", car->id);
    return situation;
}