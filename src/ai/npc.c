#include "ai.h"
#include "road.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>


void npc_car_make_decisions(Car* self) {
    SituationalAwareness situation = situational_awareness_build(self);
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
            speed_at_target = 0;
        } else {
            // We are not close to the end of the lane.
            turn_indicator = turn_sample_possible(&situation);
            // printf("Turn indicator: %d\n", turn_indicator);
            lane_change_indicator = (rand_0_to_1() < 0.02) ? lane_change_sample_possible(&situation) : INDICATOR_NONE;
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
        lane_change_indicator = (rand_0_to_1() < 0.02) ? lane_change_sample_possible(&situation) : INDICATOR_NONE;
        if (lane_change_indicator != INDICATOR_NONE) {
            // printf("0. Not an intersection or dead end: lane change indicator: %d\n", lane_change_indicator);
        }
        if (situation.is_exit_possible) {
            // printf("Exit is possible. Indicating right.\n");
            lane_change_indicator = INDICATOR_RIGHT;
        }
    }

    // cancel lane change if it is dangerous
    if (car_is_lane_change_dangerous(self, &situation, lane_change_indicator)) {
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
        accel += situation.lead_vehicle->acceleration; // add the lead vehicle's acceleration to our acceleration to account for the fact that we are following it and it may be accelerating or braking.

        // If are already within 2.0 feet of the lead vehicle, let's try to change lanes to avoid rear-ending it.
        if (situation.distance_to_lead_vehicle - (car_half_length + car_get_length(situation.lead_vehicle) / 2) < from_feet(2.0)) {
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
            // start slowing down to the speed limit of the next lane
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
        // printf("3. But, approaching intersection\n");
        TrafficLight light = situation.light_for_turn[turn_indicator];
        bool is_emergency_braking_possible = (situation.braking_distance.capable <= distance_to_stop_line + meters(1) + 1e-6); // meters(1) for a buffer -- such that it is ok to overshoot by 1m when emergency braking.
        bool is_comfy_braking_possible = (situation.braking_distance.preferred_smooth <= distance_to_stop_line + 1e-6);

        switch (light) {
        case TRAFFIC_LIGHT_STOP:
            // printf("6. Stop sign\n");
            should_brake = true;
            if (!is_emergency_braking_possible) {
                // printf("7. Distance to stop line: %f and with buffer: %f\n", distance_to_stop_line, distance_to_stop_line + meters(1) + 1e-6);
                // printf("8. Braking distances: capable, preferred, smooth: %f, %f, %f\n", situation.braking_distance.capable, situation.braking_distance.preferred, situation.braking_distance.preferred_smooth);
                // printf("9. But, we are not able to stop in time with emergency braking, so we jump the stop sign.\n");
                should_brake = false;
            }
            break;
        case TRAFFIC_LIGHT_RED:
            // printf("6. Red light\n");
            // printf("7. Emergency braking possible: %d\n", is_emergency_braking_possible);
            should_brake = true;
            if (!is_emergency_braking_possible) {
                // printf("7. Distance to stop line: %f and with buffer: %f\n", distance_to_stop_line, distance_to_stop_line + meters(1) + 1e-6);
                // printf("8. Braking distances: capable, preferred, smooth: %f, %f, %f\n", situation.braking_distance.capable, situation.braking_distance.preferred, situation.braking_distance.preferred_smooth);
                should_brake = false;
                // printf("9. But, we are not able to stop in time with emergency braking, so we jump the red light.\n");
            }
            break;
        case TRAFFIC_LIGHT_YELLOW:
            // TODO: incorporate situation.intersection->countdown
            if (is_comfy_braking_possible) {
                // printf("7. We can stop comfortably. Let's do that\n");
                should_brake = true;
            } else {
                // printf("7. We can't stop comfortably\n");
                should_brake = !self->preferences.run_yellow_light;  // run the yellow if the user prefers it
                if (should_brake) {
                    // printf("8. But, we don't mind braking hard to stop at the yellow light\n");
                    if (!is_emergency_braking_possible) {
                        // printf("9. But, we are not able to stop in time with emergency braking, so we jump the yellow light.\n");
                        should_brake = false;
                    }
                }
            }
            break;
        default:
            // printf("6. Default intersection case: just do what we are doing\n");
            should_brake = false;  // For green, none, and all types of yield // TODO: handle yeild cases differently.
            break;
        }
    }

    if (should_brake) {
        // printf("10. Potentially braking\n");
        position_target = car_position + distance_to_stop_line;
        position_target_overshoot_buffer = meters(1);
        speed_at_target = 0;
        MetersPerSecondSquared _accel = car_compute_acceleration_chase_target(self, position_target, speed_at_target, position_target_overshoot_buffer, speed_cruise);
        accel = fmin(accel, _accel); // take the minimum of the two accelerations
    }

    // Set the decision variables
    // printf("11. Setting acceleration to %f\n\n", accel);
    car_set_acceleration(self, accel);
    car_set_indicator_turn(self, turn_indicator);
    car_set_indicator_lane(self, lane_change_indicator);
}


SituationalAwareness situational_awareness_build(const Car* car) {
    SituationalAwareness situation;
    const Lane* lane = car->lane;
    const Road* road = lane->road;
    const Intersection* intersection = road_leads_to_intersection(road);    // may be null
    double lane_progress = car_get_lane_progress(car);
    Meters lane_progress_m = car_get_lane_progress_meters(car);
    Meters lane_length = lane_get_length(lane);
    Direction lane_dir = lane->direction;
    Meters distance_to_end_of_lane = lane_length - lane_progress_m;

    situation.is_approaching_dead_end = lane_get_num_connections(lane) == 0;
    situation.intersection = intersection;
    if (road->type == INTERSECTION) {
        situation.intersection = (const Intersection*) road;
        situation.is_on_intersection = true;
        situation.is_stopped_at_intersection = false;
    } else {
        situation.is_on_intersection = false;
        situation.is_approaching_intersection = intersection != NULL;
        situation.is_stopped_at_intersection = intersection && (fabs(car_get_speed(car)) < 0.001) && (distance_to_end_of_lane < 10.0);
    }
    situation.road = road;
    situation.lane = lane;
    situation.lane_left = (lane->adjacent_left && lane->adjacent_left->direction == lane->direction) ? lane->adjacent_left : lane->adjacent_right;
    situation.lane_right = lane->adjacent_right;
    situation.distance_to_intersection = intersection ? distance_to_end_of_lane : 1e6;
    situation.distance_to_end_of_lane = distance_to_end_of_lane;
    situation.is_close_to_end_of_lane = distance_to_end_of_lane < 10.0;
    situation.is_on_leftmost_lane = lane == road_get_leftmost_lane(road);
    situation.is_on_rightmost_lane = lane == road_get_rightmost_lane(road);
    situation.lane_progress = lane_progress;
    situation.lane_progress_m = lane_progress_m;

    situation.is_turn_possible[INDICATOR_LEFT] = lane->for_left_connects_to != NULL;
    situation.is_turn_possible[INDICATOR_NONE] = lane->for_straight_connects_to != NULL;
    situation.is_turn_possible[INDICATOR_RIGHT] = lane->for_right_connects_to != NULL;
    situation.lane_next_after_turn[INDICATOR_LEFT] = lane->for_left_connects_to;
    situation.lane_next_after_turn[INDICATOR_NONE] = lane->for_straight_connects_to;
    situation.lane_next_after_turn[INDICATOR_RIGHT] = lane->for_right_connects_to;
    if (intersection) {
        bool free_right = true;
        bool is_NS = (lane_dir == DIRECTION_NORTH || lane_dir == DIRECTION_SOUTH);
        switch (intersection->state) {
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
    } else {
        situation.light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_NONE;
        situation.light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_NONE;
        situation.light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_NONE;
    }
    situation.is_lane_change_left_possible = situation.lane_left != NULL;
    situation.is_lane_change_right_possible = situation.lane_right != NULL;
    situation.is_merge_possible = lane_is_merge_available(lane);
    situation.is_exit_possible = lane_is_exit_lane_available(lane, lane_progress);
    situation.is_on_entry_ramp = road_is_merge_available(road);
    situation.is_exit_available = road_is_exit_road_available(road, lane_progress);
    situation.is_exit_available_eventually = road_is_exit_road_eventually_available(road, lane_progress);
    situation.merges_into_lane = lane->merges_into;
    situation.exit_lane = (situation.is_exit_available || situation.is_exit_available_eventually) ? lane->exit_lane : NULL;
    situation.lane_target_for_indicator[INDICATOR_LEFT] = situation.merges_into_lane ? situation.merges_into_lane : situation.lane_left;
    situation.lane_target_for_indicator[INDICATOR_NONE] = situation.lane;
    situation.lane_target_for_indicator[INDICATOR_RIGHT] = situation.exit_lane ? situation.exit_lane : situation.lane_right;

    const Car* closest_forward = NULL;
    Meters closest_forward_distance = 1e6;
    const Car* closest_behind = NULL;
    Meters closest_behind_distance = 1e6;
    for (int i = 0; i < lane_get_num_cars(lane); i++) { // TODO: make the cars sorted by progress in the cars array on lane.
        const Car* car_i = lane_get_car(lane, i);
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

    situation.braking_distance = car_compute_braking_distance(car);

    // printf("Situational awareness built\n");
    return situation;
}