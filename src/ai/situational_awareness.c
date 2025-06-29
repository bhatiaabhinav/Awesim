#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "logging.h"
#include "ai.h"

void situational_awareness_build(Simulation* sim, CarId car_id) {
    if (!sim) {
        LOG_ERROR("Invalid parameters: sim is NULL. Cannot build situational awareness. Simulation: %p, Car ID: %d", sim, car_id);
        return;
    }
    Car* car = sim_get_car(sim, car_id);
    SituationalAwareness* situation = sim_get_situational_awareness(sim, car_id);
    if (situation->is_valid) {
        LOG_TRACE("Situational awareness for car %d is already valid. No need to rebuild.", car_id);
        return;
    }

    const Lane* lane_prev_t = situation->lane; // Previous lane, used to detect lane changes.

    LOG_TRACE("\nBuilding situational awareness for car %d:", car->id);
    Map* map = sim_get_map(sim);
    Meters car_half_length = car_get_length(car) / 2.0; // Half the length of the car
    const Lane* lane = car_get_lane(car, map);
    const Road* road = lane_get_road(lane, map);    // Road we are on. May be NULL if we are already on an intersection.
    const Intersection* intersection = lane_get_intersection(lane, map); // Intersection we are on. May be NULL if we are not on an intersection.
    double lane_progress = car_get_lane_progress(car);
    Meters lane_progress_m = car_get_lane_progress_meters(car);
    MetersPerSecond car_speed = car_get_speed(car);
    Meters lane_length = lane_get_length(lane);
    Direction lane_dir = lane->direction;
    Meters distance_to_end_of_lane = lane_length - lane_progress_m;
    Meters distance_to_end_of_lane_from_leading_edge = distance_to_end_of_lane - car_half_length; // Distance to the end of the lane from the leading edge of the car.
    LOG_TRACE("Car %d is %.2f meters (progress = %.2f), travelling at %.2f mph along lane %d (dir = %d) on %s", car->id, lane_progress_m, lane_progress, to_mph(car_speed), lane->id, lane_dir, road ? road_get_name(road) : intersection_get_name(intersection));
    const Intersection* intersection_upcoming = road ? road_leads_to_intersection(road, map) : NULL;   // may be null if we are already on an intersection or this road does not lead to an intersection.

    situation->is_approaching_dead_end = lane_get_num_connections(lane) == 0;
    if (lane->is_at_intersection) {
        situation->intersection = intersection;
        situation->is_on_intersection = true;
        situation->is_stopped_at_intersection = false;
    } else {
        situation->is_on_intersection = false;
        situation->intersection = intersection_upcoming; // may be NULL
        situation->is_an_intersection_upcoming = intersection_upcoming != NULL;
        situation->is_stopped_at_intersection = intersection_upcoming && (fabs(to_mph(car_get_speed(car))) < 0.1) && (distance_to_end_of_lane_from_leading_edge < 50.0);
        if (intersection_upcoming) LOG_TRACE("Car %d is approaching intersection %s in %.2f meters (from car's leading edge). Stopped for it = %d", car->id, intersection_get_name(intersection_upcoming), distance_to_end_of_lane_from_leading_edge, situation->is_stopped_at_intersection);
    }
    situation->road = road;
    situation->lane = lane;
    Lane* adjacent_left = lane_get_adjacent_left(lane, map);
    Lane* adjacent_right = lane_get_adjacent_right(lane, map);
    situation->lane_left = (adjacent_left && adjacent_left->direction == lane->direction) ? adjacent_left : NULL;
    situation->lane_right = adjacent_right;
    situation->distance_to_intersection = intersection_upcoming ? distance_to_end_of_lane : 1e6;
    situation->distance_to_end_of_lane = distance_to_end_of_lane;
    situation->distance_to_end_of_lane_from_leading_edge = distance_to_end_of_lane_from_leading_edge;
    situation->is_approaching_end_of_lane = distance_to_end_of_lane_from_leading_edge < 50.0;
    situation->is_at_end_of_lane = distance_to_end_of_lane_from_leading_edge < 4.0;
    situation->is_on_leftmost_lane = road ? lane == road_get_leftmost_lane(road, map) : false;
    situation->is_on_rightmost_lane = road ? lane == road_get_rightmost_lane(road, map) : false;
    situation->lane_progress = lane_progress;
    situation->lane_progress_m = lane_progress_m;
    LOG_TRACE("Car %d (current lane %d (is leftmost=%d, rightmost=%d)) adjacent left lane: %d, right lane: %d." , car->id, lane->id, situation->is_on_leftmost_lane, situation->is_on_rightmost_lane, situation->lane_left ? situation->lane_left->id : -1, situation->lane_right ? situation->lane_right->id : -1);

    situation->is_turn_possible[INDICATOR_LEFT] = lane->connections[INDICATOR_LEFT] != ID_NULL;
    situation->is_turn_possible[INDICATOR_NONE] = lane->connections[INDICATOR_NONE] != ID_NULL;
    situation->is_turn_possible[INDICATOR_RIGHT] = lane->connections[INDICATOR_RIGHT] != ID_NULL;
    situation->lane_next_after_turn[INDICATOR_LEFT] = lane_get_connection_left(lane, map);
    situation->lane_next_after_turn[INDICATOR_NONE] = lane_get_connection_straight(lane, map);
    situation->lane_next_after_turn[INDICATOR_RIGHT] = lane_get_connection_right(lane, map);
    LOG_TRACE("Car %d: Turn indicator outcomes: left=%d, straight=%d, right=%d", car->id, lane->connections[INDICATOR_LEFT], lane->connections[INDICATOR_NONE], lane->connections[INDICATOR_RIGHT]);
    if (intersection_upcoming) {
        bool free_right = true;
        bool is_NS = (lane_dir == DIRECTION_NORTH || lane_dir == DIRECTION_SOUTH);
        switch (intersection_upcoming->state) {
            case FOUR_WAY_STOP:
                situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_STOP;
                situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_STOP;
                situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_STOP;
                break;
            case NS_GREEN_EW_RED:
                if (is_NS) {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_GREEN_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_GREEN;
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_GREEN;
                } else {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                }
                break;
            case NS_YELLOW_EW_RED:
                if (is_NS) {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_YELLOW_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_YELLOW;
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_YELLOW;
                } else {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                }
                break;
            case ALL_RED_BEFORE_EW_GREEN:
                situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                break;
            case NS_RED_EW_GREEN:
                if (is_NS) {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                } else {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_GREEN_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_GREEN;
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_GREEN;
                }
                break;
            case EW_YELLOW_NS_RED:
                if (is_NS) {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                } else {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_YELLOW_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_YELLOW;
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_YELLOW;
                }
                break;
            case ALL_RED_BEFORE_NS_GREEN:
                situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                break;
            default:
                situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_RED;
                break;
        }
        LOG_TRACE("Car %d: Traffic lights at upcoming intersection %s (free right = %d): left=%d, straight=%d, right=%d", car->id, intersection_get_name(intersection_upcoming), free_right, situation->light_for_turn[INDICATOR_LEFT], situation->light_for_turn[INDICATOR_NONE], situation->light_for_turn[INDICATOR_RIGHT]);
    } else {
        situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_NONE;
        situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_NONE;
        situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_NONE;
        LOG_TRACE("Car %d: No upcoming traffic lights", car->id);
    }
    situation->is_lane_change_left_possible = situation->lane_left != NULL;
    situation->is_lane_change_right_possible = situation->lane_right != NULL;
    situation->is_merge_possible = lane_is_merge_available(lane);
    situation->is_exit_possible = lane_is_exit_lane_available(lane, lane_progress);
    situation->is_on_entry_ramp = road ? road_is_merge_available(road, map) : false;
    situation->is_exit_available = road ? road_is_exit_road_available(road, map, lane_progress) : false;
    situation->is_exit_available_eventually = road ? road_is_exit_road_eventually_available(road, map, lane_progress) : false;
    LOG_TRACE("Car %d: Lane change and merge/exit possibilities: left=%d, right=%d, merge=%d, exit=%d, entry ramp=%d, exit available now=%d, exit available eventually=%d", car->id, situation->is_lane_change_left_possible, situation->is_lane_change_right_possible, situation->is_merge_possible, situation->is_exit_possible, situation->is_on_entry_ramp, situation->is_exit_available, situation->is_exit_available_eventually);

    situation->merges_into_lane = lane_get_merge_into(lane, map);
    situation->exit_lane = (situation->is_exit_available || situation->is_exit_available_eventually) ? lane_get_exit_lane(lane, map) : NULL;
    situation->lane_target_for_indicator[INDICATOR_LEFT] = situation->merges_into_lane ? situation->merges_into_lane : situation->lane_left;
    situation->lane_target_for_indicator[INDICATOR_NONE] = situation->lane;
    situation->lane_target_for_indicator[INDICATOR_RIGHT] = situation->exit_lane ? situation->exit_lane : situation->lane_right;
    LaneId left_outcome_id = situation->lane_target_for_indicator[INDICATOR_LEFT] ? situation->lane_target_for_indicator[INDICATOR_LEFT]->id : ID_NULL;
    LaneId straight_outcome_id = situation->lane_target_for_indicator[INDICATOR_NONE] ? situation->lane_target_for_indicator[INDICATOR_NONE]->id : ID_NULL;
    LaneId right_outcome_id = situation->lane_target_for_indicator[INDICATOR_RIGHT] ? situation->lane_target_for_indicator[INDICATOR_RIGHT]->id : ID_NULL;
    LOG_TRACE("Car %d: Lane indicator outcomes: left=%d (is a merge = %d), none=%d, right=%d (is an exit = %d)", car->id, left_outcome_id, situation->merges_into_lane != NULL, straight_outcome_id, right_outcome_id, situation->exit_lane != NULL);

    const Car* closest_forward = NULL;
    Meters closest_forward_distance = 1e6;
    const Car* closest_behind = NULL;
    Meters closest_behind_distance = 1e6;
    int my_rank = car_get_lane_rank(car);
    int closest_forward_rank = my_rank - 1; // the next car in the lane
    if (closest_forward_rank >= 0 && closest_forward_rank < lane_get_num_cars(lane)) {
        closest_forward = lane_get_car(lane, sim, closest_forward_rank);
        closest_forward_distance = car_get_lane_progress_meters(closest_forward) - lane_progress_m;
    }
    // If no vehicle found ahead on this lane, pick the closest vehicle on the next lane(s)
    if (closest_forward == NULL && !situation->is_approaching_dead_end) {
        for (int turn_intent = 0; turn_intent < 3; turn_intent++) {
            const Lane* next_lane = situation->lane_next_after_turn[turn_intent];
            if (next_lane) {
                int next_lane_rank = lane_get_num_cars(next_lane) - 1; // the last car in the next lane
                if (next_lane_rank >= 0) {
                    const Car* next_car = lane_get_car(next_lane, sim, next_lane_rank);
                    Meters next_car_progress_m = car_get_lane_progress_meters(next_car);
                    Meters distance_to_next_car = next_car_progress_m + distance_to_end_of_lane;
                    if (distance_to_next_car < closest_forward_distance) {
                        closest_forward = next_car;
                        closest_forward_distance = distance_to_next_car;
                    }
                }
            }
        }
    }
    int closest_behind_rank = my_rank + 1; // the previous car in the lane
    if (closest_behind_rank < lane_get_num_cars(lane)) {
        closest_behind = lane_get_car(lane, sim, closest_behind_rank);
        closest_behind_distance = lane_progress_m - car_get_lane_progress_meters(closest_behind);
    }
    // TODO: if no vehicle found behind on this lane, pick the closest vehicle on the previous lane
    situation->is_vehicle_ahead = closest_forward != NULL;
    situation->is_vehicle_behind = closest_behind != NULL;
    situation->lead_vehicle = closest_forward;
    situation->following_vehicle = closest_behind;
    situation->distance_to_lead_vehicle = closest_forward_distance;
    situation->distance_to_following_vehicle = closest_behind_distance;
    LOG_TRACE("Car %d: Lead vehicle: %d (distance = %.2f m), following vehicle: %d (distance = %.2f m)", car->id, situation->is_vehicle_ahead ? situation->lead_vehicle->id : -1, situation->distance_to_lead_vehicle, situation->is_vehicle_behind ? situation->following_vehicle->id : -1,  situation->distance_to_following_vehicle);

    situation->braking_distance = car_compute_braking_distance(car);
    LOG_TRACE("Car %d: Braking distance: capable=%.2f m, preferred=%.2f m, preferred smooth=%.2f m", car->id, situation->braking_distance.capable, situation->braking_distance.preferred, situation->braking_distance.preferred_smooth);

    if (lane_prev_t && lane == lane_prev_t) {
        // No lane change detected, just update the position and speed
        if (car_speed < situation->slowest_speed_yet_on_current_lane) {
            situation->slowest_speed_yet_on_current_lane = car_speed;
            situation->slowest_speed_position = lane_progress_m;
            LOG_TRACE("Car %d has achieved slowest speed yet on current lane = %.2f mph at progress %.2f", car->id, to_mph(situation->slowest_speed_yet_on_current_lane), situation->slowest_speed_position / lane_length);
        }
    } else {
        // Lane change detected, reset the slowest speed
        LOG_TRACE("Car %d has just changed lanes from %d to %d. So, resetting slowest speed yet on current lane and the corresponding position.", car->id, lane_prev_t ? lane_prev_t->id : -1, lane->id);
        situation->slowest_speed_yet_on_current_lane = car_speed;
        situation->slowest_speed_position = lane_progress_m;
    }

    if (situation->is_at_end_of_lane) {
        situation->slowest_speed_yet_at_end_of_lane = fmin(situation->slowest_speed_yet_at_end_of_lane, car_speed);
        if (!situation->full_stopped_yet_at_end_of_lane && to_mph(car_speed) < 0.1) {
            situation->full_stopped_yet_at_end_of_lane = true; // Consider it a full stop if speed is less than 0.1 mph
        }
        if (!situation->rolling_stopped_yet_at_end_of_lane && to_mph(car_speed) < 5.0) {
            situation->rolling_stopped_yet_at_end_of_lane = true; // Consider it a rolling stop if speed is less than 5 mph
        }
        LOG_TRACE("Car %d is at the end of lane %d (%.2f m from leading edge), slowest speed yet at end of lane = %.2f mph. Rolling stop achieved = %d, full stop achieved = %d.", car->id, lane->id, distance_to_end_of_lane_from_leading_edge, to_mph(situation->slowest_speed_yet_at_end_of_lane), situation->rolling_stopped_yet_at_end_of_lane, situation->full_stopped_yet_at_end_of_lane);
    } else {
        situation->slowest_speed_yet_at_end_of_lane = car_speed;
        situation->full_stopped_yet_at_end_of_lane = false;
        situation->rolling_stopped_yet_at_end_of_lane = false;
    }


    LOG_TRACE("Situational awareness built for car %d\n", car->id);
    situation->is_valid = true; // Mark situational awareness as valid
    situation->timestamp = sim_get_time(sim); // Set the timestamp of the situational awareness
}
