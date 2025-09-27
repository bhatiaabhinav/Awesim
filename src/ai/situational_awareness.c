#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "logging.h"
#include "ai.h"

static void _find_vehicle_behind_and_at_and_ahead_of_point(Simulation* sim, const Lane* lane, Meters point, Meters at_buffer, Car** car_behind_out, Meters* car_behind_distance_out, Car** car_at_out, Meters* car_at_distance_out, Car** car_ahead_out, Meters* car_ahead_distance_out, bool consider_prev_lanes, bool consider_next_lanes) {
    *car_behind_out = NULL;
    *car_behind_distance_out = INFINITY;
    *car_at_out = NULL;
    *car_at_distance_out = INFINITY;
    *car_ahead_out = NULL;
    *car_ahead_distance_out = INFINITY;

    if (!lane) return;

    Meters at_buffer_half = at_buffer / 2.0; // Half the buffer distance

    Map* map = sim_get_map(sim);

    for (int i = 0; i < lane->num_cars; i++) {  // traversing from leading vehicle to last vehicle on the lane
        Car* car = sim_get_car(sim, lane->cars_ids[i]);
        if (!car) continue; // No vehicle behind
        Meters car_position = car_get_lane_progress_meters(car);
        Meters car_half_length = car_get_length(car) / 2.0; // Half the length of the car
        

        // Logic:
        // if car's rear is ahead of point + at_buffer_half, then it is ahead of the point.
        // if not, if the center-to-center distance is less than sum of half lengths, then it in collision with the at_buffer.
        // if not, if the car's front is behind point - at_buffer_half, then it is behind the point. The moment we find such a car, we can stop searching since we are traversing from leading vehicle to last vehicle on the lane.

        if (point + at_buffer_half < car_position - car_half_length) {
            *car_ahead_out = car;   // this will keep getting overwritten until we find the first car ahead of the point.
            *car_ahead_distance_out = car_position - point - (at_buffer_half + car_half_length); // bumper-to-bumper distance
        } else if (fabs(car_position - point) <= car_half_length + at_buffer_half) {
            if (*car_at_out) { continue; } // We already have a car at, so we can skip this one.
            *car_at_out = car;
            *car_at_distance_out = car_position - point; // center-to-center distance, also marks whether it is ahead (positive) or behind (negative)
        } else if (car_position + car_half_length < point - at_buffer_half) {
            *car_behind_out = car;
            *car_behind_distance_out = point - car_position - (at_buffer_half + car_half_length);
            break; // since we are traversing from leading vehicle to last vehicle on the lane, we can stop searching here.
        }
    }

    // assert the bumper-to-bumper distances are non-negative, else throw an error and exit since this is a bug.
    if (*car_ahead_distance_out < 0 || *car_behind_distance_out < 0) {
        LOG_ERROR("For lane %d at position %.2f: Negative distance found: car_ahead_distance = %.2f, car_behind_distance = %.2f", lane->id, point, *car_ahead_distance_out, *car_behind_distance_out);
        exit(EXIT_FAILURE);
    }

    // if we did not find a car ahead, then we should pick the trailing car in the next lane in the same direction, if it exists.
    LOG_TRACE("here 1");
    if (!*car_ahead_out && consider_next_lanes) {
        LOG_TRACE("No car ahead found, checking next lane");
        Lane* next_lane = lane_get_connection_straight(lane, map);
        if (next_lane && lane_get_num_cars(next_lane) > 0) {
            Car* car = lane_get_car(next_lane, sim, lane_get_num_cars(next_lane) - 1); // last car in the next lane
            Meters car_half_length = car_get_length(car) / 2.0; // Half the length of the car
            Meters point_to_end_of_lane = lane_get_length(lane) - point; // distance from the point to the end of the lane
            Meters start_of_next_lane_to_car_ahead = car_get_lane_progress_meters(car);
            Meters point_to_center_distance = point_to_end_of_lane + start_of_next_lane_to_car_ahead;

            // Edge case: what if this car is in collision with the at_buffer?
            if (point_to_center_distance <= car_half_length + at_buffer_half) {
                // If we already don't have a car at, then we can set it to this car.
                if (!*car_at_out) {
                    *car_at_out = car;
                    *car_at_distance_out = point_to_center_distance; // center-to-center distance, also marks whether it is ahead (positive) or behind (negative). Positive in this case.
                }
                // TODO: should keep scanning to find a car properly ahead
            } else {
                *car_ahead_out = car;
                *car_ahead_distance_out = point_to_center_distance - (car_half_length + at_buffer_half);
            }
        }
    }
    LOG_TRACE("here 2");
    // if we did not find a car behind, then we should pick the leading car in the previous lane in the same direction, if it exists
    if (!*car_behind_out && consider_prev_lanes) {
        LOG_TRACE("No car behind found, checking previous lane");
        Lane* prev_lane = lane_get_connection_incoming_straight(lane, map);
        if (prev_lane && lane_get_num_cars(prev_lane) > 0) {
            LOG_TRACE("Found previous lane with %d cars", lane_get_num_cars(prev_lane));
            Car* car = lane_get_car(prev_lane, sim, 0); // first car in the previous lane
            LOG_TRACE("Found car %d in previous lane", car_get_id(car));
            Meters car_half_length = car_get_length(car) / 2.0; // Half the length of the car
            Meters point_to_start_of_lane = point; // distance from the point to the start of the lane
            Meters end_of_prev_lane_to_car_behind = lane_get_length(prev_lane) - car_get_lane_progress_meters(car);
            Meters point_to_center_distance = point_to_start_of_lane + end_of_prev_lane_to_car_behind;

            // Edge case: what if this car is in collision with the at_buffer?
            if (point_to_center_distance <= car_half_length + at_buffer_half) {
                LOG_TRACE("Car %d is in collision with the at_buffer", car_get_id(car));
                // If we already don't have a car at, then we can set it to this car.
                if (!*car_at_out) {
                    *car_at_out = car;
                    *car_at_distance_out = -point_to_center_distance; // center-to-center distance, also marks whether it is ahead (positive) or behind (negative). Negative in this case.
                }
                // TODO: should keep scanning to find a car properly behind
            } else {
                LOG_TRACE("Car %d is behind the point", car_get_id(car));
                *car_behind_out = car;
                *car_behind_distance_out = point_to_center_distance - (car_half_length + at_buffer_half);
            }
        }
    }
    LOG_TRACE("done");
}


static void _update_ttc_headways(Car* car, NearbyVehicles* nearby_vehicles) {
    MetersPerSecond car_speed = car_get_speed(car);
    // first do for all front vehicles
    for (CarIndicator i = 0; i < 3; i++) {
        if (nearby_vehicles->ahead[i].car) {
            const Car* front_car = nearby_vehicles->ahead[i].car;
            MetersPerSecond front_car_speed = car_get_speed(front_car);
            Meters distance_bumper_to_bumper = nearby_vehicles->ahead[i].distance;
            Seconds time_headway = distance_bumper_to_bumper / car_speed;
            if (time_headway < 0) {
                time_headway = INFINITY; // If our speed is negative, we set time headway to infinity since we are not approaching the front car.
            }
            MetersPerSecond closing_in_speed = car_speed - front_car_speed;
            Seconds time_to_collision = closing_in_speed > 0 ? distance_bumper_to_bumper / closing_in_speed : INFINITY; // If we are faster than the front car, we can calculate time to collision, otherwise it is infinite.
            nearby_vehicles->ahead[i].time_to_collision = time_to_collision;
            nearby_vehicles->ahead[i].time_headway = time_headway;
            LOG_TRACE("Car %d: Front vehicle in lane change %d is %d. It is %.2f meters ahead and closing in at %.2f m/s, time to collision = %.2f seconds, time headway = %.2f seconds", 
                      car->id, i, car_get_id(front_car), 
                      nearby_vehicles->ahead[i].distance, 
                      closing_in_speed,
                      time_to_collision,
                      time_headway);
        } else {
            // If there is no front car, we can set the time to collision and headway to INFINITY.
            nearby_vehicles->ahead[i].time_to_collision = INFINITY;
            nearby_vehicles->ahead[i].time_headway = INFINITY;
        }
    }
    // then do for all rear vehicles
    for (CarIndicator i = 0; i < 3; i++) {
        if (nearby_vehicles->behind[i].car) {
            const Car* rear_car = nearby_vehicles->behind[i].car;
            MetersPerSecond rear_car_speed = car_get_speed(rear_car);
            Meters distance_bumper_to_bumper = nearby_vehicles->behind[i].distance;
            Seconds time_headway = distance_bumper_to_bumper / rear_car_speed;
            if (time_headway < 0) {
                time_headway = INFINITY; // If the rear car's speed is negative, we set time headway to infinity since it is not approaching us.
            }
            MetersPerSecond closing_in_speed = rear_car_speed - car_speed;
            Seconds time_to_collision = closing_in_speed > 0 ? distance_bumper_to_bumper / closing_in_speed : INFINITY; // If we are slower than the rear car, we can calculate time to collision, otherwise it is infinite.
            nearby_vehicles->behind[i].time_to_collision = time_to_collision;
            nearby_vehicles->behind[i].time_headway = time_headway;
            LOG_TRACE("Car %d: Rear vehicle in lane change %d is %d. It is %.2f meters behind and closing in at %.2f m/s, time to collision = %.2f seconds, time headway = %.2f seconds", 
                      car->id, i, car_get_id(rear_car), 
                      nearby_vehicles->behind[i].distance, 
                      closing_in_speed,
                      time_to_collision,
                      time_headway);
        } else {
            // If there is no rear car, we can set the time to collision and headway to INFINITY.
            nearby_vehicles->behind[i].time_to_collision = INFINITY;
            nearby_vehicles->behind[i].time_headway = INFINITY;
        }
    }
    // then do for all colliding vehicles
    for (CarIndicator i = 0; i < 3; i++) {
        if (nearby_vehicles->colliding[i].car) {
            nearby_vehicles->colliding[i].time_headway = 0; // For colliding vehicles, time headway is 0 since they are already colliding.
            nearby_vehicles->colliding[i].time_to_collision = 0; // For colliding vehicles, time to collision is 0 since they are already colliding.
            LOG_TRACE("Car %d: Colliding vehicle in lane change %d is %d. It is %.2f meters away, time to collision = 0 seconds, time headway = 0 seconds", 
                      car->id, i, car_get_id(nearby_vehicles->colliding[i].car), 
                      nearby_vehicles->colliding[i].distance);
        } else {
            // If there is no colliding car, we can set the time to collision and headway to INFINITY.
            nearby_vehicles->colliding[i].time_to_collision = INFINITY;
            nearby_vehicles->colliding[i].time_headway = INFINITY;
        }
    }
    // for turn front vehicles, logic is similar to front vehicles.
    for (CarIndicator i = 0; i < 3; i++) {
        if (nearby_vehicles->after_next_turn[i].car) {
            const Car* front_car = nearby_vehicles->after_next_turn[i].car;
            MetersPerSecond front_car_speed = car_get_speed(front_car);
            Meters distance_bumper_to_bumper = nearby_vehicles->after_next_turn[i].distance;
            Seconds time_headway = distance_bumper_to_bumper / car_speed;
            if (time_headway < 0) {
                time_headway = INFINITY; // If our speed is negative, we set time headway to infinity since we are not approaching the front car.
            }
            MetersPerSecond closing_in_speed = car_speed - front_car_speed;
            Seconds time_to_collision = closing_in_speed > 0 ? distance_bumper_to_bumper / closing_in_speed : INFINITY; // If we are faster than the front car, we can calculate time to collision, otherwise it is infinite.
            nearby_vehicles->after_next_turn[i].time_to_collision = time_to_collision;
            nearby_vehicles->after_next_turn[i].time_headway = time_headway;
            LOG_TRACE("Car %d: Front vehicle in turn %d is %d. It is %.2f meters ahead and closing in at %.2f m/s, time to collision = %.2f seconds, time headway = %.2f seconds", 
                      car->id, i, car_get_id(front_car), 
                      nearby_vehicles->after_next_turn[i].distance, 
                      closing_in_speed,
                      time_to_collision,
                      time_headway);
        } else {
            // If there is no front car, we can set the time to collision and headway to INFINITY.
            nearby_vehicles->after_next_turn[i].time_to_collision = INFINITY;
            nearby_vehicles->after_next_turn[i].time_headway = INFINITY;
        }
    }
    // for merge approaching vehicles, logic is similar to rear vehicles.
    for (CarIndicator i = 0; i < 3; i++) {
        if (nearby_vehicles->incoming_from_previous_turn[i].car) {
            const Car* rear_car = nearby_vehicles->incoming_from_previous_turn[i].car;
            MetersPerSecond rear_car_speed = car_get_speed(rear_car);
            Meters distance_bumper_to_bumper = nearby_vehicles->incoming_from_previous_turn[i].distance;
            Seconds time_headway = distance_bumper_to_bumper / rear_car_speed;
            if (time_headway < 0) {
                time_headway = INFINITY; // If the rear car's speed is negative, we set time headway to infinity since it is not approaching us.
            }
            MetersPerSecond closing_in_speed = rear_car_speed - car_speed;
            Seconds time_to_collision = closing_in_speed > 0 ? distance_bumper_to_bumper / closing_in_speed : INFINITY; // If we are slower than the rear car, we can calculate time to collision, otherwise it is infinite.
            nearby_vehicles->incoming_from_previous_turn[i].time_to_collision = time_to_collision;
            nearby_vehicles->incoming_from_previous_turn[i].time_headway = time_headway;
            LOG_TRACE("Car %d: Approaching vehicle in merge %d is %d. It is %.2f meters behind and closing in at %.2f m/s, time to collision = %.2f seconds, time headway = %.2f seconds", 
                      car->id, i, car_get_id(rear_car), 
                      nearby_vehicles->incoming_from_previous_turn[i].distance, 
                      closing_in_speed,
                      time_to_collision,
                      time_headway);
        } else {
            // If there is no rear car, we can set the time to collision and headway to INFINITY.
            nearby_vehicles->incoming_from_previous_turn[i].time_to_collision = INFINITY;
            nearby_vehicles->incoming_from_previous_turn[i].time_headway = INFINITY;
        }
    }

    // For vehicles turning into the same lane at us at intersections:
    for (CarIndicator our_turn = 0; our_turn < 3; our_turn++) {
        for (CarIndicator other_turn = 0; other_turn < 3; other_turn++) {
            int offset = our_turn * 9 + other_turn * 3;
            const Car* car_ahead = nearby_vehicles->turning_into_same_lane[offset + 0].car;
            Meters car_ahead_distance = nearby_vehicles->turning_into_same_lane[offset + 0].distance;
            if (car_ahead) {
                MetersPerSecond front_car_speed = car_get_speed(car_ahead);
                Meters distance_bumper_to_bumper = car_ahead_distance;
                Seconds time_headway = distance_bumper_to_bumper / car_speed;
                if (time_headway < 0) {
                    time_headway = INFINITY; // If our speed is negative, we set time headway to infinity since we are not approaching the front car.
                }
                MetersPerSecond closing_in_speed = car_speed - front_car_speed;
                Seconds time_to_collision = closing_in_speed > 0 ? distance_bumper_to_bumper / closing_in_speed : INFINITY; // If we are faster than the front car, we can calculate time to collision, otherwise it is infinite.
                nearby_vehicles->turning_into_same_lane[offset + 0].time_to_collision = time_to_collision;
                nearby_vehicles->turning_into_same_lane[offset + 0].time_headway = time_headway;
                LOG_TRACE("Car %d: Front vehicle turning into same lane for our turn %d from their turn %d is %d. It is %.2f meters ahead and closing in at %.2f m/s, time to collision = %.2f seconds, time headway = %.2f seconds", 
                          car->id, our_turn, other_turn, car_get_id(car_ahead), 
                          nearby_vehicles->turning_into_same_lane[offset + 0].distance, 
                          closing_in_speed,
                          time_to_collision,
                          time_headway);
            } else {
                // If there is no front car, we can set the time to collision and headway to INFINITY.
                nearby_vehicles->turning_into_same_lane[offset + 0].time_to_collision = INFINITY;
                nearby_vehicles->turning_into_same_lane[offset + 0].time_headway = INFINITY;
            }

            const Car* car_at = nearby_vehicles->turning_into_same_lane[offset + 1].car;
            if (car_at) {
                // THW and TTC are both 0 for colliding vehicles
                nearby_vehicles->turning_into_same_lane[offset + 1].time_to_collision = 0;
                nearby_vehicles->turning_into_same_lane[offset + 1].time_headway = 0;
                LOG_TRACE("Car %d: Colliding vehicle turning into same lane for our turn %d from their turn %d is %d. It is %.2f meters away, time to collision = 0 seconds, time headway = 0 seconds", 
                          car->id, our_turn, other_turn, car_get_id(car_at), 
                          nearby_vehicles->turning_into_same_lane[offset + 1].distance);
            } else {
                // If there is no colliding car, we can set the time to collision and headway to INFINITY.
                nearby_vehicles->turning_into_same_lane[offset + 1].time_to_collision = INFINITY;
                nearby_vehicles->turning_into_same_lane[offset + 1].time_headway = INFINITY;
            }

            const Car* car_behind = nearby_vehicles->turning_into_same_lane[offset + 2].car;
            Meters car_behind_distance = nearby_vehicles->turning_into_same_lane[offset + 2].distance;
            if (car_behind) {
                MetersPerSecond rear_car_speed = car_get_speed(car_behind);
                Meters distance_bumper_to_bumper = car_behind_distance;
                Seconds time_headway = distance_bumper_to_bumper / rear_car_speed;
                if (time_headway < 0) {
                    time_headway = INFINITY; // If the rear car's speed is negative, we set time headway to infinity since it is not approaching us.
                }
                MetersPerSecond closing_in_speed = rear_car_speed - car_speed;
                Seconds time_to_collision = closing_in_speed > 0 ? distance_bumper_to_bumper / closing_in_speed : INFINITY; // If we are slower than the rear car, we can calculate time to collision, otherwise it is infinite.
                nearby_vehicles->turning_into_same_lane[offset + 2].time_to_collision = time_to_collision;
                nearby_vehicles->turning_into_same_lane[offset + 2].time_headway = time_headway;
                LOG_TRACE("Car %d: Rear vehicle turning into same lane for our turn %d from their turn %d is %d. It is %.2f meters behind and closing in at %.2f m/s, time to collision = %.2f seconds, time headway = %.2f seconds", 
                          car->id, our_turn, other_turn, car_get_id(car_behind), 
                          nearby_vehicles->turning_into_same_lane[offset + 2].distance, 
                          closing_in_speed,
                          time_to_collision,
                          time_headway);
            } else {
                // If there is no rear car, we can set the time to collision and headway to INFINITY.
                nearby_vehicles->turning_into_same_lane[offset + 2].time_to_collision = INFINITY;
                nearby_vehicles->turning_into_same_lane[offset + 2].time_headway = INFINITY;
            }
        }
    }
}


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


    LOG_TRACE("\nBuilding situational awareness for car %d:", car->id);
    Map* map = sim_get_map(sim);
    Meters car_length = car_get_length(car);
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
        situation->is_an_intersection_upcoming = false;
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
    situation->is_merge_possible = lane_is_merge_available(lane);
    situation->is_exit_possible = lane_is_exit_lane_available(lane, lane_progress);
    situation->is_on_entry_ramp = road ? road_is_merge_available(road, map) : false;
    situation->is_exit_available = road ? road_is_exit_road_available(road, map, lane_progress) : false;
    situation->is_exit_available_eventually = road ? road_is_exit_road_eventually_available(road, map, lane_progress) : false;
    situation->is_lane_change_left_possible = situation->lane_left != NULL || situation->is_merge_possible;
    situation->is_lane_change_right_possible = situation->lane_right != NULL || situation->is_exit_possible;

    LOG_TRACE("Car %d: Lane change and merge/exit possibilities: left=%d, right=%d, merge=%d, exit=%d, entry ramp=%d, exit available now=%d, exit available eventually=%d", car->id, situation->is_lane_change_left_possible, situation->is_lane_change_right_possible, situation->is_merge_possible, situation->is_exit_possible, situation->is_on_entry_ramp, situation->is_exit_available, situation->is_exit_available_eventually);

    situation->merges_into_lane = lane_get_merge_into(lane, map);
    situation->exit_lane = (situation->is_exit_available || situation->is_exit_available_eventually) ? lane_get_exit_lane(lane, map) : NULL;
    situation->lane_target_for_indicator[INDICATOR_LEFT] = situation->merges_into_lane ? situation->merges_into_lane : situation->lane_left;
    situation->lane_target_for_indicator[INDICATOR_NONE] = situation->lane;
    situation->lane_target_for_indicator[INDICATOR_RIGHT] = (situation->exit_lane && situation->is_exit_possible)  ? situation->exit_lane : situation->lane_right;
    LaneId left_outcome_id = situation->lane_target_for_indicator[INDICATOR_LEFT] ? situation->lane_target_for_indicator[INDICATOR_LEFT]->id : ID_NULL;
    LaneId straight_outcome_id = situation->lane_target_for_indicator[INDICATOR_NONE] ? situation->lane_target_for_indicator[INDICATOR_NONE]->id : ID_NULL;
    LaneId right_outcome_id = situation->lane_target_for_indicator[INDICATOR_RIGHT] ? situation->lane_target_for_indicator[INDICATOR_RIGHT]->id : ID_NULL;
    LOG_TRACE("Car %d: Lane indicator outcomes: left=%d (is a merge = %d), none=%d, right=%d (is an exit = %d)", car->id, left_outcome_id, situation->merges_into_lane != NULL, straight_outcome_id, right_outcome_id, situation->exit_lane != NULL);



    // build nearby vehicles information:

    // start with what happens on the current lane:
    Car* car_ahead = NULL;
    Meters car_ahead_distance = INFINITY;
    Car* car_behind = NULL;
    Meters car_behind_distance = INFINITY;
    Meters car_at_distance = INFINITY;
    Car* car_at = NULL;
    _find_vehicle_behind_and_at_and_ahead_of_point(sim, lane, lane_progress_m, car_length, &car_behind, &car_behind_distance, &car_at, &car_at_distance, &car_ahead, &car_ahead_distance, true, true);
    LOG_TRACE("Car %d: Found vehicle ahead: %d (distance = %.2f m), behind: %d (distance = %.2f m), at: %d (distance = %.2f m)", car->id, car_ahead ? car_ahead->id : -1, car_ahead_distance, car_behind ? car_behind->id : -1, car_behind_distance, car_at ? car_at->id : -1, car_at_distance);
    situation->nearby_vehicles.ahead[INDICATOR_NONE].car = car_ahead;
    situation->nearby_vehicles.ahead[INDICATOR_NONE].distance = car_ahead_distance;
    situation->nearby_vehicles.behind[INDICATOR_NONE].car = car_behind;
    situation->nearby_vehicles.behind[INDICATOR_NONE].distance = car_behind_distance;
    if (car_at == car) {
        // If the car_at is the same as the current car, we should not consider it as colliding.
        car_at = NULL;
        car_at_distance = INFINITY;
    }
    situation->nearby_vehicles.colliding[INDICATOR_NONE].car = car_at;
    situation->nearby_vehicles.colliding[INDICATOR_NONE].distance = car_at_distance;
    
    // what happens if we switch lane left:
    const Lane* target_lane_left = situation->lane_target_for_indicator[INDICATOR_LEFT];
    Car* car_ahead_left = NULL;
    Meters car_ahead_distance_left = INFINITY;
    Car* car_behind_left = NULL;
    Meters car_behind_distance_left = INFINITY;
    Car* car_at_left = NULL;
    Meters car_at_distance_left = INFINITY;
    Meters hypothetical_position = calculate_hypothetical_position_on_lane_change(car, lane, target_lane_left, map);
    _find_vehicle_behind_and_at_and_ahead_of_point(sim, target_lane_left, hypothetical_position, car_length, &car_behind_left, &car_behind_distance_left, &car_at_left, &car_at_distance_left, &car_ahead_left, &car_ahead_distance_left, true, true);
    LOG_TRACE("Car %d: Found vehicle ahead on left lane: %d (distance = %.2f m), behind: %d (distance = %.2f m), at: %d (distance = %.2f m)", car->id, car_ahead_left ? car_ahead_left->id : -1, car_ahead_distance_left, car_behind_left ? car_behind_left->id : -1, car_behind_distance_left, car_at_left ? car_at_left->id : -1, car_at_distance_left);
    situation->nearby_vehicles.ahead[INDICATOR_LEFT].car = car_ahead_left;
    situation->nearby_vehicles.ahead[INDICATOR_LEFT].distance = car_ahead_distance_left;
    situation->nearby_vehicles.behind[INDICATOR_LEFT].car = car_behind_left;
    situation->nearby_vehicles.behind[INDICATOR_LEFT].distance = car_behind_distance_left;
    situation->nearby_vehicles.colliding[INDICATOR_LEFT].car = car_at_left;
    situation->nearby_vehicles.colliding[INDICATOR_LEFT].distance = car_at_distance_left;

    // what happens if we switch lane right:
    const Lane* target_lane_right = situation->lane_target_for_indicator[INDICATOR_RIGHT];
    Car* car_ahead_right = NULL;
    Meters car_ahead_distance_right = INFINITY;
    Car* car_behind_right = NULL;
    Meters car_behind_distance_right = INFINITY;
    Car* car_at_right = NULL;
    Meters car_at_distance_right = INFINITY;
    hypothetical_position = calculate_hypothetical_position_on_lane_change(car, lane, target_lane_right, map);
    _find_vehicle_behind_and_at_and_ahead_of_point(sim, target_lane_right, hypothetical_position, car_length, &car_behind_right, &car_behind_distance_right, &car_at_right, &car_at_distance_right, &car_ahead_right, &car_ahead_distance_right, true, true);
    LOG_TRACE("Car %d: Found vehicle ahead on right lane: %d (distance = %.2f m), behind: %d (distance = %.2f m), at: %d (distance = %.2f m)", car->id, car_ahead_right ? car_ahead_right->id : -1, car_ahead_distance_right, car_behind_right ? car_behind_right->id : -1, car_behind_distance_right, car_at_right ? car_at_right->id : -1, car_at_distance_right);
    situation->nearby_vehicles.ahead[INDICATOR_RIGHT].car = car_ahead_right;
    situation->nearby_vehicles.ahead[INDICATOR_RIGHT].distance = car_ahead_distance_right;
    situation->nearby_vehicles.behind[INDICATOR_RIGHT].car = car_behind_right;
    situation->nearby_vehicles.behind[INDICATOR_RIGHT].distance = car_behind_distance_right;
    situation->nearby_vehicles.colliding[INDICATOR_RIGHT].car = car_at_right;
    situation->nearby_vehicles.colliding[INDICATOR_RIGHT].distance = car_at_distance_right;

    bool we_are_leading_on_our_lane = car_get_lane_rank(car) == 0;
    bool we_are_last_on_our_lane = car_get_lane_rank(car) == lane_get_num_cars(lane) - 1;

    // find the closest vehicle after each turn (only if we are leading in our lane and preparing to turn):
    for (CarIndicator turn_indicator = 0; turn_indicator < 3; turn_indicator++) {
        Car* closest_car = NULL;
        Meters closest_distance = INFINITY;
        const Lane* next_lane = situation->lane_next_after_turn[turn_indicator];
        if (we_are_leading_on_our_lane && next_lane && lane_get_num_cars(next_lane) > 0) {
            // The closest car in the next lane is the last car in the lane.
            int last_car_rank = lane_get_num_cars(next_lane) - 1;
            closest_car = lane_get_car(next_lane, sim, last_car_rank);
            Meters closest_car_progress_m = car_get_lane_progress_meters(closest_car);
            Meters distance_to_closest_car = closest_car_progress_m + distance_to_end_of_lane;
            closest_distance = distance_to_closest_car;
            LOG_TRACE("Car %d: Closest car in next lane %d for turn indicator %d is %d (distance = %.2f m)", car->id, next_lane->id, turn_indicator, closest_car->id, closest_distance);
        }
        situation->nearby_vehicles.after_next_turn[turn_indicator].car = closest_car;
        situation->nearby_vehicles.after_next_turn[turn_indicator].distance = closest_distance;
    }

    // find closest vehicle turning into our lane from each incoming direction (only if we are last on our lane and worried about incoming cars tailgating us):
    for (CarIndicator turn_indicator = 0; turn_indicator < 3; turn_indicator++) {
        Car* closest_car = NULL;
        Meters closest_distance = INFINITY;
        LaneId incoming_lane_id = lane->connections_incoming[turn_indicator];
        Lane* incoming_lane = incoming_lane_id != ID_NULL ? map_get_lane(map, incoming_lane_id) : NULL;
        if (we_are_last_on_our_lane && incoming_lane && lane_get_num_cars(incoming_lane) > 0) {
            // The closest car in the incoming lane is the first car in the lane.
            int first_car_rank = 0;
            closest_car = lane_get_car(incoming_lane, sim, first_car_rank);
            Meters closest_car_distance_to_its_lane_end = lane_get_length(incoming_lane) - car_get_lane_progress_meters(closest_car);
            closest_distance = closest_car_distance_to_its_lane_end + lane_progress_m;
            LOG_TRACE("Car %d: Closest car in incoming lane %d for turn indicator %d is %d (distance = %.2f m)", car->id, incoming_lane->id, turn_indicator, closest_car->id, closest_distance);
        }
        situation->nearby_vehicles.incoming_from_previous_turn[turn_indicator].car = closest_car;
        situation->nearby_vehicles.incoming_from_previous_turn[turn_indicator].distance = closest_distance;
    }

    // Worry about vehicles turning into the same lane as us at intersections:
    // We need to consider all possible turns we might take (if we are on an intersection, then we are already committed to one turn, otherwise we might take any of the three turns).
    // Don't worry if we haven't entered the intersection yet and there is no intersection upcoming or even if there is an intersection upcoming, we are too far away from it or not the lead car.
    const Lane* ego_intersection_entry_lane = (situation->is_an_intersection_upcoming && car_get_lane_rank(car) == 0 && situation->is_approaching_end_of_lane) ? lane : (situation->is_on_intersection ? lane_get_connection_incoming_straight(lane, map) : NULL);
    
    for (CarIndicator our_possible_turn = 0; our_possible_turn < 3; our_possible_turn++) {
        LaneId ego_intersection_lane_id = ego_intersection_entry_lane ? ego_intersection_entry_lane->connections[our_possible_turn] : ID_NULL;
        Lane* ego_intersection_lane = ego_intersection_lane_id != ID_NULL ? map_get_lane(map, ego_intersection_lane_id) : NULL;
        Lane* ego_intersection_exit_lane = ego_intersection_lane ? lane_get_connection_straight(ego_intersection_lane, map) : NULL;
        Meters ego_distance_to_exit = ego_intersection_exit_lane ? (situation->is_on_intersection ? distance_to_end_of_lane : (distance_to_end_of_lane + ego_intersection_lane->length)) : INFINITY;

        // if we are already turning, then do not worry about other exits
        if (situation->is_on_intersection && ego_intersection_lane != lane) {
            // set everything to NULL/INFINITY so that the code further down just ignores this turn.
            ego_intersection_lane = NULL;
            ego_intersection_exit_lane = NULL;
            ego_distance_to_exit = INFINITY;
        }


        // Other lanes joining to the ego_intersection_exit_lane:
        for (CarIndicator other_turn = 0; other_turn < 3; other_turn++) {
            LaneId other_intersection_lane_id = ego_intersection_exit_lane ? ego_intersection_exit_lane->connections_incoming[other_turn] : ID_NULL;
            Lane* other_intersection_lane = other_intersection_lane_id != ID_NULL ? map_get_lane(map, other_intersection_lane_id) : NULL;
            Lane* other_intersection_entry_lane = other_intersection_lane ? lane_get_connection_incoming_straight(other_intersection_lane, map) : NULL;

            // we need to find the point on other_intersection_lane or on other_intersection_entry_lane (whereever it is) that is at the distance ego_distance_to_exit to the exit lane.
            Lane* hypothetical_position_lane = other_intersection_lane;
            Meters hypothetical_position = hypothetical_position_lane ? (hypothetical_position_lane->length - ego_distance_to_exit) : INFINITY;
            // if this is negative, then we need to look into the other_intersection_entry_lane
            if (hypothetical_position < 0 && other_intersection_entry_lane) {
                hypothetical_position = other_intersection_entry_lane->length + hypothetical_position;
                hypothetical_position_lane = other_intersection_entry_lane;
            }
            // if still negative, then it is too far back, we can ignore it.
            if (hypothetical_position < 0) {
                hypothetical_position_lane = NULL;
                hypothetical_position = INFINITY;
            }

            // now that we have the hypothetical position, we can record the ahead/at/behind vehicles relative to that point.
            Car* car_ahead_other = NULL;
            Meters car_ahead_distance_other = INFINITY;
            Car* car_behind_other = NULL;
            Meters car_behind_distance_other = INFINITY;
            Car* car_at_other = NULL;
            Meters car_at_distance_other = INFINITY;
            if (hypothetical_position_lane) {
                bool consider_next_of_the_hypothetical_lane = hypothetical_position_lane == other_intersection_entry_lane; // so that both the entry and the on-intersection lanes are considered.
                bool consider_previous_of_the_hypothetical_lane = hypothetical_position_lane == other_intersection_lane; // so that both the entry and the on-intersection lanes are considered.
                _find_vehicle_behind_and_at_and_ahead_of_point(sim, hypothetical_position_lane, hypothetical_position, car_length, &car_behind_other, &car_behind_distance_other, &car_at_other, &car_at_distance_other, &car_ahead_other, &car_ahead_distance_other, consider_previous_of_the_hypothetical_lane, consider_next_of_the_hypothetical_lane);
            }
            if (car_at_other == car) {
                // If the car_at is the same as the current car, we should not consider it as colliding.
                car_at_other = NULL;
                car_at_distance_other = INFINITY;
            }
            LOG_TRACE("Car %d: For our turn %d, other lane %d (from turn %d) hypothetical position is %.2f on lane %d. Found vehicle ahead: %d (distance = %.2f m), behind: %d (distance = %.2f m), at: %d (distance = %.2f m)", 
                      car->id, our_possible_turn, other_intersection_lane ? other_intersection_lane->id : -1, other_turn, 
                      hypothetical_position, hypothetical_position_lane ? hypothetical_position_lane->id : -1,
                      car_ahead_other ? car_ahead_other->id : -1, car_ahead_distance_other, 
                      car_behind_other ? car_behind_other->id : -1, car_behind_distance_other, 
                      car_at_other ? car_at_other->id : -1, car_at_distance_other);
            
            int offset = our_possible_turn * 9 + other_turn * 3;
            situation->nearby_vehicles.turning_into_same_lane[offset + 0].car = car_ahead_other;
            situation->nearby_vehicles.turning_into_same_lane[offset + 0].distance = car_ahead_distance_other;
            situation->nearby_vehicles.turning_into_same_lane[offset + 1].car = car_at_other;
            situation->nearby_vehicles.turning_into_same_lane[offset + 1].distance = car_at_distance_other;
            situation->nearby_vehicles.turning_into_same_lane[offset + 2].car = car_behind_other;
            situation->nearby_vehicles.turning_into_same_lane[offset + 2].distance = car_behind_distance_other;
        }
    }

    _update_ttc_headways(car, &situation->nearby_vehicles);


    int my_rank = car_get_lane_rank(car);
    if (my_rank > 0) {
        // We are not the first car in the lane, so we can find the lead vehicle.
        situation->lead_vehicle = lane_get_car(lane, sim, my_rank - 1);
        situation->distance_to_lead_vehicle = car_get_lane_progress_meters(situation->lead_vehicle) - lane_progress_m;
    } else {
        // We are the first car in the lane, so there is no lead vehicle, but maybe on next lane.
        Lane* lane_next = lane_get_connection_straight(lane, map);
        Car* lane_next_last_vehicle = (lane_next && lane_get_num_cars(lane_next) > 0) ? lane_get_car(lane_next, sim, lane_get_num_cars(lane_next) - 1) : NULL;
        situation->lead_vehicle = lane_next_last_vehicle;
        situation->distance_to_lead_vehicle = situation->lead_vehicle ? (car_get_lane_progress_meters(situation->lead_vehicle) + distance_to_end_of_lane) : INFINITY;
    }

    if (my_rank < lane_get_num_cars(lane) - 1) {
        // We are not the last car in the lane, so we can find the following vehicle.
        situation->following_vehicle = lane_get_car(lane, sim, my_rank + 1);
        situation->distance_to_following_vehicle = lane_progress_m - car_get_lane_progress_meters(situation->following_vehicle);
    } else {
        // We are the last car in the lane, so there is no following vehicle, but maybe on previous lane.
        Lane* lane_prev = lane_get_connection_incoming_straight(lane, map);
        Car* lane_prev_first_vehicle = (lane_prev && lane_get_num_cars(lane_prev) > 0) ? lane_get_car(lane_prev, sim, 0) : NULL;
        situation->following_vehicle = lane_prev_first_vehicle;
        situation->distance_to_following_vehicle = situation->following_vehicle ? (lane_progress_m + lane_get_length(lane_prev) - car_get_lane_progress_meters(situation->following_vehicle)) : INFINITY;
    }
    situation->is_vehicle_ahead = situation->lead_vehicle != NULL;
    situation->is_vehicle_behind = situation->following_vehicle != NULL;
    LOG_TRACE("Car %d: Lead vehicle: %d (distance = %.2f m), following vehicle: %d (distance = %.2f m)", car->id, situation->is_vehicle_ahead ? situation->lead_vehicle->id : -1, situation->distance_to_lead_vehicle, situation->is_vehicle_behind ? situation->following_vehicle->id : -1,  situation->distance_to_following_vehicle);

    situation->braking_distance = car_compute_braking_distance(car);
    LOG_TRACE("Car %d: Braking distance: capable=%.2f m, preferred=%.2f m, preferred smooth=%.2f m", car->id, situation->braking_distance.capable, situation->braking_distance.preferred, situation->braking_distance.preferred_smooth);

    if (car->prev_lane_id == lane->id) {
        // No lane change detected, just update the position and speed
        if (car_speed < situation->slowest_speed_yet_on_current_lane) {
            situation->slowest_speed_yet_on_current_lane = car_speed;
            situation->slowest_speed_position = lane_progress_m;
            LOG_TRACE("Car %d has achieved slowest speed yet on current lane = %.2f mph at progress %.2f", car->id, to_mph(situation->slowest_speed_yet_on_current_lane), situation->slowest_speed_position / lane_length);
        }
    } else {
        // Lane change detected, reset the slowest speed
        LOG_TRACE("Car %d has just changed lanes from %d to %d. So, resetting slowest speed yet on current lane and the corresponding position.", car->id, car->prev_lane_id, lane->id);
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
