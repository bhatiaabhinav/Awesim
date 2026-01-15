#include "ai.h"
#include "logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

CarIndicator turn_sample_possible(const SituationalAwareness* situation) {
    if (situation->is_approaching_dead_end) {
        return INDICATOR_NONE;
    }
    CarIndicator possibles[3];
    int num_possibles = 0;
    for (int i = 0; i < 3; i++) {
        if (situation->is_turn_possible[i]) {
            possibles[num_possibles++] = i;
        }
    }
    if (num_possibles == 0) {
        return INDICATOR_NONE;
    }
    return possibles[rand_int(num_possibles)];
}

CarIndicator lane_change_sample_possible(const SituationalAwareness* situation) {
    CarIndicator possibles[2];
    int num_possibles = 0;
    if (situation->is_lane_change_left_possible) {
        possibles[num_possibles++] = INDICATOR_LEFT;
    }
    if (situation->is_lane_change_right_possible) {
        possibles[num_possibles++] = INDICATOR_RIGHT;
    }
    if (num_possibles == 0) {
        return INDICATOR_NONE;
    }
    if (num_possibles == 1) {
        return possibles[0];
    }
    int chosen = rand_int(num_possibles);
    return possibles[chosen];
}

Meters calculate_hypothetical_position_on_lane_change(const Car* car, const Lane* lane, const Lane* lane_target, Map* map) {
    if (!car || !lane_target) return 0;

    Lane* merges_into = lane_get_merge_into(lane, map);
    Lane* exit_lane = lane_get_exit_lane(lane, map);
    double my_progress = car_get_lane_progress(car);
    Meters my_progress_m = car_get_lane_progress_meters(car);
    double my_hypothetical_target_progress_m;
    if (lane_target == merges_into) {
        my_hypothetical_target_progress_m = my_progress_m + lane->merges_into_start * merges_into->length;
    } else if (lane_target == exit_lane) {
        my_hypothetical_target_progress_m = (my_progress - lane->exit_lane_start) * lane->length;
    } else {
        my_hypothetical_target_progress_m = my_progress * lane_target->length; // since this is just an adjacent lane on the same road, progress is the same
    }
    return my_hypothetical_target_progress_m;
}

bool car_is_lane_change_dangerous(Car* car, Simulation* sim, const SituationalAwareness* situation, CarIndicator lane_change_indicator, Seconds time_headway_threshold, Meters buffer) {
    const Lane* lane = situation->lane;
    const Lane* lane_target = situation->lane_target_for_indicator[lane_change_indicator];
    if (lane_target == lane) return false;      // no lane change
    if (lane_target == NULL) return true;       // Will fly off the road!

    NearbyVehicle car_ahead = situation->nearby_vehicles.ahead[lane_change_indicator];
    NearbyVehicle car_behind = situation->nearby_vehicles.behind[lane_change_indicator];
    NearbyVehicle car_colliding = situation->nearby_vehicles.colliding[lane_change_indicator];

    if (car_colliding.car) {
        return true; // there is a car in the target lane that is colliding with us
    }
    if (car_ahead.car && (car_ahead.time_headway < time_headway_threshold || car_ahead.distance < buffer)) {
        // We will be too close to the car ahead
        return true;
    }
    if (car_behind.car && (car_behind.time_headway < time_headway_threshold || car_behind.distance < buffer)) {
        // We will be too close to the car behind
        return true;
    }
    if (car->lane_progress_meters - car_get_length(car) / 2 < meters(15.0)) {
        // We just entered the lane, do not change lanes immediately.
        return true;
    }
    return false;
}


bool car_has_something_to_yield_to_at_intersection(const Car* self, Simulation* sim, const SituationalAwareness* situation, CarIndicator turn_indicator) {
    Map* map = sim_get_map(sim);

    const Intersection* intersection = situation->intersection;

    Road* inbound_roads[4] = {
        [DIRECTION_EAST] = intersection_get_road_eastbound_from(intersection, map),
        [DIRECTION_WEST] = intersection_get_road_westbound_from(intersection, map),
        [DIRECTION_NORTH] = intersection_get_road_northbound_from(intersection, map),
        [DIRECTION_SOUTH] = intersection_get_road_southbound_from(intersection, map),
    };
    Direction my_dir = situation->lane->direction;
    Road* road_from_left = inbound_roads[direction_perp_cw(my_dir)];
    Road* road_from_right = inbound_roads[direction_perp_ccw(my_dir)];
    Road* road_from_opposite = inbound_roads[direction_opposite(my_dir)];

    Lane* incoming_lanes_to_worry_about[3 * MAX_NUM_LANES_PER_ROAD] = {NULL};           // max 3 roads to worry about, each with max lanes.
    int num_incoming_lanes_to_worry_about = 0;
    Lane* intersection_lanes_to_worry_about[MAX_NUM_LANES_PER_INTERSECTION] = {NULL};    // lanes that must not have cars on them
    int num_intersection_lanes_to_worry_about = 0;

    if (intersection->is_T_junction) {
        bool i_am_merging = (intersection->is_T_junction_north_south && (my_dir == DIRECTION_EAST || my_dir == DIRECTION_WEST)) ||
                            (!intersection->is_T_junction_north_south && (my_dir == DIRECTION_NORTH || my_dir == DIRECTION_SOUTH));
        if (i_am_merging) {
            // If turning right, worry about the rightmost lane on the road coming from the left
            if (turn_indicator == INDICATOR_RIGHT) {
                incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = road_get_rightmost_lane(road_from_left, map);
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(road_get_rightmost_lane(road_from_left, map), map);
            } else if (turn_indicator == INDICATOR_LEFT) {
                // If turning left, worry about the leftmost lane on the road coming from the right and all lanes on the road coming from left
                incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = road_get_leftmost_lane(road_from_right, map);
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(road_get_leftmost_lane(road_from_right, map), map);
                for (int i = 0; i < road_get_num_lanes(road_from_left); i++) {
                    incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = road_get_lane(road_from_left, map, i);
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(road_get_lane(road_from_left, map, i), map);
                }
                // also worry about the traffic exiting left from the road coming from the right as it has the right of way over us
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(road_get_leftmost_lane(road_from_right, map), map);
            } else {
                // Not possible to go straight in a T-junction
            }
        } else {
            if (turn_indicator == INDICATOR_RIGHT) {
                // free exit
            } else if (turn_indicator == INDICATOR_LEFT) {
                // exit while yeilding to oncoming traffic
                for (int i = 0; i < road_get_num_lanes(road_from_opposite); i++) {
                    incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = road_get_lane(road_from_opposite, map, i);
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(road_get_lane(road_from_opposite, map, i), map);
                    // if we are exiting on to a single lane road, we need to worry about the oncoming traffic turning into that lane as well
                    if (road_get_num_lanes(road_from_right) == 1) { // assuming both road_from_right and road_to_left have same number of lanes
                        intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_right(road_get_rightmost_lane(road_from_opposite, map), map);
                    }
                }
            } else {
                // free straight through
            }
        }
    } else if (intersection->state != FOUR_WAY_STOP) {
        if (turn_indicator == INDICATOR_RIGHT) {
            // If turning right, worry about the rightmost lane on the road coming from the left
            incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = road_get_rightmost_lane(road_from_left, map);
            intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(road_get_rightmost_lane(road_from_left, map), map);
        } else if (turn_indicator == INDICATOR_LEFT) {
            // If turning left, worry about all the lanes on the road coming from the opposite direction
            for (int i = 0; i < road_get_num_lanes(road_from_opposite); i++) {
                incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = road_get_lane(road_from_opposite, map, i);
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(road_get_lane(road_from_opposite, map, i), map);
            }
            // if we are turning left onto a single lane road, we need to worry about the oncoming traffic turning into that lane as well
            if (road_get_num_lanes(road_from_right) == 1) { // assuming both road_from_right and road_to_left have same number of lanes
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_right(road_get_rightmost_lane(road_from_opposite, map), map);
            }
        }
    } else {
        // Yeilding not revelevant for four-way stop
        return false;
    }

    // Go through the intersection lanes to worry about and if any of them have cars, we have to yield.
    for (int i = 0; i < num_intersection_lanes_to_worry_about; i++) {
        Lane* lane = intersection_lanes_to_worry_about[i];
        if (lane && lane_get_num_cars(lane) > 0) {
            return true;    // have to yield
        }
    }

    // Go through the leading cars on incoming lanes and if any of them are less than 3 seconds away from their lane end, then we have to yield.
    for (int i = 0; i < num_incoming_lanes_to_worry_about; i++) {
        Lane* lane = incoming_lanes_to_worry_about[i];
        if (!lane || lane_get_num_cars(lane) == 0) continue;
        Car* c = lane_get_car(lane, sim, 0); // get the leading car
        if (c) {
            Meters progress_m = car_get_lane_progress_meters(c);
            Meters lane_length = lane_get_length(lane);
            Meters distance_to_end_m = lane_length - progress_m;
            Meters speed = car_get_speed(c);
            Seconds time_to_end_s = (speed > STOP_SPEED_THRESHOLD) ? distance_to_end_m / speed : INFINITY;
            if (time_to_end_s < 3.0) {
                return true;    // have to yield if coming in fast or has reached the stop line even if slow
            }
        }
    }
    return false;
}

void nearby_vehicles_flatten(NearbyVehicles* nearby_vehicles, NearbyVehiclesFlattened* flattened, bool include_outgoing_and_incoming_lanes) {
    int count = 0;

    // Add vehicles from the front lane
    for (int i = 0; i < 3; i++) {
        NearbyVehicle v = nearby_vehicles->ahead[i];
        const Car* c = v.car;
        CarId id = c ? c->id : ID_NULL;
        flattened->car_ids[count] = id;
        flattened->distances[count] = v.distance;
        flattened->time_to_collisions[count] = v.time_to_collision;
        flattened->time_headways[count] = v.time_headway;
        count++;
    }

    // Add vehicles from the colliding lane
    for (int i = 0; i < 3; i++) {
        NearbyVehicle v = nearby_vehicles->colliding[i];
        const Car* c = v.car;
        CarId id = c ? c->id : ID_NULL;
        flattened->car_ids[count] = id;
        flattened->distances[count] = v.distance;
        flattened->time_to_collisions[count] = v.time_to_collision;
        flattened->time_headways[count] = v.time_headway;
        count++;
    }

    // Add vehicles from the rear lane
    for (int i = 0; i < 3; i++) {
        NearbyVehicle v = nearby_vehicles->behind[i];
        const Car* c = v.car;
        CarId id = c ? c->id : ID_NULL;
        flattened->car_ids[count] = id;
        flattened->distances[count] = v.distance;
        flattened->time_to_collisions[count] = v.time_to_collision;
        flattened->time_headways[count] = v.time_headway;
        count++;
    }

    // Add vehicles that may turn into same lane when we turn into something at an intersection
    for (CarIndicator our_turn = 0; our_turn < 3; our_turn++) {
        for (CarIndicator their_turn = 0; their_turn < 3; their_turn++) {
            int offset = (our_turn * 3 + their_turn) * 3;
            for (int i = 0; i < 3; i++) {
                NearbyVehicle v = nearby_vehicles->turning_into_same_lane[offset + i];
                const Car* c = v.car;
                CarId id = c ? c->id : ID_NULL;
                flattened->car_ids[count] = id;
                flattened->distances[count] = v.distance;
                flattened->time_to_collisions[count] = v.time_to_collision;
                flattened->time_headways[count] = v.time_headway;
                count++;
            }
        }
    }

    if (include_outgoing_and_incoming_lanes) {

        // Add vehicles from the outgoing lane
        for (int i = 0; i < 3; i++) {
            NearbyVehicle v = nearby_vehicles->after_next_turn[i];
            const Car* c = v.car;
            CarId id = c ? c->id : ID_NULL;
            flattened->car_ids[count] = id;
            flattened->distances[count] = v.distance;
            flattened->time_to_collisions[count] = v.time_to_collision;
            flattened->time_headways[count] = v.time_headway;
            count++;
        }

        // Add vehicles from the incoming lanes
        for (int i = 0; i < 3; i++) {
            NearbyVehicle v = nearby_vehicles->incoming_from_previous_turn[i];
            const Car* c = v.car;
            CarId id = c ? c->id : ID_NULL;
            flattened->car_ids[count] = id;
            flattened->distances[count] = v.distance;
            flattened->time_to_collisions[count] = v.time_to_collision;
            flattened->time_headways[count] = v.time_headway;
            count++;
        }
    }
    
    flattened->count = count;

    // Fill the rest with ID_NULL if we have less than 42 cars
    for (; count < 42; count++) {
        flattened->car_ids[count] = ID_NULL;
        flattened->distances[count] = INFINITY;
        flattened->time_to_collisions[count] = INFINITY;
        flattened->time_headways[count] = INFINITY;
    }
}

CarId nearby_vehicles_flattened_get_car_id(const NearbyVehiclesFlattened* flattened, int index) {
    if (index < 0 || index >= flattened->count) {
        return ID_NULL; // Invalid index
    }
    return flattened->car_ids[index];
}

Meters nearby_vehicles_flattened_get_distance(const NearbyVehiclesFlattened* flattened, int index) {
    if (index < 0 || index >= flattened->count) {
        return INFINITY; // Invalid index
    }
    return flattened->distances[index];
}

Seconds nearby_vehicles_flattened_get_time_to_collision(const NearbyVehiclesFlattened* flattened, int index) {
    if (index < 0 || index >= flattened->count) {
        return INFINITY; // Invalid index
    }
    return flattened->time_to_collisions[index];
}

Seconds nearby_vehicles_flattened_get_time_headway(const NearbyVehiclesFlattened* flattened, int index) {
    if (index < 0 || index >= flattened->count) {
        return INFINITY; // Invalid index
    }
    return flattened->time_headways[index];
}

int nearby_vehicles_flattened_get_count(const NearbyVehiclesFlattened* flattened) {
    return flattened->count;
}
