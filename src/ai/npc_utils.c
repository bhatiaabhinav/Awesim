#include "ai.h"
#include "logging.h"
#include "sim.h"
#include <stdlib.h>
#include <stdio.h>
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

static Seconds compute_time_headway(Meters distance, MetersPerSecond speed) {
    if (speed <= STOP_SPEED_THRESHOLD) {
        return INFINITY; // effectively infinite time headway if not moving
    }
    return distance / speed;
}

bool perceive_lead_vehicle(const Car* self, Simulation* sim, const SituationalAwareness* situation, Meters *out_distance, MetersPerSecond* out_speed, MetersPerSecondSquared* out_acceleration) {
    return car_noisy_perceive_other(self, situation->nearby_vehicles.lead, sim, NULL, NULL, out_speed, out_acceleration, out_distance, NULL, NULL, NULL, NULL, false, 0.0);
}

bool car_is_lane_change_dangerous(Car* car, Simulation* sim, const SituationalAwareness* situation, CarIndicator lane_change_indicator, Seconds time_headway_threshold, Meters buffer, double dropout_probability_multiplier) {
    const Lane* lane = situation->lane;
    const Lane* lane_target = situation->lane_target_for_indicator[lane_change_indicator];
    if (lane_target == lane) return false;      // no lane change
    if (lane_target == NULL) return true;       // Will fly off the road!

    const Car* car_ahead = situation->nearby_vehicles.ahead[lane_change_indicator];
    const Car* car_behind = situation->nearby_vehicles.behind[lane_change_indicator];
    const Car* car_colliding = situation->nearby_vehicles.colliding[lane_change_indicator];

    if (car_colliding) {
        bool noticed = car_noisy_perceive_other(car, car_colliding, sim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, true, dropout_probability_multiplier);   // may go unnoticed if in blind spot
        if (noticed) {
            return true; // there is a car in the target lane that is colliding with us
        }
    }
    if (car_ahead) {
        Meters hypothetical_position = calculate_hypothetical_position_on_lane_change(car, lane, lane_target, sim_get_map(sim));
        Meters other_car_position;
        MetersPerSecond other_car_speed;
        bool noticed = car_noisy_perceive_other(car, car_ahead, sim, NULL, &other_car_position, &other_car_speed, NULL, NULL, NULL, NULL, NULL, NULL, false, dropout_probability_multiplier);
        if (noticed) {
            Meters distance_to_other_car = other_car_position - hypothetical_position - car_get_length(car_ahead) / 2 - car_get_length(car) / 2;
            Seconds thw = compute_time_headway(fmax(distance_to_other_car, 0.0), situation->speed);
            if (thw < time_headway_threshold || distance_to_other_car < buffer) {
                return true;
            }
        }
    }
    if (car_behind) {
        Meters hypothetical_position = calculate_hypothetical_position_on_lane_change(car, lane, lane_target, sim_get_map(sim));
        Meters other_car_position;
        MetersPerSecond other_car_speed;
        bool noticed = car_noisy_perceive_other(car, car_behind, sim, NULL, &other_car_position, &other_car_speed, NULL, NULL, NULL, NULL, NULL, NULL, true, dropout_probability_multiplier);
        if (noticed) {
            Meters distance_to_other_car = hypothetical_position - other_car_position - car_get_length(car_behind) / 2 - car_get_length(car) / 2;
            Seconds thw = compute_time_headway(fmax(distance_to_other_car, 0.0), fmax(other_car_speed, 0.0));
            // The car behind will be too close to us
            if (thw < time_headway_threshold || distance_to_other_car < buffer) {
                return true;
            }
        }
    }
    if (situation->lane_progress_m - car_get_length(car) / 2 < meters(30.0)) {
        // We just entered the lane, do not change lanes immediately.
        return true;
    }

    return false;
}


bool car_has_something_to_yield_to_at_intersection(const Car* self, Simulation* sim, const SituationalAwareness* situation, CarIndicator turn_indicator, Seconds yield_time_headway_threshold, double dropout_probability_multiplier) {
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
            // Check if we perceive any car in the lane
            bool perceived_any = false;
            for (int j = 0; j < lane_get_num_cars(lane); j++) {
                // Determine if we see this car
                double net_dropout_probability = sim->perception_noise_dropout_probability * dropout_probability_multiplier;
                bool dropped = net_dropout_probability > 1e-9 && rand_0_to_1() < net_dropout_probability;
                if (!dropped) {    
                    perceived_any = true;
                    break;
                }
            }
            if (perceived_any) {
                return true;    // have to yield
            }
        }
    }

    // Go through the leading cars on incoming lanes and if any of them are less than yield_time_headway_threshold seconds away from their lane end, then we have to yield.
    for (int i = 0; i < num_incoming_lanes_to_worry_about; i++) {
        Lane* lane = incoming_lanes_to_worry_about[i];
        if (!lane || lane_get_num_cars(lane) == 0) continue;
        Car* c = lane_get_car(lane, sim, 0); // get the leading car
        Meters car_progress_m;
        MetersPerSecond car_speed;
        bool noticed = car_noisy_perceive_other(self, c, sim, NULL, &car_progress_m, &car_speed, NULL, NULL, NULL, NULL, NULL, NULL, false, dropout_probability_multiplier);
        if (!noticed) {
            continue;   // did not perceive this car
        }

        // deadlock avoidance: If we are turning left, we don't need to yield to opposing left turners as our paths don't cross (usually)
        if (!intersection->is_T_junction && turn_indicator == INDICATOR_LEFT && c->indicator_turn == INDICATOR_LEFT && lane->direction == direction_opposite(my_dir)) {
            continue;
        }
        Meters lane_length = lane_get_length(lane);
        Meters distance_to_stop_line = lane_length - car_progress_m - car_get_length(c) / 2 - STOP_LINE_BUFFER_METERS;

        if (distance_to_stop_line < meters(1)) {
            return true;    // have to yield if already too close to the end, irrespective of speed
        }

        Seconds time_to_end_s = compute_time_headway(distance_to_stop_line, fmax(car_speed, 0.0));
        if (time_to_end_s < yield_time_headway_threshold) {
            return true;    // have to yield if coming in fast or has reached the stop line even if slow
        }
    }
    return false;
}


bool defensive_yield_despite_right_of_way(const Car* self, Simulation* sim, const SituationalAwareness* situation, CarIndicator turn_indicator, Seconds yield_time_headway_threshold, double dropout_probability_multiplier) {
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

    Lane* incoming_lanes_to_worry_about[4 * MAX_NUM_LANES_PER_ROAD] = {NULL};           // max 4 roads to worry about, each with max lanes.
    int num_incoming_lanes_to_worry_about = 0;
    Lane* intersection_lanes_to_worry_about[MAX_NUM_LANES_PER_INTERSECTION] = {NULL};    // lanes that must not have cars on them
    int num_intersection_lanes_to_worry_about = 0;

    if (intersection->is_T_junction) {
        bool i_am_merging = (intersection->is_T_junction_north_south && (my_dir == DIRECTION_EAST || my_dir == DIRECTION_WEST)) ||
                            (!intersection->is_T_junction_north_south && (my_dir == DIRECTION_NORTH || my_dir == DIRECTION_SOUTH));
        if (i_am_merging) {
            // If turning right, worry about the rightmost lane on the road coming from the left
            if (turn_indicator == INDICATOR_RIGHT) {
                Lane* rightmost_lane_from_left = road_get_rightmost_lane(road_from_left, map);
                Car* c = lane_get_num_cars(rightmost_lane_from_left) > 0 ? lane_get_car(rightmost_lane_from_left, sim, 0) : NULL;
                if (c && c->indicator_turn == INDICATOR_NONE) {
                    incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = rightmost_lane_from_left;
                }
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(rightmost_lane_from_left, map);
            } else if (turn_indicator == INDICATOR_LEFT) {
                // If turning left, worry about the leftmost lane on the road coming from the right and all lanes on the road coming from left
                Lane* leftmost_lane_from_right = road_get_leftmost_lane(road_from_right, map);
                Car* c = lane_get_num_cars(leftmost_lane_from_right) > 0 ? lane_get_car(leftmost_lane_from_right, sim, 0) : NULL;
                if (c) {
                    incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = leftmost_lane_from_right;
                }
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(leftmost_lane_from_right, map);
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(leftmost_lane_from_right, map);

                for (int i = 0; i < road_get_num_lanes(road_from_left); i++) {
                    Lane* lane_i = road_get_lane(road_from_left, map, i);
                    Car* c = lane_get_num_cars(lane_i) > 0 ? lane_get_car(lane_i, sim, 0) : NULL;
                    if (c && c->indicator_turn == INDICATOR_NONE) {
                        incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = lane_i;
                    }
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(lane_i, map);
                }
            } else {
                // Not possible to go straight in a T-junction
            }
        } else {
            bool is_T_exit_to_our_left = lane_get_connection_left(road_get_leftmost_lane(situation->road, map), map) != NULL;
            bool is_T_exit_to_our_right = lane_get_connection_right(road_get_rightmost_lane(situation->road, map), map) != NULL;
            if (turn_indicator == INDICATOR_RIGHT) {
                bool is_exit_single_lane = road_get_num_lanes(road_from_left) == 1; // assuming both road_from_left and road_to_right have same number of lanes
                // if exiting into a single lane road, worry about exiting left from the opposite direction (even though they are supposed to yield to us, they might not)
                if (is_exit_single_lane) { // assuming both road_from_left and road_to_right have same number of lanes
                    Lane* opposite_leftmost_lane = road_get_leftmost_lane(road_from_opposite, map);
                    Car* c = lane_get_num_cars(opposite_leftmost_lane) > 0 ? lane_get_car(opposite_leftmost_lane, sim, 0) : NULL;
                    if (c && c->indicator_turn == INDICATOR_LEFT) {
                        incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = opposite_leftmost_lane;
                    }
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(opposite_leftmost_lane, map);
                }
            } else if (turn_indicator == INDICATOR_LEFT) {
                bool is_exit_single_lane = road_get_num_lanes(road_from_right) == 1; // assuming both road_from_right and road_to_left have same number of lanes
                // exit while yeilding to oncoming traffic. if we are exiting on to a single lane road, we need to worry about the oncoming traffic turning into that lane as well
                for (int i = 0; i < road_get_num_lanes(road_from_opposite); i++) {
                    Lane* lane_i = road_get_lane(road_from_opposite, map, i);
                    Car* c = lane_get_num_cars(lane_i) > 0 ? lane_get_car(lane_i, sim, 0) : NULL;
                    if (c && (c->indicator_turn == INDICATOR_NONE || (is_exit_single_lane && road_get_rightmost_lane(road_from_opposite, map) == lane_i && c->indicator_turn == INDICATOR_RIGHT))) {
                        incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = lane_i;
                    }
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(lane_i, map);
                    if (is_exit_single_lane && road_get_rightmost_lane(road_from_opposite, map) == lane_i ) {
                        intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_right(lane_i, map);
                    }
                }
                // also worry about the traffic from the left road merging into our road, even though they are supposed to yield to us, they might not
                Lane* leftmost_lane_from_left = road_get_leftmost_lane(road_from_left, map);
                Car* c = lane_get_num_cars(leftmost_lane_from_left) > 0 ? lane_get_car(leftmost_lane_from_left, sim, 0) : NULL;
                if (c && c->indicator_turn == INDICATOR_LEFT) {
                    incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = leftmost_lane_from_left;
                }
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(leftmost_lane_from_left, map);
            } else {
                if (is_T_exit_to_our_left) {
                    if (situation->lane == road_get_leftmost_lane(situation->road, map)) { // leftmost lane, worry about traffic from the left road merging into us
                        Lane* leftmost_lane_from_left = road_get_leftmost_lane(road_from_left, map);
                        Car* c = lane_get_num_cars(leftmost_lane_from_left) > 0 ? lane_get_car(leftmost_lane_from_left, sim, 0) : NULL;
                        if (c && c->indicator_turn == INDICATOR_LEFT) {
                            incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = leftmost_lane_from_left;
                        }
                        intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(leftmost_lane_from_left, map);
                    }
                }
                if (is_T_exit_to_our_right) {
                    // worry about oncoming traffic going to their left (our right)
                    Lane* opposite_road_leftmost_lane = road_get_leftmost_lane(road_from_opposite, map);
                    Car* c = lane_get_num_cars(opposite_road_leftmost_lane) > 0 ? lane_get_car(opposite_road_leftmost_lane, sim, 0) : NULL;
                    if (c && c->indicator_turn == INDICATOR_LEFT) {
                        incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = opposite_road_leftmost_lane;
                    }
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(opposite_road_leftmost_lane, map);

                    // worry about traffic from right merging left into the oncoming traffic
                    Lane* right_road_leftmost_lane = road_get_leftmost_lane(road_from_right, map);
                    c = lane_get_num_cars(right_road_leftmost_lane) > 0 ? lane_get_car(right_road_leftmost_lane, sim, 0) : NULL;
                    if (c && c->indicator_turn == INDICATOR_LEFT) {
                        incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = right_road_leftmost_lane;
                    }
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(right_road_leftmost_lane, map);

                    // if we are on rightmost lane, we also need to worry about traffic from the right merging into our lane
                    if (situation->lane == road_get_rightmost_lane(situation->road, map)) {
                        Lane* rightmost_lane_from_right = road_get_rightmost_lane(road_from_right, map);
                        c = lane_get_num_cars(rightmost_lane_from_right) > 0 ? lane_get_car(rightmost_lane_from_right, sim, 0) : NULL;
                        if (c && c->indicator_turn == INDICATOR_RIGHT) {
                            incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = rightmost_lane_from_right;
                        }
                        intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_right(rightmost_lane_from_right, map);
                    }
                }
            }
        }
    } else {  // any 4 way intersection
        bool is_going_into_single_lane_road = (turn_indicator == INDICATOR_RIGHT && road_get_num_lanes(road_from_left) == 1) ||
                                     (turn_indicator == INDICATOR_LEFT && road_get_num_lanes(road_from_right) == 1) || (
                                         turn_indicator == INDICATOR_NONE && road_get_num_lanes(situation->road) == 1);
        if (turn_indicator == INDICATOR_RIGHT) {
            // If turning right, worry about the jumpers on the rightmost lane on the road coming from the left
            Lane* rightmost_lane_from_left = road_get_rightmost_lane(road_from_left, map);
            Car* c = lane_get_num_cars(rightmost_lane_from_left) > 0 ? lane_get_car(rightmost_lane_from_left, sim, 0) : NULL;
            if (c && c->indicator_turn == INDICATOR_NONE) {
                incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = rightmost_lane_from_left;
            }
            intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(rightmost_lane_from_left, map);
            // if turning into single lane road, also worry about jumpers from the opposite direction turning left into that lane
            if (is_going_into_single_lane_road) {
                Lane* opposite_leftmost_lane = road_get_leftmost_lane(road_from_opposite, map);
                c = lane_get_num_cars(opposite_leftmost_lane) > 0 ? lane_get_car(opposite_leftmost_lane, sim, 0) : NULL;
                if (c && c->indicator_turn == INDICATOR_LEFT) {
                    incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = opposite_leftmost_lane;
                }
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(opposite_leftmost_lane, map);
            }
        } else if (turn_indicator == INDICATOR_LEFT) {
            // If turning left, worry about all the lanes on the road coming from the opposite direction
            for (int i = 0; i < road_get_num_lanes(road_from_opposite); i++) {
                Lane* lane_i = road_get_lane(road_from_opposite, map, i);
                Car* c = lane_get_num_cars(lane_i) > 0 ? lane_get_car(lane_i, sim, 0) : NULL;
                if (c && (c->indicator_turn == INDICATOR_NONE || (is_going_into_single_lane_road && road_get_rightmost_lane(road_from_opposite, map) == lane_i && c->indicator_turn == INDICATOR_RIGHT))) {
                    incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = lane_i;
                }
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(lane_i, map);
                if (is_going_into_single_lane_road && road_get_rightmost_lane(road_from_opposite, map) == lane_i ) {
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_right(lane_i, map);
                }
            }
            // All left-to-right cross traffic:
            for (int i = 0; i < road_get_num_lanes(road_from_left); i++) {
                Lane* lane_i = road_get_lane(road_from_left, map, i);
                Car* c = lane_get_num_cars(lane_i) > 0 ? lane_get_car(lane_i, sim, 0) : NULL;
                if (c && c->indicator_turn == INDICATOR_LEFT) {
                    incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = lane_i;
                }
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(lane_i, map);
            }
            // leftmost lane of the road coming from right:
            Lane* leftmost_lane_from_right = road_get_leftmost_lane(road_from_right, map);
            Car* c = lane_get_num_cars(leftmost_lane_from_right) > 0 ? lane_get_car(leftmost_lane_from_right, sim, 0) : NULL;
            if (c && (c->indicator_turn == INDICATOR_LEFT || c->indicator_turn == INDICATOR_NONE)) {
                incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = leftmost_lane_from_right;
            }
            intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(leftmost_lane_from_right, map);
            intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(leftmost_lane_from_right, map);
        } else {
            // opposite road traffic turning left:
            Lane* opposite_leftmost_lane = road_get_leftmost_lane(road_from_opposite, map);
            Car* c = lane_get_num_cars(opposite_leftmost_lane) > 0 ? lane_get_car(opposite_leftmost_lane, sim, 0) : NULL;
            if (c && c->indicator_turn == INDICATOR_LEFT) {
                incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = opposite_leftmost_lane;
            }

            // all left to right trafic:
            bool we_in_leftmost_lane = situation->lane == road_get_leftmost_lane(situation->road, map);
            for (int i = 0; i < road_get_num_lanes(road_from_left); i++) {
                Lane* lane_i = road_get_lane(road_from_left, map, i);
                bool i_is_leftmost = lane_i == road_get_leftmost_lane(road_from_left, map);
                Car* c = lane_get_num_cars(lane_i) > 0 ? lane_get_car(lane_i, sim, 0) : NULL;
                if (c && (c->indicator_turn == INDICATOR_NONE || (we_in_leftmost_lane && i_is_leftmost && c->indicator_turn == INDICATOR_LEFT))) {
                    incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = lane_i;
                }
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(lane_i, map);
                if (we_in_leftmost_lane && i_is_leftmost) {
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(lane_i, map);
                }
            }
            
            // all right to left traffic:
            bool we_in_rightmost_lane = situation->lane == road_get_rightmost_lane(situation->road, map);
            for (int i = 0; i < road_get_num_lanes(road_from_right); i++) {
                Lane* lane_i = road_get_lane(road_from_right, map, i);
                bool i_is_leftmost = lane_i == road_get_leftmost_lane(road_from_right, map);
                bool i_is_rightmost = lane_i == road_get_rightmost_lane(road_from_right, map);
                Car* c = lane_get_num_cars(lane_i) > 0 ? lane_get_car(lane_i, sim, 0) : NULL;
                // always worry about straight and left turners, and right turners if we are in the rightmost lane and they are in the rightmost lane as well
                if (c && (c->indicator_turn == INDICATOR_NONE || c->indicator_turn == INDICATOR_LEFT || (we_in_rightmost_lane && i_is_rightmost && c->indicator_turn == INDICATOR_RIGHT))) {
                    incoming_lanes_to_worry_about[num_incoming_lanes_to_worry_about++] = lane_i;
                }
                intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_straight(lane_i, map);
                if (i_is_leftmost) {
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_left(lane_i, map);
                }
                if (i_is_rightmost && we_in_rightmost_lane) {
                    intersection_lanes_to_worry_about[num_intersection_lanes_to_worry_about++] = lane_get_connection_right(lane_i, map);
                }
            }
        }
    }

    // Go through the intersection lanes to worry about and if any of them have cars, we should be defensive and yield to them even if we technically have the right of way, as they might be jumpers who are not yielding to us or running a red light.
    for (int i = 0; i < num_intersection_lanes_to_worry_about; i++) {
        Lane* lane = intersection_lanes_to_worry_about[i];
        if (lane && lane_get_num_cars(lane) > 0) {
            // Check if we perceive any car in the lane
            bool perceived_any = false;
            for (int j = 0; j < lane_get_num_cars(lane); j++) {
                // Determine if we see this car
                double net_dropout_probability = sim->perception_noise_dropout_probability * dropout_probability_multiplier;
                bool dropped = net_dropout_probability > 1e-9 && rand_0_to_1() < net_dropout_probability;
                if (!dropped) {    
                    perceived_any = true;
                    break;
                }
            }
            if (perceived_any) {
                return true;    // should yield
            }
        }
    }

    // Go through the leading cars on incoming lanes and if any of them are less than yield_time_headway_threshold seconds away from their lane end, or creeping beyond the lane end, just let them go.
    for (int i = 0; i < num_incoming_lanes_to_worry_about; i++) {
        Lane* lane = incoming_lanes_to_worry_about[i];
        if (!lane || lane_get_num_cars(lane) == 0) continue;
        Car* c = lane_get_car(lane, sim, 0); // get the leading car
        Meters car_progress_m;
        MetersPerSecond car_speed;
        MetersPerSecondSquared car_acceleration;
        bool noticed = car_noisy_perceive_other(self, c, sim, NULL, &car_progress_m, &car_speed, &car_acceleration, NULL, NULL, NULL, NULL, NULL, false, dropout_probability_multiplier);
        if (!noticed) {
            continue;   // did not perceive this car
        }

        // If we are turning left, we don't need to worry about opposing left turners as our paths don't cross
        if (!intersection->is_T_junction && turn_indicator == INDICATOR_LEFT && c->indicator_turn == INDICATOR_LEFT && lane->direction == direction_opposite(my_dir)) {
            continue;
        }
        // if we are turning right, and cross traffic is not going straight or left, we don't need to worry about them as our paths don't cross
        if (turn_indicator == INDICATOR_RIGHT && lane->direction == direction_perp_cw(my_dir)) {
            if (c->indicator_turn != INDICATOR_NONE && c->indicator_turn != INDICATOR_LEFT) {
                continue;
            }
        }


        Meters lane_length = lane_get_length(lane);
        Meters distance_to_lane_end = lane_length - car_progress_m - car_get_length(c) / 2;

        if (distance_to_lane_end > 0) {
            if (car_speed > 0) {
                if (car_acceleration < 0) {
                    // they are braking, but they do seem to brake hard enough?
                    Meters stopping_distance = (car_speed * car_speed) / (2 * (-car_acceleration));
                    if (stopping_distance > distance_to_lane_end) {
                        return true;    // they won't be able to stop in time, let them go
                    }
                } else {
                    // Not braking. Linearly project
                    Seconds time_to_lane_end = distance_to_lane_end / car_speed;
                    if (time_to_lane_end < yield_time_headway_threshold) {
                        return true;    // they will reach the end of the lane within our yield time headway threshold, let them go
                    }
                }
            } else {
                // they are stopped or going backwards, so, not a threat
            }
        } else {
            if (car_speed > CREEP_SPEED || car_acceleration > 0) {
                return true;    // they are already beyond the lane end and still moving, or beginning to move, let them go
            }
        }
    }
    return false;
}

void nearby_vehicles_flatten(NearbyVehicles* nearby_vehicles, NearbyVehiclesFlattened* flattened) {
    int count = 0;

    // Add lead vehicle
    const Car* c = nearby_vehicles->lead;
    CarId id = c ? c->id : ID_NULL;
    flattened->car_ids[count++] = id;

    // Add following vehicle
    c = nearby_vehicles->following;
    id = c ? c->id : ID_NULL;
    flattened->car_ids[count++] = id;

    // Add vehicles from the front lane
    for (int i = 0; i < 3; i++) {
        c = nearby_vehicles->ahead[i];
        id = c ? c->id : ID_NULL;
        flattened->car_ids[count++] = id;
    }

    // Add vehicles from the colliding lane
    for (int i = 0; i < 3; i++) {
        c = nearby_vehicles->colliding[i];
        id = c ? c->id : ID_NULL;
        flattened->car_ids[count++] = id;
    }

    // Add vehicles from the rear lane
    for (int i = 0; i < 3; i++) {
        c = nearby_vehicles->behind[i];
        id = c ? c->id : ID_NULL;
        flattened->car_ids[count++] = id;
    }
    
    flattened->count = count;
}

CarId nearby_vehicles_flattened_get_car_id(const NearbyVehiclesFlattened* flattened, int index) {
    if (index < 0 || index >= flattened->count) {
        return ID_NULL; // Invalid index
    }
    return flattened->car_ids[index];
}

int nearby_vehicles_flattened_get_count(const NearbyVehiclesFlattened* flattened) {
    return flattened->count;
}
