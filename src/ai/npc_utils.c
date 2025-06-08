#include "ai.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

CarIndictor turn_sample_possible(const SituationalAwareness* situation) {
    if (situation->is_approaching_dead_end) {
        return INDICATOR_NONE;
    }
    CarIndictor possibles[3];
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

CarIndictor lane_change_sample_possible(const SituationalAwareness* situation) {
    CarIndictor possibles[2];
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

bool car_is_lane_change_dangerous(const Car* car, Simulation* sim, const SituationalAwareness* situation, CarIndictor lane_change_indicator) {
    const Lane* lane = situation->lane;
    const Lane* lane_target = situation->lane_target_for_indicator[lane_change_indicator];
    if (lane_target == lane) return false;      // no lane change
    if (lane_target == NULL) return true;       // Will fly off the road!
    double my_progress = car_get_lane_progress(car);
    double my_hypothetical_target_progress;
    if (lane_target == situation->merges_into_lane) {
        my_hypothetical_target_progress =  (lane->merges_into_start + my_progress);
    } else if (lane_target == situation->exit_lane) {
        my_hypothetical_target_progress = my_progress - lane->exit_lane_start;
    } else {
        my_hypothetical_target_progress = my_progress; // since this is just an adjant lane on the same road
    }
    Meters my_hypothetical_target_progress_m = lane_target->length * my_hypothetical_target_progress;
    Meters car_half_length = car_get_length(car) / 2;
    for (int i = 0; i < lane_target->num_cars; i++) {
        const Car* other_car = lane_get_car(lane_target, sim, i);
        if (other_car == car) continue; // skip self
        Meters other_car_progress_m = car_get_lane_progress_meters(other_car);
        
        if (other_car_progress_m > my_hypothetical_target_progress_m) {
            // We are behind the other car
            Meters target_gap = car_get_speed(car) * 3 + from_feet(2);       // 3 second rule + 2 feet buffer
            target_gap += car_half_length + car_get_length(other_car) / 2;   // account for the lengths of both cars
            if (other_car_progress_m - my_hypothetical_target_progress_m < target_gap) {
                // We are too close to the other car
                return true;
            }
        } else {
            // We are ahead of the other car
            Meters target_gap = car_get_speed(other_car) * 3 + from_feet(2);       // 3 second rule + 2 feet buffer
            target_gap += car_half_length + car_get_length(other_car) / 2;   // account for the lengths of both cars
            if (my_hypothetical_target_progress_m - other_car_progress_m < target_gap) {
                // We are too close to the other car
                return true;
            }
        }
    }
    return false;
}


bool car_should_yield_at_intersection(const Car* self, Simulation* sim, const SituationalAwareness* situation, CarIndictor turn_indicator) {
    // TODO: Implement metric logger to show number of collisions on left or right turns.
    // TODO: Fix left turn yielding when the oncoming car stops or breaks in the middle of the intersection. The current ego car assumes the car maintains its speed and does not stop, which is not always true.
    // TODO: Tune the hyperparameters for yield decision making.
    Map* map = sim_get_map(sim); //
    if (!situation->is_approaching_intersection && !situation->is_on_intersection) { //
        return false; // Not at an intersection
    }

    const Intersection* intersection_obj = situation->intersection; //
    if (!intersection_obj) {
        return false;
    }

    Meters car_speed = car_get_speed(self); //
    Meters self_half_length = car_get_length(self) / 2; //
    Meters car_front_bumper_dist_to_stop_line = situation->distance_to_end_of_lane - self_half_length; //

    // 1. Left Turn Yield (e.g., TRAFFIC_LIGHT_GREEN_YIELD, self intends INDICATOR_LEFT)
    if (turn_indicator == INDICATOR_LEFT && //
        (situation->light_for_turn[INDICATOR_LEFT] == TRAFFIC_LIGHT_GREEN_YIELD || //
         (situation->light_for_turn[INDICATOR_LEFT] == TRAFFIC_LIGHT_YELLOW_YIELD && self->preferences.run_yellow_light))) { //
        
        const Meters YIELD_DECISION_PROXIMITY_THRESHOLD = meters(5.0); // Tunable //
        if (car_front_bumper_dist_to_stop_line > YIELD_DECISION_PROXIMITY_THRESHOLD) {
            return false;
        }

        Direction my_dir = situation->lane->direction; //
        const Road* opposite_approach_straight_road = NULL;

        // NOTE: Assuming intersection_get_road_... functions exist and take (Intersection*, Map*)
        // and return const Road*. The original road.h showed direct member access.
        switch (my_dir) {
            case DIRECTION_EAST: //
                opposite_approach_straight_road = intersection_get_road_westbound_from(intersection_obj, map);
                break;
            case DIRECTION_WEST: //
                opposite_approach_straight_road = intersection_get_road_eastbound_from(intersection_obj, map);
                break;
            case DIRECTION_NORTH: //
                opposite_approach_straight_road = intersection_get_road_southbound_from(intersection_obj, map);
                break;
            case DIRECTION_SOUTH: //
                opposite_approach_straight_road = intersection_get_road_northbound_from(intersection_obj, map);
                break;
            default:
                return false; 
        }

        if (!opposite_approach_straight_road) {
            return false; 
        }

        const Road* opposite_approach_road = opposite_approach_straight_road;
        
        for (int i = 0; i < road_get_num_lanes(opposite_approach_road); ++i) {
            // NOTE: Assuming road_get_lane now takes (Road*, Map*, int)
            const Lane* oncoming_lane = road_get_lane(opposite_approach_road, map, i);
            if (!oncoming_lane) continue;

            for (int k = 0; k < lane_get_num_cars(oncoming_lane); ++k) {
                // NOTE: Assuming lane_get_car now takes (Lane*, Simulation*, int)
                const Car* oncoming_car = lane_get_car(oncoming_lane, sim, k);
                if (!oncoming_car) continue;

                Meters oncoming_progress_m = car_get_lane_progress_meters(oncoming_car);
                Meters oncoming_car_length = car_get_length(oncoming_car);
                Meters oncoming_lane_length = lane_get_length(oncoming_lane);
                Meters oncoming_dist_to_their_stop_line = oncoming_lane_length - oncoming_progress_m - (oncoming_car_length / 2.0);
                Meters oncoming_speed = car_get_speed(oncoming_car);

                bool oncoming_is_a_factor = false;

                // Check if oncoming car is already in the intersection's conflict area for our turn
                // (i.e., past its stop line but not too far past the typical conflict zone width).
                // A typical intersection might be 15-20m wide.
                if (oncoming_dist_to_their_stop_line < meters(0) && oncoming_dist_to_their_stop_line > -meters(15.0)) {
                    // Car is in the intersection, 0 to 15m past its stop line.
                    // Its presence is a factor, regardless of its current speed (it might be stopped/braking).
                    oncoming_is_a_factor = true;
                }
                // Else, check if it's approaching and poses a threat based on speed and proximity.
                else if (oncoming_dist_to_their_stop_line >= meters(0) && oncoming_dist_to_their_stop_line < meters(20) && oncoming_speed > meters(1.0)) {
                    oncoming_is_a_factor = true;
                }

                if (oncoming_is_a_factor) {
                    // If the ego car is already (almost) stopped at its line, it must yield.
                    if (car_speed < meters(0.5) && car_front_bumper_dist_to_stop_line < meters(3.0)) {
                        return true; // Yield
                    }

                    // If oncoming car is already IN the intersection (and thus blocking the path)
                    // and our ego car is very close to entering (within YIELD_DECISION_PROXIMITY_THRESHOLD),
                    // the ego car must yield.
                    if (oncoming_dist_to_their_stop_line < meters(0) && oncoming_dist_to_their_stop_line > -meters(15.0)) {
                        // Ego car is already within 5m of its line (YIELD_DECISION_PROXIMITY_THRESHOLD).
                        // If oncoming car is present in the intersection, yield.
                        // printf("Yielding to oncoming car in intersection: %d\n", oncoming_car->id);
                        return true; // Yield
                    }
                    
                    // If oncoming car is APPROACHING (not yet in intersection), and ego is moving, compare times.
                    // This case is now only for when oncoming_dist_to_their_stop_line >= 0.
                    if (oncoming_dist_to_their_stop_line >= meters(0)) {
                        double time_for_oncoming_to_enter_intersection = (oncoming_speed > 0.1) ? (oncoming_dist_to_their_stop_line / oncoming_speed) : 1000.0;
                        double time_for_me_to_enter_conflict_zone = (car_speed > 0.1) ? (car_front_bumper_dist_to_stop_line / car_speed) : 1000.0;
                        
                        if (time_for_oncoming_to_enter_intersection < time_for_me_to_enter_conflict_zone + 5.0) { // 5.0s safety buffer
                            return true; // Yield
                        }
                    }
                }
            }
        }
    }

    // 2. Right Turn on Red Yield / General Yield Sign
    if ((turn_indicator == INDICATOR_RIGHT && situation->light_for_turn[INDICATOR_RIGHT] == TRAFFIC_LIGHT_RED_YIELD) || //
         situation->light_for_turn[turn_indicator] == TRAFFIC_LIGHT_YIELD) { //

        Meters yield_decision_proximity_threshold;
        if (turn_indicator == INDICATOR_RIGHT && situation->light_for_turn[INDICATOR_RIGHT] == TRAFFIC_LIGHT_RED_YIELD) { //
            yield_decision_proximity_threshold = meters(5.0); //
        } else {
            yield_decision_proximity_threshold = meters(10.0); // Tunable //
        }

        if (car_front_bumper_dist_to_stop_line > yield_decision_proximity_threshold) {
            return false;
        }

        Direction my_dir = situation->lane->direction; //
        const Road* conflicting_approach_straight_road = NULL;

        // NOTE: Assuming intersection_get_road_... functions exist
        switch (my_dir) {
            case DIRECTION_NORTH: //
                conflicting_approach_straight_road = intersection_get_road_westbound_from(intersection_obj, map);
                break;
            case DIRECTION_EAST: //
                conflicting_approach_straight_road = intersection_get_road_northbound_from(intersection_obj, map);
                break;
            case DIRECTION_SOUTH: //
                conflicting_approach_straight_road = intersection_get_road_eastbound_from(intersection_obj, map);
                break;
            case DIRECTION_WEST: //
                conflicting_approach_straight_road = intersection_get_road_southbound_from(intersection_obj, map);
                break;
            default:
                return false;
        }

        if (!conflicting_approach_straight_road) {
            return false;
        }

        const Road* conflicting_approach_road = conflicting_approach_straight_road;

        for (int i = 0; i < road_get_num_lanes(conflicting_approach_road); ++i) { //
            // NOTE: Assuming road_get_lane now takes (Road*, Map*, int)
            const Lane* conflicting_lane = road_get_lane(conflicting_approach_road, map, i); //
            if (!conflicting_lane) continue;

            for (int k = 0; k < lane_get_num_cars(conflicting_lane); ++k) { //
                // NOTE: Assuming lane_get_car now takes (Lane*, Simulation*, int)
                const Car* conflicting_car = lane_get_car(conflicting_lane, sim, k); //
                if (!conflicting_car) continue;

                Meters conflicting_car_progress_m = car_get_lane_progress_meters(conflicting_car); //
                Meters conflicting_car_length = car_get_length(conflicting_car); //
                Meters conflicting_lane_length = lane_get_length(conflicting_lane); //
                Meters conflicting_car_dist_to_its_stop_line = conflicting_lane_length - conflicting_car_progress_m - (conflicting_car_length / 2.0);
                Meters conflicting_car_speed = car_get_speed(conflicting_car); //

                if (conflicting_car_dist_to_its_stop_line < meters(15.0) && conflicting_car_speed > meters(1.0)) { //
                    if (car_speed < meters(0.5) && car_front_bumper_dist_to_stop_line < meters(3.0)) { //
                        return true; // Yield
                    }
                    double time_for_conflicting_to_enter_intersection = (conflicting_car_speed > 0.1) ? (conflicting_car_dist_to_its_stop_line / conflicting_car_speed) : 1000.0;
                    const double CONFLICTING_CAR_TIME_THRESHOLD_S = 4.0; // Tunable
                    if (time_for_conflicting_to_enter_intersection < CONFLICTING_CAR_TIME_THRESHOLD_S) {
                        return true; // Yield
                    }
                }
            }
        }
    }
    return false;
}