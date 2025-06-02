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
    Map* map = sim_get_map(sim);
    if (!situation->is_approaching_intersection && !situation->is_on_intersection) {
        return false; // Not at an intersection
    }

    const Intersection* intersection_obj = situation->intersection;
    if (!intersection_obj) {
        return false;
    }

    Meters car_speed = car_get_speed(self);
    Meters self_half_length = car_get_length(self) / 2;
    Meters car_front_bumper_dist_to_stop_line = situation->distance_to_end_of_lane - self_half_length;

    // 1. Left Turn Yield (e.g., TRAFFIC_LIGHT_GREEN_YIELD, self intends INDICATOR_LEFT)
    if (turn_indicator == INDICATOR_LEFT &&
        (situation->light_for_turn[INDICATOR_LEFT] == TRAFFIC_LIGHT_GREEN_YIELD ||
         (situation->light_for_turn[INDICATOR_LEFT] == TRAFFIC_LIGHT_YELLOW_YIELD && self->preferences.run_yellow_light))) {
        
        // This check ensures the car doesn't yield when it's still far down the road from the intersection.
        const Meters YIELD_DECISION_PROXIMITY_THRESHOLD = meters(5.0); // Tunable
        if (car_front_bumper_dist_to_stop_line > YIELD_DECISION_PROXIMITY_THRESHOLD) {
            // Too far from the stop line to make a specific yield decision against oncoming traffic yet.
            // The main driving logic in npc_car_make_decisions should handle general approach speed.
            return false;
        }

        Direction my_dir = situation->lane->direction;
        const Road* opposite_approach_straight_road = NULL;

        switch (my_dir) {
            case DIRECTION_EAST:
                opposite_approach_straight_road = intersection_get_road_westbound_from(intersection_obj, map);
                break;
            case DIRECTION_WEST:
                opposite_approach_straight_road = intersection_get_road_eastbound_from(intersection_obj, map);
                break;
            case DIRECTION_NORTH:
                opposite_approach_straight_road = intersection_get_road_southbound_from(intersection_obj, map);
                break;
            case DIRECTION_SOUTH:
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
            const Lane* oncoming_lane = road_get_lane(opposite_approach_road, map, i);
            if (!oncoming_lane) continue;

            for (int k = 0; k < lane_get_num_cars(oncoming_lane); ++k) {
                const Car* oncoming_car = lane_get_car(oncoming_lane, sim, k);
                if (!oncoming_car) continue;

                Meters oncoming_progress_m = car_get_lane_progress_meters(oncoming_car);
                Meters oncoming_car_length = car_get_length(oncoming_car);
                Meters oncoming_lane_length = lane_get_length(oncoming_lane);
                Meters oncoming_dist_to_their_stop_line = oncoming_lane_length - oncoming_progress_m - (oncoming_car_length / 2.0);
                Meters oncoming_speed = car_get_speed(oncoming_car);

                // Check if the oncoming car is relevant (close to its stop line and moving)
                if (oncoming_dist_to_their_stop_line < meters(20) && oncoming_speed > meters(1.0)) { // Tunable: meters(20)
                    
                    // If the ego car is already (almost) stopped at its line, it should yield if any relevant oncoming traffic exists.
                    if (car_speed < meters(0.5) && car_front_bumper_dist_to_stop_line < meters(3.0)) { // Essentially stopped at line
                        return true; // Yield
                    }

                    // If ego car is still moving (but within YIELD_DECISION_PROXIMITY_THRESHOLD), compare times.
                    double time_for_oncoming_to_enter_intersection = (oncoming_speed > 0.1) ? (oncoming_dist_to_their_stop_line / oncoming_speed) : 1000.0;
                    double time_for_me_to_enter_conflict_zone = (car_speed > 0.1) ? (car_front_bumper_dist_to_stop_line / car_speed) : 1000.0;
                    
                    if (time_for_oncoming_to_enter_intersection < time_for_me_to_enter_conflict_zone + 5.0) { // 5.0s safety buffer
                        return true; // Yield
                    }
                }
            }
        }
    }

    // 2. Right Turn on Red Yield / General Yield Sign
    // This handles TRAFFIC_LIGHT_RED_YIELD specifically for right turns,
    // and TRAFFIC_LIGHT_YIELD for any turn_indicator (though the conflict logic below is tailored for cross-traffic from left).
    if ((turn_indicator == INDICATOR_RIGHT && situation->light_for_turn[INDICATOR_RIGHT] == TRAFFIC_LIGHT_RED_YIELD) ||
         situation->light_for_turn[turn_indicator] == TRAFFIC_LIGHT_YIELD) {

        Meters yield_decision_proximity_threshold;
        if (turn_indicator == INDICATOR_RIGHT && situation->light_for_turn[INDICATOR_RIGHT] == TRAFFIC_LIGHT_RED_YIELD) {
            // For Right Turn on Red, car should be very close, typically having already stopped or preparing to.
            yield_decision_proximity_threshold = meters(5.0);
        } else {
            // For a general TRAFFIC_LIGHT_YIELD, allow a slightly larger distance,
            yield_decision_proximity_threshold = meters(10.0); // Tunable
        }

        if (car_front_bumper_dist_to_stop_line > yield_decision_proximity_threshold) {
            // Too far from the stop line to make this specific yield decision yet.
            return false;
        }

        Direction my_dir = situation->lane->direction; //
        const Road* conflicting_approach_straight_road = NULL;

        // Determine the primary conflicting approach road (traffic from the ego car's left).
        switch (my_dir) {
            case DIRECTION_NORTH: // Ego car going North, turning right (East) OR yielding. Conflict from West.
                conflicting_approach_straight_road = intersection_get_road_westbound_from(intersection_obj, map); //
                break;
            case DIRECTION_EAST:  // Ego car going East, turning right (South) OR yielding. Conflict from North.
                conflicting_approach_straight_road = intersection_get_road_northbound_from(intersection_obj, map); //
                break;
            case DIRECTION_SOUTH: // Ego car going South, turning right (West) OR yielding. Conflict from East.
                conflicting_approach_straight_road = intersection_get_road_eastbound_from(intersection_obj, map); //
                break;
            case DIRECTION_WEST:  // Ego car going West, turning right (North) OR yielding. Conflict from South.
                conflicting_approach_straight_road = intersection_get_road_southbound_from(intersection_obj, map); //
                break;
            default:
                // Non-cardinal direction for the current lane, or situation not easily mapped
                return false; // Cannot determine conflict reliably.
        }

        if (!conflicting_approach_straight_road) {
            // This might occur at a T-junction where there's no road to the left,
            // or if the intersection definition is incomplete.
            // printf("DEBUG Car %p: Right/Yield Turn - Conflicting approach road is NULL for my_dir %d.\n", (void*)self, my_dir);
            return false; // Assume clear if no conflicting path is defined.
        }

        const Road* conflicting_approach_road = conflicting_approach_straight_road;

        for (int i = 0; i < road_get_num_lanes(conflicting_approach_road); ++i) { //
            const Lane* conflicting_lane = road_get_lane(conflicting_approach_road, map, i); //
            if (!conflicting_lane) continue;

            for (int k = 0; k < lane_get_num_cars(conflicting_lane); ++k) { //
                const Car* conflicting_car = lane_get_car(conflicting_lane, sim, k); //
                if (!conflicting_car) continue;

                Meters conflicting_car_progress_m = car_get_lane_progress_meters(conflicting_car); //
                Meters conflicting_car_length = car_get_length(conflicting_car); //
                Meters conflicting_lane_length = lane_get_length(conflicting_lane); //
                Meters conflicting_car_dist_to_its_stop_line = conflicting_lane_length - conflicting_car_progress_m - (conflicting_car_length / 2.0);
                Meters conflicting_car_speed = car_get_speed(conflicting_car); //

                // Heuristic: If a conflicting car (from the left) is close to its intersection entry and moving.
                // Tunable: meters(25.0) gives them a bit more space/time than direct oncoming for left turns.
                if (conflicting_car_dist_to_its_stop_line < meters(15.0) && conflicting_car_speed > meters(1.0)) {
                    
                    // If ego car is already (almost) stopped at its line (expected for Right Turn on Red,
                    // or after deciding to stop for a Yield sign).
                    if (car_speed < meters(0.5) && car_front_bumper_dist_to_stop_line < meters(3.0)) { // Essentially stopped at line
                        // printf("DEBUG Car %p: Yielding (stopped at line) to conflicting cross-traffic car %p\n", (void*)self, (void*)conflicting_car);
                        return true; // Yield
                    }

                    // If ego car is still moving (but within yield_decision_proximity_threshold),
                    // be cautious. For yields/right-on-red, it's generally not about "beating" the other car
                    // but ensuring they pass safely if they are close.
                    double time_for_conflicting_to_enter_intersection = (conflicting_car_speed > 0.1) ? (conflicting_car_dist_to_its_stop_line / conflicting_car_speed) : 1000.0;

                    // If conflicting car is very close in time (e.g., less than ~3 seconds away),
                    // it's safer to yield.
                    const double CONFLICTING_CAR_TIME_THRESHOLD_S = 3.0; // Tunable
                    if (time_for_conflicting_to_enter_intersection < CONFLICTING_CAR_TIME_THRESHOLD_S) {
                        // printf("DEBUG Car %p: Yielding (moving) to conflicting cross-traffic car %p (time_to_their_entry %f s)\n", (void*)self, (void*)conflicting_car, time_for_conflicting_to_enter_intersection);
                        return true; // Yield
                    }
                }
            }
        }
    }

    return false;
}