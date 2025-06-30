#include "ai.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
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


static bool intersection_has_oncoming_vehicle(const Road* approach_road,
                        Map* map,
                        Simulation* sim,
                        const Car* self,
                        double time_threshold_s,
                        Meters slow_speed_cutoff)
{
    for (int i = 0; i < road_get_num_lanes(approach_road); ++i) {
        const Lane* base_lane = road_get_lane(approach_road, map, i);
        if (!base_lane) continue;

        const Lane* lanes_to_check[2] = {
            base_lane,
            lane_get_connection_straight(base_lane, map)   /* may be NULL */
        };

        for (int l = 0; l < 2; ++l) {
            const Lane* lane = lanes_to_check[l];
            if (!lane) continue;

            for (int k = 0; k < lane_get_num_cars(lane); ++k) {
                const Car* c = lane_get_car(lane, sim, k);
                if (!c) continue;

                /* --- distance & time to the “center” of intersection ---- */
                Meters progress = car_get_lane_progress_meters(c);
                Meters car_len  = car_get_length(c);
                Meters lane_len = lane_get_length(lane);
                Meters dist_to_center =
                    lane_len - progress - (car_len / 2.0);   /* ≈ stop-line */

                Meters speed = car_get_speed(c);

                double t_to_center = (speed > slow_speed_cutoff)
                                   ? dist_to_center / speed
                                   : INFINITY;

                bool conflict = lane->is_at_intersection ||
                                (t_to_center <= time_threshold_s);

                /* ---------- DEBUG LOG ------------------------- */
                #ifdef NPCUTILS_DEBUG
                Seconds sim_s  = sim_get_time(sim);
                int hh = (8 + (int)sim_s / 3600) % 24;
                int mm = ((int)sim_s % 3600) / 60;
                int ss =  (int)sim_s % 60;
                printf("%02d:%02d:%02d | ego %d vs car %d "
                       "speed %.2f m/s, t_center %.2f s, conflict %d\n",
                       hh, mm, ss,
                       car_get_id(self), car_get_id(c),
                       (double)speed, t_to_center, conflict);
                #endif
                /* ------------------------------------------------------- */

                if (conflict) return true;
            }
        }
    }
    return false;
}


bool car_should_yield_at_intersection(const Car* self, Simulation* sim, const SituationalAwareness* situation, CarIndictor turn_indicator) {
    Map* map = sim_get_map(sim);
    // Not at an intersection
    if (!situation->is_an_intersection_upcoming && !situation->is_on_intersection) {
        return false;
    }
     // Not at an intersection
    const Intersection* intersection_obj = situation->intersection;
    if (!intersection_obj) {
        return false;
    }

    // Meters car_speed = car_get_speed(self);
    Meters self_half_length = car_get_length(self) / 2;
    Meters car_front_bumper_dist_to_stop_line = situation->distance_to_end_of_lane - self_half_length;

    // 1. Left Turn Yield (e.g., TRAFFIC_LIGHT_GREEN_YIELD, self intends INDICATOR_LEFT)
    if (turn_indicator == INDICATOR_LEFT &&
        (situation->light_for_turn[INDICATOR_LEFT] == TRAFFIC_LIGHT_GREEN_YIELD ||
         (situation->light_for_turn[INDICATOR_LEFT] == TRAFFIC_LIGHT_YELLOW_YIELD && self->preferences.run_yellow_light))) {
        
        const Meters YIELD_DECISION_PROXIMITY_THRESHOLD = meters(5.0);
        if (car_front_bumper_dist_to_stop_line > YIELD_DECISION_PROXIMITY_THRESHOLD) {
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

        // Nothing to worry about ....
        if (!opposite_approach_straight_road) {
            return false; 
        }

        const Road* opposite_approach_road = opposite_approach_straight_road;
        
        // Left Turn on Yield
        // If the oncoming vehicle is in the intersection or the n second rule is violated, then yield
        if (intersection_has_oncoming_vehicle(opposite_approach_road,
                            map, sim, self,
                            4.0, // Time threshold
                            meters(0.25))) {
            return true;   /* yield */
        }
    }

    // 2. Right Turn on Red Yield / General Yield Sign
    if ((turn_indicator == INDICATOR_RIGHT && situation->light_for_turn[INDICATOR_RIGHT] == TRAFFIC_LIGHT_RED_YIELD) ||
        situation->light_for_turn[INDICATOR_RIGHT] == TRAFFIC_LIGHT_YIELD) {

        Meters yield_decision_proximity_threshold = meters(5.0);
        // Keep inching up (Car can get closer to front line)
        if (car_front_bumper_dist_to_stop_line > yield_decision_proximity_threshold) {
            return false;
        }

        Direction my_dir = situation->lane->direction;
        const Road* left_approach_straight_road = NULL;

        // NOTE: Assuming intersection_get_road_... functions exist
        switch (my_dir) {
            case DIRECTION_NORTH:
                left_approach_straight_road = intersection_get_road_westbound_from(intersection_obj, map);
                break;
            case DIRECTION_EAST:
                left_approach_straight_road = intersection_get_road_northbound_from(intersection_obj, map);
                break;
            case DIRECTION_SOUTH:
                left_approach_straight_road = intersection_get_road_eastbound_from(intersection_obj, map);
                break;
            case DIRECTION_WEST:
                left_approach_straight_road = intersection_get_road_southbound_from(intersection_obj, map);
                break;
            default:
                return false;
        }
        
        // Nothing to worry about ....
        if (!left_approach_straight_road) {
            return false;
        }

        const Road* left_approach_road = left_approach_straight_road;

        if (intersection_has_oncoming_vehicle(left_approach_road,
                            map, sim, self,
                            3.0, // Time threshold
                            meters(0.25))) {
            return true;   /* yield */
        }
    }
    return false;
}