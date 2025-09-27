#include "ai.h"
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


bool car_should_yield_at_intersection(const Car* self, Simulation* sim, const SituationalAwareness* situation, CarIndicator turn_indicator) {
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
