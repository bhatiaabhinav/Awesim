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

bool car_is_lane_change_dangerous(const Car* car, const SituationalAwareness* situation, CarIndictor lane_change_indicator) {
    const Lane* lane = car->lane;
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
        const Car* other_car = lane_target->cars[i];
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