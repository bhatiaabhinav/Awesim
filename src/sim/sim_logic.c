#include "sim.h"
#include "ai.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

void car_handle_lane_change(Car* car, Lane* new_lane) {
    Lane* current_lane = (Lane*)car_get_lane(car);
    if (current_lane == new_lane) {
        return; // No change needed
    }
    assert(current_lane != NULL && "Car must be on a lane to change lanes");
    assert(new_lane != NULL && "New lane must not be NULL");
    assert(new_lane->direction != direction_opposite(current_lane->direction) && "Cannot change to a lane in the opposite direction");
    lane_remove_car(current_lane, car);
    lane_add_car(new_lane, car);
    car_set_lane(car, new_lane);
}

void simulate(Simulation* self, Seconds time_period) {
    Seconds t0 = self->time;
    Seconds dt = self->dt;
    // printf("time period %f\n", time_period);
    while (self->time < t0 + time_period) {

        for (int i = 0; i < self->map->num_intersections; i++) {
            const Intersection* intersection = self->map->intersections[i];
            intersection_update((Intersection*)intersection, dt);
        }


        for(int car_id=0; car_id < self->num_cars; car_id++) {
            // bool is_NPC = car_id > 0;
            Car* car = self->cars[car_id];
            npc_car_make_decisions(car);
            const Lane* lane = car_get_lane(car);
            double progress = car_get_lane_progress(car);
            Meters s = car_get_lane_progress_meters(car);
            CarIndictor lane_change_intent = car_get_indicator_lane(car);
            if(lane_change_intent == INDICATOR_LEFT && lane->adjacent_left && lane->adjacent_left->direction == lane->direction) {
                // TODO: ignore lane change if this is an NPC if a collision is possible with another vehicle
                lane = lane->adjacent_left;
                lane_change_intent = INDICATOR_NONE;   // Important: To mark lane change intent already executed, so that merge is ignored if a merge is available on the new lane.
            } else if (lane_change_intent == INDICATOR_RIGHT && lane->adjacent_right) {
                assert(lane->adjacent_right->direction == lane->direction && "How did this happen? Map error!");
                // TODO: ignore lane change if this is an NPC if a collision is possible with another vehicle
                lane = lane->adjacent_right;
                lane_change_intent = INDICATOR_NONE;
            } else {
                lane = lane;
            }

            MetersPerSecondSquared a = car_get_acceleration(car);
            MetersPerSecond v = car_get_speed(car); 

            // ! for debugging
            // a = 0;
            // v = lane->speed_limit;

            // handle top speed contraint
            if (v + a * dt > car->capabilities.top_speed) {
                a = (car->capabilities.top_speed - v) / dt;
            }
            
            // update speed and position
            MetersPerSecond dv = a * dt;
            Meters ds = v * dt + 0.5 * a * dt * dt;
            v += dv;
            s += ds;    // May cause overshooting end of lane.
            progress = s / lane->length;    // progress after accounting for motion and maybe a lange change. Maybe greater than 1.

            // --------------- handle approaching dead end ---------------
            // -----------------------------------------------------------
            bool approaching_dead_end = lane_get_num_connections(lane) == 0 && s >= lane->length - meters(8.0);
            if (approaching_dead_end) {
                // teleport to the leftmost lane, to have a chance at merging into something.    
                while (lane->adjacent_left && lane->adjacent_left->direction == lane->direction) {
                    lane = lane->adjacent_left;
                }
                s = progress * lane->length;            // progress is constant across neighbor lanes, but s can be different for curved roads.
            }

            // ------------ handle merge --------------------
            // ----------------------------------------------
            bool merge_available = lane_is_merge_available(lane);
            bool merge_intent = lane_change_intent == INDICATOR_LEFT || approaching_dead_end;
            bool is_merge_safe = true;  // todo
            if (merge_available && merge_intent && is_merge_safe) {
                s += lane->merges_into_start * lane->merges_into->length;
                lane = lane->merges_into;
                progress = s / lane->length;
                lane_change_intent = INDICATOR_NONE;
            }

            // ---------- handle turn -----------------------
            // ----------------------------------------------
            while (s > lane->length) {
                s -= lane->length;
                CarIndictor turn_intent = car_get_indicator_turn(car);
                if (turn_intent == INDICATOR_LEFT && lane->for_left_connects_to) {
                    lane = lane->for_left_connects_to;
                } else if (turn_intent == INDICATOR_RIGHT && lane->for_right_connects_to) {
                    lane = lane->for_right_connects_to;
                } else {
                    lane = lane->for_straight_connects_to;
                }
                progress = s / lane->length;
            }

            // ------------ handle exit ---------------------
            // --------------------------------------------
            bool exit_intent = lane_change_intent == INDICATOR_RIGHT;
            // exit_intent = true; // ! for debugging
            bool exit_available = lane_is_exit_lane_available(lane, progress);
            if (exit_intent && exit_available) {
                s = (progress - lane->exit_lane_start) * lane->length;
                lane = lane->exit_lane;
                progress = s / lane->length;
            }

            car_handle_lane_change(car, (Lane*)lane);
            car_set_lane_progress(car, progress);
            car_set_speed(car, v);
        }
        self->time += dt;
    }
}

