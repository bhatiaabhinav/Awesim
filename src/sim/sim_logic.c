#include "sim.h"
#include "logging.h"
#include "ai.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

void car_handle_lane_change(Map* map, Car* car, Lane* new_lane) {
    Lane* current_lane = car_get_lane(car, map);
    if (current_lane == new_lane) {
        return; // No change needed
    }
    if (current_lane == NULL || new_lane == NULL) {
        LOG_ERROR("Car %d cannot change lanes: current lane or new lane is NULL", car->id);
        return;
    }
    if (new_lane->direction == direction_opposite(current_lane->direction)) {
        LOG_ERROR("Car %d cannot change to a lane in the opposite direction: current lane %d, new lane %d", car->id, current_lane->id, new_lane->id);
        return;
    }
    LOG_TRACE("Processing car %d changing lane from %d to %d", car->id, current_lane->id, new_lane->id);
    lane_remove_car(current_lane, car->id);
    lane_add_car(new_lane, car->id);
    car_set_lane(car, new_lane);
}

void simulate(Simulation* self, Seconds time_period) {
    Seconds t0 = self->time;
    Seconds dt = self->dt;
    Map* map = sim_get_map(self);
    LOG_TRACE("Simulating from time %.2f to %.2f with dt = %.2f", t0, t0 + time_period, dt);
    while (self->time < t0 + time_period) {

        for (int i = 0; i < map->num_intersections; i++) {
            const Intersection* intersection = map_get_intersection(map, i);
            intersection_update((Intersection*)intersection, dt);
        }
        LOG_TRACE("Updated intersections");


        for(int car_id=0; car_id < self->num_cars; car_id++) {
            Car* car = sim_get_car(self, car_id);
            if (!sim_is_agent_enabled(self) || car_id > 0) {
                // NPC car logic
                npc_car_make_decisions(car, self);
            } else {
                // Assume that the first car (id 0) is the agent car and it is controlled externally.
                LOG_TRACE("Skipping decision-making for agent car %d, assuming it is controlled externally using car_set_acceleration, car_set_indicator_turn, and car_set_indicator_lane functions.", car->id);
            }
            Lane* lane = car_get_lane(car, map);
            double progress = car_get_lane_progress(car);
            Meters s = car_get_lane_progress_meters(car);
            CarIndictor lane_change_intent = car_get_indicator_lane(car);
            CarIndictor turn_intent = car_get_indicator_turn(car);
            MetersPerSecondSquared a = car_get_acceleration(car);
            MetersPerSecond v = car_get_speed(car);

            LOG_TRACE("T=%.2f: Processing car %d on lane %d with progress %.2f (s = %.2f), v = %.2f, a = %.2f, lane_change_intent = %d, turn_intent = %d", self->time, car->id, lane->id, progress, s, v, a, lane_change_intent, turn_intent);

            Lane* adjacent_left = lane_get_adjacent_left(lane, map);
            Lane* adjacent_right = lane_get_adjacent_right(lane, map);
            if(lane_change_intent == INDICATOR_LEFT && adjacent_left && adjacent_left->direction == lane->direction) {
                // TODO: ignore lane change if this is an NPC if a collision is possible with another vehicle
                // TODO: Allow non-NPC to do unsafe lane change (and receive a penalty for that).
                // TODO: Maybe it is ok to allow to NPCs to do unsafe lane changes as well? So that they can pass the leading vehicle?
                lane = adjacent_left;
                lane_change_intent = INDICATOR_NONE;   // Important: To mark lane change intent already executed, so that merge is ignored if a merge is available on the new lane.
                LOG_TRACE("Car %d changed to left lane %d", car->id, lane->id);
            } else if (lane_change_intent == INDICATOR_RIGHT && adjacent_right) {
                assert(adjacent_right->direction == lane->direction && "How did this happen? Map error!");
                // TODO: ignore lane change if this is an NPC if a collision is possible with another vehicle
                lane = adjacent_right;
                lane_change_intent = INDICATOR_NONE;
                LOG_TRACE("Car %d changed to right lane %d", car->id, lane->id);
            } else {
                // Do nothing
            }

            // ! for debugging
            // a = 0;
            // v = lane->speed_limit;

            // handle top speed contraint
            if (v + a * dt > car->capabilities.top_speed) {
                a = (car->capabilities.top_speed - v) / dt;
                LOG_TRACE("Handling top speed constraint for car %d: adjusted a = %.2f, v = %.2f", car->id, a, v);
            }
            
            // update speed and position
            MetersPerSecond dv = a * dt;
            Meters ds = v * dt + 0.5 * a * dt * dt;
            v += dv;
            s += ds;    // May cause overshooting end of lane.
            progress = s / lane->length;    // progress after accounting for motion and maybe a lange change. Maybe greater than 1.
            if (progress < 0.0) {
                LOG_TRACE("Car %d: Progress %.4f (s = %.2f) on lane %d has become negative. Maybe it reversed (speed = %.2f) too much? Clamping progress and speed to 0.0. The sim is yet to support backing on to a previous lane.", car->id, progress, s, lane->id, v);
                progress = 0.0; // Clamp progress to 0 if it goes negative.
                if (v < 0) v = 0; // If speed is negative, clamp it to 0 as well.
            }
            LOG_TRACE("Car %d: Updated speed = %.2f, position s = %.2f, progress = %.2f", car->id, v, s, progress);

            // --------------- handle approaching dead end ---------------
            // -----------------------------------------------------------
            bool approaching_dead_end = lane_get_num_connections(lane) == 0 && s >= lane->length - meters(8.0);
            if (approaching_dead_end) {
                // teleport to the leftmost lane, to have a chance at merging into something.
                adjacent_left = lane_get_adjacent_left(lane, map);
                adjacent_right = lane_get_adjacent_right(lane, map);
                LOG_TRACE("Car %d is approaching a dead end on lane %d at progress %.2f (s = %.2f). Adjacent lanes are left: %d, right: %d", 
                          car->id, lane->id, progress, s, lane->adjacents[0], lane->adjacents[1]);
                bool teleported = false;
                while (adjacent_left && adjacent_left->direction == lane->direction) {
                    LOG_TRACE("Car %d: Moving to adjacent left lane %d", car->id, adjacent_left->id);
                    lane = adjacent_left;
                    adjacent_left = lane_get_adjacent_left(lane, map);
                    teleported = true;
                }
                s = progress * lane->length;            // progress is constant across neighbor lanes, but s can be different for curved roads.
                if (teleported) LOG_TRACE("Car %d: Teleported to lane %d at progress %.2f (s = %.2f)", car->id, lane->id, progress, s);
            }

            // ------------ handle merge --------------------
            // ----------------------------------------------
            bool merge_available = lane_is_merge_available(lane);
            bool merge_intent = lane_change_intent == INDICATOR_LEFT || approaching_dead_end;
            bool is_merge_safe = true;  // todo
            if (merge_available && merge_intent && is_merge_safe) {
                Lane* merges_into = lane_get_merge_into(lane, map);
                s += lane->merges_into_start * merges_into->length;
                lane = merges_into;
                progress = s / lane->length;
                lane_change_intent = INDICATOR_NONE;
                LOG_TRACE("Car %d is merged into lane %d at progress %.2f (s = %.2f)", car->id, lane->id, progress, s);
            }

            // ---------- handle turn -----------------------
            // ----------------------------------------------
            while (s > lane->length) {
                s -= lane->length;
                Lane* connection_left = lane_get_connection_left(lane, map);
                Lane* connection_straight = lane_get_connection_straight(lane, map);
                Lane* connection_right = lane_get_connection_right(lane, map);
                LOG_TRACE("Car %d: Progress %.2f (s = %.2f) exceeds lane length %.2f. Handling turn. Connections: left = %d, straight = %d, right = %d", car->id, progress, s, lane->length, lane->connections[0], lane->connections[1], lane->connections[2]);
                if (turn_intent == INDICATOR_LEFT && connection_left) {
                    lane = connection_left;
                    turn_intent = INDICATOR_NONE; // Mark turn intent as executed.
                    LOG_TRACE("Car %d turned left into lane %d. Indicator turned off.", car->id, lane->id);
                } else if (turn_intent == INDICATOR_RIGHT && connection_right) {
                    lane = connection_right;
                    turn_intent = INDICATOR_NONE; // Mark turn intent as executed.
                    LOG_TRACE("Car %d turned right into lane %d. Indicator turned off.", car->id, lane->id);
                } else {
                    lane = connection_straight;
                    LOG_TRACE("Car %d went straight into lane %d", car->id, lane->id);
                }
                progress = s / lane->length;
                LOG_TRACE("Car %d: New lane %d at progress %.2f (s = %.2f)", car->id, lane->id, progress, s);
            }

            // ------------ handle exit ---------------------
            // --------------------------------------------
            bool exit_intent = lane_change_intent == INDICATOR_RIGHT;
            // exit_intent = true; // ! for debugging
            bool exit_available = lane_is_exit_lane_available(lane, progress);
            if (exit_intent && exit_available) {
                s = (progress - lane->exit_lane_start) * lane->length;
                lane = lane_get_exit_lane(lane, map);
                progress = s / lane->length;
                lane_change_intent = INDICATOR_NONE; // Mark exit intent as executed.
                LOG_TRACE("Car %d exited into lane %d at progress %.2f (s = %.2f)", car->id, lane->id, progress, s);
            }
            s = progress * lane->length; // update s after all changes to lane and progress.

            car_handle_lane_change(map, car, lane);
            car_set_lane_progress(car, progress, s);
            car_set_speed(car, v);
            LOG_TRACE("Car %d processing complete: Updated lane %d, progress %.2f (s = %.2f), speed = %.2f, acceleration = %.2f", car->id, lane->id, progress, s, v, a);
        }
        self->time += dt;
    }
    LOG_TRACE("Simulation advanced to time %.2f", self->time);
}
