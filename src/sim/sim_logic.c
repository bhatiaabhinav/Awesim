#include "sim.h"
#include "logging.h"
#include "ai.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

static void car_handle_movement_or_lane_change(Simulation* sim, Car* car, Lane* new_lane) {
    Map* map = sim_get_map(sim);
    Lane* current_lane = car_get_lane(car, map);
    if (current_lane == new_lane) {
        lane_move_car(current_lane, car, sim);
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
    lane_remove_car(current_lane, car, sim);
    lane_add_car(new_lane, car, sim);
    car_set_lane(car, new_lane);
}

void sim_integrate(Simulation* self, Seconds time_period) {
    Seconds t0 = self->time;
    Seconds dt = self->dt;
    Map* map = sim_get_map(self);
    if (t0 + time_period < t0) {
        return;
    }
    LOG_TRACE("Simulating from time %.2f to %.2f with dt = %.2f", t0, t0 + time_period, dt);
    while (self->time < t0 + time_period) {

        for (int i = 0; i < map->num_intersections; i++) {
            const Intersection* intersection = map_get_intersection(map, i);
            intersection_update((Intersection*)intersection, dt);
        }
        LOG_TRACE("Updated intersections");

        LOG_TRACE("Building situational awareness for all cars");
        for(int car_id=0; car_id < self->num_cars; car_id++) {
            situational_awareness_build(self, car_id);
        }

        LOG_TRACE("Having all NPC cars make decisions");
        for(int car_id=0; car_id < self->num_cars; car_id++) {
            Car* car = sim_get_car(self, car_id);
            if (!sim_is_agent_enabled(self) || car_id > 0) {
                // NPC car logic
                npc_car_make_decisions(car, self);
            } else {
                // Assume that the first car (id 0) is the agent car and it is either controlled externally or has a driving assistant enabled.
                if (self->is_agent_driving_assistant_enabled) {
                    DrivingAssistant* da = sim_get_driving_assistant(self, car_id);
                    LOG_TRACE("Agent car %d has a driving assistant enabled. Using it to set controls.", car->id);
                    driving_assistant_control_car(da, car, self);
                } else {
                    // Agent car logic without driving assistant
                    LOG_TRACE("Agent car %d does not have a driving assistant enabled. Assuming it is controlled externally.", car->id);
                }
            }
        }

        LOG_TRACE("Updating map and car state for all cars");
        for(int car_id=0; car_id < self->num_cars; car_id++) {
            Car* car = sim_get_car(self, car_id);
            Lane* lane = car_get_lane(car, map);
            double progress = car_get_lane_progress(car);
            Meters s = car_get_lane_progress_meters(car);
            CarIndicator lane_change_intent = car_get_indicator_lane(car);
            CarIndicator turn_intent = car_get_indicator_turn(car);
            CarIndicator lane_change_requested = car_get_request_indicated_lane(car) ? lane_change_intent : INDICATOR_NONE;
            CarIndicator turn_requested = car_get_request_indicated_turn(car) ? turn_intent : INDICATOR_NONE;
            MetersPerSecondSquared a = car_get_acceleration(car);
            MetersPerSecond v = car_get_speed(car);
            bool lane_changed = false;
            bool turned = false;

            LOG_TRACE("Car %d fuel level: %.2f liters", car->id, car_get_fuel_level(car));
            // If fuel is zero, simulate coasting to a stop in 1 minute using a speed-proportional controller (a = -√k ⋅ v). The settling time is 4 / √k => k = (4 / settling_time)^2 = (4/60)^2. If the car is applying brakes, add that effect as well.
            if (car_get_fuel_level(car) <= 1e-10) {
                double sqrt_k = 4.0 / 60.0;
                double a_coast = -sqrt_k * v;
                double a_brake = 0;
                // now check if brakes are being applied.
                if (v * a < 0) {
                    // brakes are being applied, so add that effect as well.
                    a_brake = a + a_coast; // a and a_coast are aligned.
                } else {
                    // no brakes, so use coasting deceleration
                    a_brake = a_coast;
                }
                a = a_brake;
                car_set_acceleration(car, a);
                LOG_TRACE("Car %d is out of fuel and is coasting to a stop with acceleration %.2f", car->id, a);
            }

            LOG_TRACE("T=%.2f: Processing car %d on lane %d with progress %.5f (s = %.5f), v = %.2f, a = %.2f, lane_change_requested = %d, turn_requested = %d", self->time, car->id, lane->id, progress, s, v, a, lane_change_requested, turn_requested);

            Lane* adjacent_left = lane_get_adjacent_left(lane, map);
            Lane* adjacent_right = lane_get_adjacent_right(lane, map);
            if(lane_change_requested == INDICATOR_LEFT && adjacent_left && adjacent_left->direction == lane->direction) {
                // TODO: ignore lane change if this is an NPC if a collision is possible with another vehicle
                // TODO: Allow non-NPC to do unsafe lane change (and receive a penalty for that).
                // TODO: Maybe it is ok to allow to NPCs to do unsafe lane changes as well? So that they can pass the leading vehicle?
                lane = adjacent_left;
                lane_change_requested = INDICATOR_NONE; // Reset request after lane change
                lane_changed = true;
                s = progress * lane->length; // Update s after changing lane. Progress is constant across neighbor lanes, but s can be different for curved roads.
                LOG_TRACE("Car %d changed to left lane %d", car->id, lane->id);
            } else if (lane_change_requested == INDICATOR_RIGHT && adjacent_right) {
                assert(adjacent_right->direction == lane->direction && "How did this happen? Map error!");
                // TODO: ignore lane change if this is an NPC if a collision is possible with another vehicle
                lane = adjacent_right;
                lane_change_requested = INDICATOR_NONE; // Reset request after lane change
                lane_changed = true;
                s = progress * lane->length; // Update s after changing lane. Progress is constant across neighbor lanes, but s can be different for curved roads.
                LOG_TRACE("Car %d changed to right lane %d", car->id, lane->id);
            } else {
                // Do nothing
            }

            // ! for debugging
            // a = 0;
            // v = lane->speed_limit;

            // handle top speed contraint
            MetersPerSecond _v = v; // v before applying acceleration
            MetersPerSecond dv = a * dt;
            v = _v + dv;
            // handle top speed constraint
            if (v > car->capabilities.top_speed) {
                a = (car->capabilities.top_speed - _v) / dt;
                v = car->capabilities.top_speed;
                dv = v - _v;
                LOG_TRACE("Handling top speed constraint for car %d: adjusted a = %.2f, adjusted v = %.2f", car->id, a, v);
                car_set_acceleration(car, a);
            }
            // handle case where a car is applying brakes and should not reverse speed sign in the same frame
            if (_v > 0 && v < 0) {
                a = -_v / dt;
                v = 0;
                dv = v - _v;
                car_set_acceleration(car, a);
            }
            if (_v < 0 && v > 0) {
                a = -_v / dt;
                v = -0;
                dv = v - _v;
                car_set_acceleration(car, a);
            }

            Meters ds = _v * dt + 0.5 * a * dt * dt;
            
            Meters s_initial = s; // Save initial position
            s += ds;    // May cause overshooting end of lane.
            progress = s / lane->length;    // progress after accounting for motion and maybe a lane change. Maybe greater than 1.
            if (progress < 0.0) {
                LOG_TRACE("Car %d: Progress %.4f (s = %.2f) on lane %d has become negative. Maybe it reversed (speed = %.2f) too much? Clamping progress and speed to 0.0. The sim is yet to support backing on to a previous lane.", car->id, progress, s, lane->id, v);
                progress = 0.0; // Clamp progress to 0 if it goes negative.
                progress += rand_0_to_1() * 0.01; // Add a small random value to avoid multiple cars having the same progress.
                if (v < 0) v = 0; // If speed is negative, clamp it to 0 as well.
                s = progress * lane->length; // Update s after calculating progress.
            }
            Meters net_forward_movement_meters = s - s_initial; // Calculate net movement in meters.
            LOG_TRACE("Car %d: Updated speed = %.2f, position s = %.2f, Net forward movement = %.2f, progress = %.5f", car->id, v, s, net_forward_movement_meters, progress);
            Liters fuel_consumed = fuel_consumption_compute_typical(_v, a, dt);
            Liters fuel_remaining = fmax(car_get_fuel_level(car) - fuel_consumed, 0);

            // ------------ handle merge --------------------
            // ----------------------------------------------
            bool merge_available = lane_is_merge_available(lane);
            bool merge_intent = lane_change_requested == INDICATOR_LEFT;
            if (merge_available && merge_intent) {
                Lane* merges_into = lane_get_merge_into(lane, map);
                s += lane->merges_into_start * merges_into->length;
                lane = merges_into;
                progress = s / lane->length;
                lane_change_requested = INDICATOR_NONE; // Reset request after merge
                lane_changed = true;
                LOG_TRACE("Car %d is merged into lane %d at progress %.2f (s = %.2f)", car->id, lane->id, progress, s);
            }

            // ---------- handle turn -----------------------
            // ----------------------------------------------
            while (s > lane->length) {
                LOG_TRACE("Car %d: Progress %.2f (s = %.2f) exceeds lane length %.2f. Handling turn. Connections: left = %d, straight = %d, right = %d", car->id, progress, s, lane->length, lane->connections[0], lane->connections[1], lane->connections[2]);
                Lane* connection_left = lane_get_connection_left(lane, map);
                Lane* connection_straight = lane_get_connection_straight(lane, map);
                Lane* connection_right = lane_get_connection_right(lane, map);
                if (turn_requested == INDICATOR_LEFT && connection_left) {
                    s -= lane->length;
                    lane = connection_left;
                    turn_requested = INDICATOR_NONE; // Reset request after turn
                    turned = true;
                    LOG_TRACE("Car %d turned left into lane %d.", car->id, lane->id);
                } else if (turn_requested == INDICATOR_RIGHT && connection_right) {
                    s -= lane->length;
                    lane = connection_right;
                    turn_requested = INDICATOR_NONE; // Reset request after turn
                    turned = true;
                    LOG_TRACE("Car %d turned right into lane %d.", car->id, lane->id);
                } else {
                    if (connection_straight) {
                        s -= lane->length;
                        lane = connection_straight;
                        LOG_TRACE("Car %d went straight into lane %d", car->id, lane->id);
                    } else {
                        // dead end. Clamp to end of lane and stop the car. Movement ignored = s - lane->length.
                        net_forward_movement_meters -= (s - lane->length);
                        s = lane->length; // no connections, so clamp s to lane length.
                        v = 0; // and stop the car.
                    }
                }
                progress = s / lane->length;
                LOG_TRACE("Car %d: New lane %d at progress %.2f (s = %.2f)", car->id, lane->id, progress, s);
            }

            // ------------ handle exit ---------------------
            // --------------------------------------------
            bool exit_intent = lane_change_requested == INDICATOR_RIGHT;
            // exit_intent = true; // ! for debugging
            bool exit_available = lane_is_exit_lane_available(lane, progress);
            if (exit_intent && exit_available) {
                s = (progress - lane->exit_lane_start) * lane->length;
                lane = lane_get_exit_lane(lane, map);
                progress = s / lane->length;
                lane_change_requested = INDICATOR_NONE; // Reset request after exit
                lane_changed = true;
                LOG_TRACE("Car %d exited into lane %d at progress %.2f (s = %.2f)", car->id, lane->id, progress, s);
            }
            s = progress * lane->length; // update s after all changes to lane and progress.

            car_set_lane_progress(car, progress, s, net_forward_movement_meters);
            car_set_speed(car, v);
            car_set_fuel_level(car, fuel_remaining);
            car_handle_movement_or_lane_change(self, car, lane);
            car_update_geometry(self, car);
            if (lane_changed) { 
                car_set_request_indicated_lane(car, false); // Reset lane change request after successful lane change
                if (car_get_auto_turn_off_indicators(car)) {
                    car_set_indicator_lane(car, INDICATOR_NONE); // Automatically turn off lane change indicator after changing lane
                    LOG_TRACE("Car %d: Automatically turned off lane change indicator after changing to lane %d", car->id, lane->id);
                }
            }
            if (turned) {
                car_set_request_indicated_turn(car, false); // Reset turn request after successful turn
                if (car_get_auto_turn_off_indicators(car)) {
                    car_set_indicator_turn(car, INDICATOR_NONE); // Automatically turn off turn indicator after turning
                    LOG_TRACE("Car %d: Automatically turned off turn indicator after turning to lane %d", car->id, lane->id);
                }
            }
            Road* road = lane_get_road(lane, map);
            Intersection* intersection = lane_get_intersection(lane, map);
            const char* road_name = road ? road->name : intersection ? intersection->name : "Unknown";
            LOG_TRACE("Car %d processing complete: Updated state: lane %d (%s), progress %.5f (%.5f miles), speed = %.5f mps (%.5f mph), acceleration = %.5f mpss", car->id, lane->id, road_name, progress, to_miles(s), v, to_mph(v), a);
        }

        LOG_TRACE("Marking current situational awareness as invalid for all cars since the simulation has advanced.");
        for(int car_id=0; car_id < self->num_cars; car_id++) {
            SituationalAwareness* sa = sim_get_situational_awareness(self, car_id);
            sa->is_valid = false; // Reset situational awareness validity for the next iteration
        }

        if (self->is_agent_enabled && self->is_agent_driving_assistant_enabled) {
            LOG_TRACE("Agent car %d has a driving assistant enabled. Running post-simulation step for the driving assistant.", self->cars[0].id);
            DrivingAssistant* das = sim_get_driving_assistant(self, 0);
            driving_assistant_post_sim_step(das, sim_get_car(self, 0), self);
        }

        self->time += dt;
    }
    LOG_TRACE("Simulation advanced to time %.2f", self->time);
}
