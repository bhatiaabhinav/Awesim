#include "sim.h"
#include "logging.h"
#include "ai.h"
#include "bad.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>



// #define BENCHMARK_SIM_LOGIC
// #define BENCHMARK_SIM_UPDATE_CAR

#if defined(BENCHMARK_SIM_LOGIC) || defined(BENCHMARK_SIM_UPDATE_CAR)
#ifdef _WIN32
#include <windows.h>
static double _bench_get_time_us(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double _bench_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}
#endif
#endif

#ifdef BENCHMARK_SIM_LOGIC
static double _bench_sa_total_us = 0;
static double _bench_decisions_total_us = 0;
static double _bench_update_total_us = 0;
static double _bench_post_step_total_us = 0;
static int _bench_step_count = 0;
#define BENCH_INTERVAL 1000 // Print every N steps
#endif

#ifdef BENCHMARK_SIM_UPDATE_CAR
static double _bench_upd_init_total_us = 0;
static double _bench_upd_fuel_total_us = 0;
static double _bench_upd_lanechange_total_us = 0;
static double _bench_upd_physics_total_us = 0;
static double _bench_upd_merge_total_us = 0;
static double _bench_upd_turn_total_us = 0;
static double _bench_upd_exit_total_us = 0;
static double _bench_upd_commit_total_us = 0;
static double _bench_upd_commit_setstate_us = 0;
static double _bench_upd_commit_movement_us = 0;
static double _bench_upd_commit_geometry_us = 0;
static double _bench_upd_commit_indicators_us = 0;
static double _bench_upd_commit_roadinfo_us = 0;
static int _bench_upd_call_count = 0;
#define BENCH_UPD_INTERVAL 500000 // Print every N calls (steps * cars)
#endif

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
    car->lowest_speed_in_stop_zone = INFINITY; // Reset stop zone tracking on lane change
    car->lowest_speed_post_stop_line_before_lane_end = INFINITY; // Reset post stop line tracking on lane change
}

static void sim_update_car(Simulation* self, CarId car_id, Seconds dt) {
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_t0 = _bench_get_time_us();
    #endif
    Car* car = sim_get_car(self, car_id);
    Map* map = sim_get_map(self);
    Lane* lane = car_get_lane(car, map);
    double progress = car_get_lane_progress(car);
    Meters s = car_get_lane_progress_meters(car);
    CarIndicator lane_change_intent = car_get_indicator_lane(car);
    CarIndicator turn_intent = car_get_indicator_turn(car);
    CarIndicator lane_change_requested = car_get_request_indicated_lane(car) ? lane_change_intent : INDICATOR_NONE;
    CarIndicator turn_requested = car_get_request_indicated_turn(car) ? turn_intent : INDICATOR_NONE;
    MetersPerSecondSquared a = car_get_acceleration(car);
    // multiply acceleration by grip of the lane
    a *= lane->grip;
    car_set_acceleration(car, a);
    MetersPerSecond v = car_get_speed(car);
    bool lane_changed = false;
    bool turned = false;
    bool lane_changed_left = false;
    bool lane_changed_right = false;
    bool turned_left = false;
    bool turned_right = false;
    Lane* original_lane = lane;
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_t1 = _bench_get_time_us();
    #endif

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
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_t2 = _bench_get_time_us();
    #endif

    LOG_TRACE("T=%.2f: Processing car %d on lane %d with progress %.5f (s = %.5f), v = %.2f, a = %.2f, lane_change_requested = %d, turn_requested = %d", self->time, car->id, lane->id, progress, s, v, a, lane_change_requested, turn_requested);

    Lane* adjacent_left = lane_get_adjacent_left(lane, map);
    Lane* adjacent_right = lane_get_adjacent_right(lane, map);
    if(lane_change_requested == INDICATOR_LEFT && adjacent_left && adjacent_left->direction == lane->direction) {
        lane = adjacent_left;
        lane_change_requested = INDICATOR_NONE; // Reset request after lane change
        lane_changed = true;
        lane_changed_left = true;
        s = progress * lane->length; // Update s after changing lane. Progress is constant across neighbor lanes, but s can be different for curved roads.
        LOG_TRACE("Car %d changed to left lane %d", car->id, lane->id);
    } else if (lane_change_requested == INDICATOR_RIGHT && adjacent_right) {
        assert(adjacent_right->direction == lane->direction && "How did this happen? Map error!");
        lane = adjacent_right;
        lane_change_requested = INDICATOR_NONE; // Reset request after lane change
        lane_changed = true;
        lane_changed_right = true;
        s = progress * lane->length; // Update s after changing lane. Progress is constant across neighbor lanes, but s can be different for curved roads.
        LOG_TRACE("Car %d changed to right lane %d", car->id, lane->id);
    } else {
        // Do nothing
    }
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_t3 = _bench_get_time_us();
    #endif

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
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_t4 = _bench_get_time_us();
    #endif

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
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_t5 = _bench_get_time_us();
    #endif

    // ---------- handle turn -----------------------
    // ----------------------------------------------
    while (s > lane->length) {
        LOG_TRACE("Car %d: Progress %.2f (s = %.2f) exceeds lane length %.2f. Handling turn. Connections: left = %d, straight = %d, right = %d", car->id, progress, s, lane->length, lane->connections[0], lane->connections[1], lane->connections[2]);
        Lane* pre_transition_lane = lane;
        CarIndicator this_turn_direction = INDICATOR_NONE; // Track which direction was taken in this iteration
        Lane* connection_left = lane_get_connection_left(lane, map);
        Lane* connection_straight = lane_get_connection_straight(lane, map);
        Lane* connection_right = lane_get_connection_right(lane, map);
        if (turn_requested == INDICATOR_LEFT && connection_left) {
            s -= lane->length;
            lane = connection_left;
            turn_requested = INDICATOR_NONE; // Reset request after turn
            turned = true;
            turned_left = true;
            this_turn_direction = INDICATOR_LEFT;
            LOG_TRACE("Car %d turned left into lane %d.", car->id, lane->id);
        } else if (turn_requested == INDICATOR_RIGHT && connection_right) {
            s -= lane->length;
            lane = connection_right;
            turn_requested = INDICATOR_NONE; // Reset request after turn
            turned = true;
            turned_right = true;
            this_turn_direction = INDICATOR_RIGHT;
            LOG_TRACE("Car %d turned right into lane %d.", car->id, lane->id);
        } else {
            if (connection_straight) {
                s -= lane->length;
                lane = connection_straight;
                LOG_TRACE("Car %d went straight into lane %d", car->id, lane->id);
            } else if (connection_right) {
                s -= lane->length;
                lane = connection_right;
                turned = true;
                turned_right = true;
                this_turn_direction = INDICATOR_RIGHT;
                LOG_TRACE("Car %d had to turn right into lane %d as no straight connection was available.", car->id, lane->id);
            } else if (connection_left) {
                s -= lane->length;
                lane = connection_left;
                turned = true;
                turned_left = true;
                this_turn_direction = INDICATOR_LEFT;
                LOG_TRACE("Car %d had to turn left into lane %d as no straight or right connection was available.", car->id, lane->id);
            } else {
                LOG_TRACE("Car %d has no available connections from lane %d. It has reached a dead end.", car->id, lane->id);
                // dead end. Clamp to end of lane and stop the car. Movement ignored = s - lane->length.
                net_forward_movement_meters -= (s - lane->length);
                s = lane->length; // no connections, so clamp s to lane length.
                v = 0; // and stop the car.
            }
        }
        // Detect violations when entering an intersection lane from a road lane
        if (self->traffic_violations_logs_queue.enabled[car_id] && !pre_transition_lane->is_at_intersection && lane->is_at_intersection) {
            SituationalAwareness* sa = sim_get_situational_awareness(self, car_id);
            // Red light violation: was the light red for the direction taken?
            TrafficLight light = sa->light_for_turn[this_turn_direction];
            if (light == TRAFFIC_LIGHT_RED) {
                traffic_violation_log(&self->traffic_violations_logs_queue, car_id, TRAFFIC_VIOLATION_RED_LIGHT_RUN, self->time, fabs(v));
            }
            // Red yield fail-to-stop: entered intersection on a red yield without having stopped first
            if (light == TRAFFIC_LIGHT_RED_YIELD && !sa->did_stop_in_stop_zone && !sa->did_stop_post_stop_line_before_lane_end) {
                traffic_violation_log(&self->traffic_violations_logs_queue, car_id, TRAFFIC_VIOLATION_RED_YIELD_FAIL_TO_STOP, self->time, car->lowest_speed_in_stop_zone);
            }
            // Stop sign violation: at a 4-way stop, past the stop zone without having fully stopped
            Intersection* ixn = lane_get_intersection(lane, map);
            if (ixn && ixn->state == FOUR_WAY_STOP && !sa->did_stop_in_stop_zone) {
                traffic_violation_log(&self->traffic_violations_logs_queue, car_id, TRAFFIC_VIOLATION_STOP_SIGN_FAIL_TO_STOP, self->time, car->lowest_speed_in_stop_zone);
            }
        }
        // Track intersection passage: count when the car exits an intersection lane onto a non-intersection lane
        if (pre_transition_lane->is_at_intersection && !lane->is_at_intersection) {
            Intersection* ixn = lane_get_intersection(pre_transition_lane, map);
            if (ixn) {
                car->travel_stats.intersections_passed++;
                if (ixn->state == FOUR_WAY_STOP) {
                    car->travel_stats.stop_signs_passed++;
                } else if (ixn->is_T_junction) {
                    car->travel_stats.t_junctions_passed++;
                } else {
                    car->travel_stats.traffic_lights_passed++;
                }
            }
        }
        progress = s / lane->length;
        LOG_TRACE("Car %d: New lane %d at progress %.2f (s = %.2f)", car->id, lane->id, progress, s);
    }

    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_t6 = _bench_get_time_us();
    #endif
    // ------------ handle exit ---------------------
    // --------------------------------------------
    bool exit_intent = lane_change_requested == INDICATOR_RIGHT;
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
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_t7 = _bench_get_time_us();
    #endif

    car_set_lane_progress(car, progress, s, net_forward_movement_meters);
    car_set_speed(car, v);
    car_set_fuel_level(car, fuel_remaining);

    // Update lowest speed in stop zone tracking
    {
        Meters car_half_length = car_get_length(car) * 0.5;
        Meters dist_to_end = lane->length * (1.0 - progress);
        Meters dist_from_leading_edge = dist_to_end - car_half_length;
        Meters stop_zone_near = STOP_LINE_BUFFER_METERS + STOP_LINE_BUFFER_TOLERANCE_METERS;
        Meters stop_zone_far  = STOP_LINE_BUFFER_METERS - STOP_LINE_BUFFER_TOLERANCE_METERS;
        if (dist_from_leading_edge <= stop_zone_near && dist_from_leading_edge >= stop_zone_far) {
            MetersPerSecond abs_v = fabs(v);
            if (abs_v < car->lowest_speed_in_stop_zone) {
                car->lowest_speed_in_stop_zone = abs_v;
            }
        }
        if (dist_from_leading_edge < STOP_LINE_BUFFER_METERS && dist_from_leading_edge > 0) {
            MetersPerSecond abs_v = fabs(v);
            if (abs_v < car->lowest_speed_post_stop_line_before_lane_end) {
                car->lowest_speed_post_stop_line_before_lane_end = abs_v;
            }
        }
    }
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_c1 = _bench_get_time_us();
    #endif
    car_handle_movement_or_lane_change(self, car, lane);
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_c2 = _bench_get_time_us();
    #endif
    car_update_geometry(self, car);
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_c3 = _bench_get_time_us();
    #endif
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
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_c4 = _bench_get_time_us();
    #endif

    // ---- Update travel stats ----
    {
        CarTravelStats* stats = &car->travel_stats;
        Meters ds_abs = fabs(net_forward_movement_meters);

        // Distance and time
        stats->total_distance_traveled += ds_abs;
        stats->total_displacement += net_forward_movement_meters;
        stats->total_time_traveled += dt;
        stats->average_speed = stats->total_displacement / stats->total_time_traveled;

        // Turns and lane changes
        if (turned_left)         stats->turns_left_made++;
        if (turned_right)        stats->turns_right_made++;
        if (lane_changed_left)   stats->lane_changes_left++;
        if (lane_changed_right)  stats->lane_changes_right++;

        // Speed-based time classification
        MetersPerSecond abs_speed = fabs(v);
        if (abs_speed < STOP_SPEED_THRESHOLD) {
            stats->total_time_stopped += dt;
        }
        if (abs_speed < ALMOST_STOP_SPEED_THRESHOLD) {
            stats->total_time_almost_stopped += dt;
        } else {
            stats->total_time_moving += dt;
        }

        // Lane position and road type stats (only when on a road, not an intersection)
        Road* orig_road = lane_get_road(original_lane, map);
        if (orig_road) {
            int num_road_lanes = orig_road->num_lanes;
            int lane_index = original_lane->index_in_road;

            if (num_road_lanes == 1) {
                stats->total_distance_on_single_lane_roads += ds_abs;
                stats->total_time_on_single_lane_roads += dt;
            } else if (num_road_lanes == 2) {
                stats->total_distance_on_two_lane_roads += ds_abs;
                stats->total_time_on_two_lane_roads += dt;
                if (lane_index == 0) {
                    stats->total_distance_on_leftmost_lane += ds_abs;
                    stats->total_time_on_leftmost_lane += dt;
                } else {
                    stats->total_distance_on_rightmost_lane += ds_abs;
                    stats->total_time_on_rightmost_lane += dt;
                }
            } else if (num_road_lanes >= 3) {
                stats->total_distance_on_three_or_more_lane_roads += ds_abs;
                stats->total_time_on_three_or_more_lane_roads += dt;
                if (lane_index == 0) {
                    stats->total_distance_on_leftmost_lane += ds_abs;
                    stats->total_time_on_leftmost_lane += dt;
                } else if (lane_index == num_road_lanes - 1) {
                    stats->total_distance_on_rightmost_lane += ds_abs;
                    stats->total_time_on_rightmost_lane += dt;
                } else {
                    stats->total_distance_on_middle_lane += ds_abs;
                    stats->total_time_on_middle_lane += dt;
                }
            }
        }
    }

    // ---- Detect speeding and tailgating violations ----
    if (self->traffic_violations_logs_queue.enabled[car_id]) {
        MetersPerSecond abs_speed = fabs(v);
        MetersPerSecond speed_limit = lane->speed_limit;
        if (abs_speed > speed_limit + OVERSPEEDING_THRESHOLD_MPS) {
            // Probabilistic detection: probability per second * dt
            if (rand_0_to_1() < OVERSPEEDING_DETECTION_PROBABILITY * dt) {
                traffic_violation_log(&self->traffic_violations_logs_queue, car_id, TRAFFIC_VIOLATION_OVERSPEEDING, self->time, abs_speed - speed_limit);
            }
        }

        // Tailgating: too close to lead vehicle at speed, and not braking
        SituationalAwareness* sa = sim_get_situational_awareness(self, car_id);
        if (abs_speed > TAILGATING_MIN_SPEED_MPS && sa->nearby_vehicles.lead && a >= 0) {
            Meters distance_to_lead;
            perceive_lead_vehicle(car, self, sa, &distance_to_lead, NULL, NULL);
            Meters bumper_to_bumper = distance_to_lead - (car_get_length(car) + car_get_length(sa->nearby_vehicles.lead)) * 0.5;
            Seconds time_headway = bumper_to_bumper / abs_speed;
            if (time_headway < TAILGATING_TIME_HEADWAY_THRESHOLD) {
                if (rand_0_to_1() < TAILGATING_DETECTION_PROBABILITY * dt) {
                    traffic_violation_log(&self->traffic_violations_logs_queue, car_id, TRAFFIC_VIOLATION_TAILGATING, self->time, time_headway);
                }
            }
        }
    }

    Road* road = lane_get_road(lane, map);
    Intersection* intersection = lane_get_intersection(lane, map);
    const char* road_name = road ? road->name : intersection ? intersection->name : "Unknown";
    LOG_TRACE("Car %d processing complete: Updated state: lane %d (%s), progress %.5f (%.5f miles), speed = %.5f mps (%.5f mph), acceleration = %.5f mpss", car->id, lane->id, road_name, progress, to_miles(s), v, to_mph(v), a);
    #ifdef BENCHMARK_SIM_UPDATE_CAR
    double _buc_t8 = _bench_get_time_us();
    _bench_upd_init_total_us      += _buc_t1 - _buc_t0;
    _bench_upd_fuel_total_us      += _buc_t2 - _buc_t1;
    _bench_upd_lanechange_total_us += _buc_t3 - _buc_t2;
    _bench_upd_physics_total_us   += _buc_t4 - _buc_t3;
    _bench_upd_merge_total_us     += _buc_t5 - _buc_t4;
    _bench_upd_turn_total_us      += _buc_t6 - _buc_t5;
    _bench_upd_exit_total_us      += _buc_t7 - _buc_t6;
    _bench_upd_commit_total_us    += _buc_t8 - _buc_t7;
    _bench_upd_commit_setstate_us  += _buc_c1 - _buc_t7;
    _bench_upd_commit_movement_us  += _buc_c2 - _buc_c1;
    _bench_upd_commit_geometry_us  += _buc_c3 - _buc_c2;
    _bench_upd_commit_indicators_us += _buc_c4 - _buc_c3;
    _bench_upd_commit_roadinfo_us  += _buc_t8 - _buc_c4;
    _bench_upd_call_count++;
    if (_bench_upd_call_count % BENCH_UPD_INTERVAL == 0) {
        double n = (double)_bench_upd_call_count;
        printf("[BENCH sim_update_car] Avg over %d calls: Init=%.3f us | Fuel=%.3f us | LaneChg=%.3f us | Physics=%.3f us | Merge=%.3f us | Turn=%.3f us | Exit=%.3f us | Commit=%.3f us | Total=%.3f us\n",
            _bench_upd_call_count,
            _bench_upd_init_total_us / n,
            _bench_upd_fuel_total_us / n,
            _bench_upd_lanechange_total_us / n,
            _bench_upd_physics_total_us / n,
            _bench_upd_merge_total_us / n,
            _bench_upd_turn_total_us / n,
            _bench_upd_exit_total_us / n,
            _bench_upd_commit_total_us / n,
            (_bench_upd_init_total_us + _bench_upd_fuel_total_us + _bench_upd_lanechange_total_us + _bench_upd_physics_total_us + _bench_upd_merge_total_us + _bench_upd_turn_total_us + _bench_upd_exit_total_us + _bench_upd_commit_total_us) / n);
        printf("  [Commit breakdown] SetState=%.3f us | Movement=%.3f us | Geometry=%.3f us | Indicators=%.3f us | RoadInfo=%.3f us\n",
            _bench_upd_commit_setstate_us / n,
            _bench_upd_commit_movement_us / n,
            _bench_upd_commit_geometry_us / n,
            _bench_upd_commit_indicators_us / n,
            _bench_upd_commit_roadinfo_us / n);
    }
    #endif
}

// #define DEBUG_COLLISIONS

#ifdef DEBUG_COLLISIONS
static Collisions* collisions_checker = NULL;
static Collisions* prev_collisions_checker = NULL;
static int num_collisions = 0;
static double collisions_per_minute = 0.0;
#endif

void sim_integrate(Simulation* self, Seconds time_period) {
    #ifdef DEBUG_COLLISIONS
    if (!collisions_checker) {
        collisions_checker = malloc(sizeof(Collisions));
        if (!collisions_checker) {
            LOG_ERROR("Failed to allocate memory for collisions checker");
            exit(EXIT_FAILURE);
        }
    }
    if (!prev_collisions_checker) {
        prev_collisions_checker = malloc(sizeof(Collisions));
        if (!prev_collisions_checker) {
            LOG_ERROR("Failed to allocate memory for prev collisions checker");
            exit(EXIT_FAILURE);
        }
        collisions_reset(prev_collisions_checker);
    }
    #endif
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
            intersection_update((Intersection*)intersection, self, dt);
        }
        LOG_TRACE("Updated intersections");

        LOG_TRACE("Building situational awareness for all cars");
        #ifdef BENCHMARK_SIM_LOGIC
        double _bench_sa_start = _bench_get_time_us();
        #endif
        for(int car_id=0; car_id < self->num_cars; car_id++) {
            situational_awareness_build(self, car_id);
        }
        #ifdef BENCHMARK_SIM_LOGIC
        _bench_sa_total_us += _bench_get_time_us() - _bench_sa_start;
        #endif

        LOG_TRACE("Having all NPC cars make decisions");
        #ifdef BENCHMARK_SIM_LOGIC
        double _bench_decisions_start = _bench_get_time_us();
        #endif
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
        #ifdef BENCHMARK_SIM_LOGIC
        _bench_decisions_total_us += _bench_get_time_us() - _bench_decisions_start;
        #endif

        LOG_TRACE("Updating map and car state for all cars");
        #ifdef BENCHMARK_SIM_LOGIC
        double _bench_update_start = _bench_get_time_us();
        #endif
        for(int car_id=0; car_id < self->num_cars; car_id++) {
            sim_update_car(self, car_id, dt);
        }
        #ifdef BENCHMARK_SIM_LOGIC
        _bench_update_total_us += _bench_get_time_us() - _bench_update_start;
        #endif

        LOG_TRACE("Marking current situational awareness as invalid for all cars since the simulation has advanced.");
        for(int car_id=0; car_id < self->num_cars; car_id++) {
            SituationalAwareness* sa = sim_get_situational_awareness(self, car_id);
            sa->is_valid = false; // Reset situational awareness validity for the next iteration
        }

        LOG_TRACE("Running post-simulation step for driving assistants of all cars");
        #ifdef BENCHMARK_SIM_LOGIC
        double _bench_post_start = _bench_get_time_us();
        #endif
        for (int car_id = 0; car_id < self->num_cars; car_id++) {
            Car* car = sim_get_car(self, car_id);
            DrivingAssistant* das = sim_get_driving_assistant(self, car_id);
            if (das) {
                driving_assistant_post_sim_step(das, car, self);
            }
        }
        #ifdef BENCHMARK_SIM_LOGIC
        _bench_post_step_total_us += _bench_get_time_us() - _bench_post_start;
        _bench_step_count++;
        if (_bench_step_count % BENCH_INTERVAL == 0) {
            double avg_sa = _bench_sa_total_us / _bench_step_count;
            double avg_dec = _bench_decisions_total_us / _bench_step_count;
            double avg_upd = _bench_update_total_us / _bench_step_count;
            double avg_post = _bench_post_step_total_us / _bench_step_count;
            double avg_total = avg_sa + avg_dec + avg_upd + avg_post;
            printf("[BENCH sim_logic] Avg over %d steps (%d cars): SA=%.1f us | Decisions=%.1f us | Update=%.1f us | PostStep=%.1f us | Total=%.1f us (%.2f ms)\n",
                _bench_step_count, self->num_cars, avg_sa, avg_dec, avg_upd, avg_post, avg_total, avg_total / 1000.0);
        }
        #endif

        #ifdef DEBUG_COLLISIONS
        Collisions* temp = prev_collisions_checker;
        prev_collisions_checker = collisions_checker;
        collisions_checker = temp;

        collisions_reset(collisions_checker);
        // collisions_detect_for_car(collisions_checker, self, 0);
        collisions_detect_all(collisions_checker, self);

        int new_collisions_this_step = 0;
        if (collisions_checker->total_collisions > 0) {
            for (int i = 0; i < self->num_cars; i++) {
                int count = collisions_get_count_for_car(collisions_checker, i);
                for (int k = 0; k < count; k++) {
                    CarId other_id = collisions_get_colliding_car(collisions_checker, i, k);
                    if (i < other_id) {
                        if (!collisions_are_colliding(prev_collisions_checker, i, other_id)) {
                            new_collisions_this_step++;
                        }
                    }
                }
            }
        }

        if (new_collisions_this_step > 0) {
            num_collisions += new_collisions_this_step;
            double current_sim_time = self->time + dt;
            if (current_sim_time > 1e-6) {
                collisions_per_minute = (num_collisions / current_sim_time) * 60.0;
            } else {
                collisions_per_minute = 0.0;
            }
            LOG_INFO("New collisions detected at time %.2f seconds! Total collisions so far: %d, Collisions per minute: %.4f. Per car per minute: %.6f", current_sim_time, num_collisions, collisions_per_minute, collisions_per_minute / self->num_cars);
            // for (int car_idx = 0; car_idx < self->num_cars; car_idx++) {
            //     int num_collisions = collisions_checker->num_collisions_per_car[car_idx];
            //     if (num_collisions > 0) {
            //         Car* car = sim_get_car(self, car_idx);
            //         for (int c = 0; c < num_collisions; c++) {
            //             CarId other_car_id = collisions_checker->colliding_cars[car_idx][c];
            //             LOG_INFO(" - Car %d collided with Car %d", car->id, other_car_id);
            //         }
            //     }
            // }
        }
        #endif

        self->time += dt;
    }
    LOG_TRACE("Simulation advanced to time %.2f", self->time);
}
