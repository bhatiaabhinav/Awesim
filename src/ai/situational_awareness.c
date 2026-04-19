#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "logging.h"
#include "ai.h"
#include "sim.h"

// #define BENCHMARK_SA_BUILD
// ^^ Uncomment to enable per-section timing of situational_awareness_build
#ifdef BENCHMARK_SA_BUILD
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
static double _bsa_get_time_us(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1e6 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double _bsa_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}
#endif
static double _bsa_basic_state_us = 0;       // speed, stopped, lane/road/intersection setup + stop zone
static double _bsa_lane_context_us = 0;      // adjacency, distances, turn possibilities
static double _bsa_traffic_lights_us = 0;    // traffic light state machine
static double _bsa_merge_exit_us = 0;        // merge/exit/lane-change availability
static double _bsa_target_lanes_us = 0;      // merge-into, exit-lane, target-for-indicator
static double _bsa_tjunction_us = 0;         // T-junction + FCFS stop sign
static double _bsa_lead_us = 0;              // lead vehicle lookup
static double _bsa_brakingdist_us = 0;       // braking distance computation
static int _bsa_call_count = 0;
#define BSA_INTERVAL 500000
#endif

// Inline lane lookup: avoids cross-TU function call overhead of lane_get_connection_* -> map_get_lane
#define _LANE_BY_ID(map, id) (((id) >= 0) ? &(map)->lanes[(id)] : NULL)


void situational_awareness_build(Simulation* sim, CarId car_id) {
    if (!sim) {
        LOG_ERROR("Invalid parameters: sim is NULL. Cannot build situational awareness. Simulation: %p, Car ID: %d", sim, car_id);
        return;
    }
    Car* car = sim_get_car(sim, car_id);
    SituationalAwareness* situation = sim_get_situational_awareness(sim, car_id);
    if (situation->is_valid) {
        LOG_TRACE("Situational awareness for car %d is already valid. No need to rebuild.", car_id);
        return;
    }

    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t0 = _bsa_get_time_us();
    #endif


    // LOG_TRACE("\nBuilding situational awareness for car %d:", car->id);
    Map* map = &sim->map;
    // Meters car_length = car->dimensions.y;
    Meters car_half_length = car->cached_half_length;
    // const Lane* lane = car_get_lane(car, map);
    const Lane* lane = &map->lanes[car->lane_id];
    const Road* road =  lane->road_id >= 0 ? &map->roads[lane->road_id] : NULL;
    const Intersection* intersection = lane->intersection_id >= 0 ? &map->intersections[lane->intersection_id] : NULL;
    double lane_progress = car->lane_progress;
    Meters lane_progress_m = car->lane_progress_meters;

    MetersPerSecond car_speed = car->speed;
    situation->stopped = (fabs(car_speed) < STOP_SPEED_THRESHOLD);
    situation->reversing = (car_speed < -STOP_SPEED_THRESHOLD);
    situation->going_forward = (car_speed > STOP_SPEED_THRESHOLD);
    situation->almost_stopped = (fabs(car_speed) < ALMOST_STOP_SPEED_THRESHOLD);
    situation->speed = car_speed;

    Meters lane_length = lane->length;
    Direction lane_dir = lane->direction;
    Meters distance_to_end_of_lane = lane_length - lane_progress_m;
    Meters distance_to_end_of_lane_from_leading_edge = distance_to_end_of_lane - car_half_length; // Distance to the end of the lane from the leading edge of the car.
    const Intersection* intersection_upcoming = road ? road_leads_to_intersection(road, map) : NULL;   // may be null if we are already on an intersection or this road does not lead to an intersection.

    situation->is_approaching_dead_end = lane->cached_num_connections == 0;
    if (lane->is_at_intersection) {
        situation->intersection = intersection;
        situation->is_an_intersection_upcoming = false;
        situation->is_on_intersection = true;
        situation->is_stopped_at_intersection = false;
    } else {
        situation->is_on_intersection = false;
        situation->intersection = intersection_upcoming; // may be NULL
        situation->is_an_intersection_upcoming = intersection_upcoming != NULL;
        situation->is_stopped_at_intersection = intersection_upcoming && situation->almost_stopped && (distance_to_end_of_lane_from_leading_edge < 50.0);
    }

    // Stop zone detection
    if (situation->is_an_intersection_upcoming) {
        Meters stop_zone_near = STOP_LINE_BUFFER_METERS + STOP_LINE_BUFFER_TOLERANCE_METERS;
        Meters stop_zone_far  = STOP_LINE_BUFFER_METERS - STOP_LINE_BUFFER_TOLERANCE_METERS;

        // Check if currently in the stop zone
        if (distance_to_end_of_lane_from_leading_edge <= stop_zone_near && distance_to_end_of_lane_from_leading_edge >= stop_zone_far) {
            situation->in_stop_zone = true;
        }

        // Check if past the stop zone
        if (distance_to_end_of_lane_from_leading_edge < stop_zone_far) {
            situation->past_stop_zone = true;
        }

        if (distance_to_end_of_lane_from_leading_edge < STOP_LINE_BUFFER_METERS) {
            situation->past_stop_line = true;
        }

        // Use persistent lowest_speed_in_stop_zone from car (tracked by sim_logic, reset on lane change)
        MetersPerSecond lowest = car->lowest_speed_in_stop_zone;
            // Car has been in the stop zone at some point on this lane
        if(lowest < STOP_SPEED_THRESHOLD) {
            if (!situation->did_stop_in_stop_zone) {
                //LOG_DEBUG("Car %d came to a stop in the stop zone (lowest speed: %.2f m/s (%.2f mph))", car_id, lowest, to_mph(lowest));
            }
            situation->did_stop_in_stop_zone = true;
        } else {
            situation->did_stop_in_stop_zone = false;
        }
        if (lowest < ALMOST_STOP_SPEED_THRESHOLD) {
            if (!situation->did_almost_stop_in_stop_zone) {
                //LOG_DEBUG("Car %d almost stopped in the stop zone (lowest speed: %.2f m/s (%.2f mph))", car_id, lowest, to_mph(lowest));
            }
            situation->did_almost_stop_in_stop_zone = true;
        }
        if (lowest < CREEP_SPEED) {
            if (!situation->did_rolling_stop_in_stop_zone) {
                //LOG_DEBUG("Car %d did a rolling stop in the stop zone (lowest speed: %.2f m/s (%.2f mph))", car_id, lowest, to_mph(lowest));
            }
            situation->did_rolling_stop_in_stop_zone = true;
        }
        if (car->lowest_speed_post_stop_line_before_lane_end < STOP_SPEED_THRESHOLD) {
            situation->did_stop_post_stop_line_before_lane_end = true;
        } else {
            situation->did_stop_post_stop_line_before_lane_end = false;
        }
    } else {
        situation->in_stop_zone = false;
        situation->past_stop_zone = false;
        situation->did_stop_in_stop_zone = false;
        situation->did_almost_stop_in_stop_zone = false;
        situation->did_rolling_stop_in_stop_zone = false;
        situation->did_stop_post_stop_line_before_lane_end = false;
    }

    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t0a = _bsa_get_time_us();
    #endif

    situation->road = road;
    situation->lane = lane;
    Lane* adjacent_left = lane_get_adjacent_left(lane, map);
    Lane* adjacent_right = lane_get_adjacent_right(lane, map);
    situation->lane_left = (adjacent_left && adjacent_left->direction == lane->direction) ? adjacent_left : NULL;
    situation->lane_right = adjacent_right;
    situation->distance_to_intersection = intersection_upcoming ? distance_to_end_of_lane : 1e6;
    situation->distance_to_end_of_lane = distance_to_end_of_lane;
    situation->distance_to_end_of_lane_from_leading_edge = distance_to_end_of_lane_from_leading_edge;
    situation->is_approaching_end_of_lane = distance_to_end_of_lane_from_leading_edge < 50.0;
    situation->is_at_end_of_lane = distance_to_end_of_lane_from_leading_edge < 4.0;
    situation->is_on_leftmost_lane = road ? lane == road_get_leftmost_lane(road, map) : false;
    situation->is_on_rightmost_lane = road ? lane == road_get_rightmost_lane(road, map) : false;
    situation->lane_progress = lane_progress;
    situation->lane_progress_m = lane_progress_m;

    situation->is_turn_possible[INDICATOR_LEFT] = lane->connections[INDICATOR_LEFT] != ID_NULL;
    situation->is_turn_possible[INDICATOR_NONE] = lane->connections[INDICATOR_NONE] != ID_NULL;
    situation->is_turn_possible[INDICATOR_RIGHT] = lane->connections[INDICATOR_RIGHT] != ID_NULL;
    situation->lane_next_after_turn[INDICATOR_LEFT] = lane_get_connection_left(lane, map);
    situation->lane_next_after_turn[INDICATOR_NONE] = lane_get_connection_straight(lane, map);
    situation->lane_next_after_turn[INDICATOR_RIGHT] = lane_get_connection_right(lane, map);
    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t1 = _bsa_get_time_us();
    #endif
    if (intersection_upcoming) {
        bool free_right = true;
        bool is_NS = (lane_dir == DIRECTION_NORTH || lane_dir == DIRECTION_SOUTH);
        switch (intersection_upcoming->state) {
            case FOUR_WAY_STOP:
                situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_STOP_FCFS;
                situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_STOP_FCFS;
                situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_STOP_FCFS;
                break;
            case T_JUNC_NS:
                if (is_NS) {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_GREEN_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_GREEN;
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_GREEN;
                } else {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_GREEN_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;          // should not exist
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_GREEN_YIELD;
                }
                break;
            case T_JUNC_EW:
                if (is_NS) {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_GREEN_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;          // should not exist
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_GREEN_YIELD;
                } else {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_GREEN_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_GREEN;
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_GREEN;
                }
                break;
            case NS_GREEN_EW_RED:
                if (is_NS) {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_GREEN_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_GREEN;
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_GREEN;
                } else {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                }
                break;
            case NS_YELLOW_EW_RED:
                if (is_NS) {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_YELLOW_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_YELLOW;
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_YELLOW;
                } else {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                }
                break;
            case ALL_RED_BEFORE_EW_GREEN:
                situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                break;
            case NS_RED_EW_GREEN:
                if (is_NS) {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                } else {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_GREEN_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_GREEN;
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_GREEN;
                }
                break;
            case EW_YELLOW_NS_RED:
                if (is_NS) {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                    situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                } else {
                    situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_YELLOW_YIELD;
                    situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_YELLOW;
                    situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_YELLOW;
                }
                break;
            case ALL_RED_BEFORE_NS_GREEN:
                situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_RIGHT] = free_right ? TRAFFIC_LIGHT_RED_YIELD : TRAFFIC_LIGHT_RED;
                break;
            default:
                situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_RED;
                situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_RED;
                break;
        }
    } else {
        situation->light_for_turn[INDICATOR_LEFT] = TRAFFIC_LIGHT_NONE;
        situation->light_for_turn[INDICATOR_NONE] = TRAFFIC_LIGHT_NONE;
        situation->light_for_turn[INDICATOR_RIGHT] = TRAFFIC_LIGHT_NONE;
    }
    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t1a = _bsa_get_time_us();
    #endif
    situation->is_merge_possible = lane->merges_into_id != ID_NULL;
    situation->is_exit_possible = lane_is_exit_lane_available(lane, lane_progress);
    situation->is_on_entry_ramp = road ? road_is_merge_available(road, map) : false;
    situation->is_exit_available = road ? road_is_exit_road_available(road, map, lane_progress) : false;
    situation->is_exit_available_eventually = road ? road_is_exit_road_eventually_available(road, map, lane_progress) : false;
    situation->is_lane_change_left_possible = situation->lane_left != NULL || situation->is_merge_possible;
    situation->is_lane_change_right_possible = situation->lane_right != NULL || situation->is_exit_possible;
    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t2 = _bsa_get_time_us();
    #endif


    situation->merges_into_lane = lane_get_merge_into(lane, map);
    situation->exit_lane = (situation->is_exit_available || situation->is_exit_available_eventually) ? lane_get_exit_lane(lane, map) : NULL;
    situation->lane_target_for_indicator[INDICATOR_LEFT] = situation->merges_into_lane ? situation->merges_into_lane : situation->lane_left;
    situation->lane_target_for_indicator[INDICATOR_NONE] = situation->lane;
    situation->lane_target_for_indicator[INDICATOR_RIGHT] = (situation->exit_lane && situation->is_exit_possible)  ? situation->exit_lane : situation->lane_right;

    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t2a = _bsa_get_time_us();
    #endif

    if (situation->is_an_intersection_upcoming && situation->intersection->is_T_junction) {
        Direction my_dir = lane->direction;
        bool i_am_merging = (situation->intersection->is_T_junction_north_south && (my_dir == DIRECTION_EAST || my_dir == DIRECTION_WEST)) ||
                            (!situation->intersection->is_T_junction_north_south && (my_dir == DIRECTION_NORTH || my_dir == DIRECTION_SOUTH));
        if (i_am_merging) {
            situation->is_T_junction_exit_available = false;
            situation->is_T_junction_exit_left_available = false;
            situation->is_T_junction_exit_right_available = false;
            // situation->is_T_junction_merge_available = true;
            // situation->is_T_junction_merge_left_available = situation->is_turn_possible[INDICATOR_LEFT];
            // situation->is_T_junction_merge_right_available = situation->is_turn_possible[INDICATOR_RIGHT];
        } else {
            // I am exiting the T-junction
            // situation->is_T_junction_merge_available = false;
            // situation->is_T_junction_merge_left_available = false;
            // situation->is_T_junction_merge_right_available = false;
            situation->is_T_junction_exit_left_available = situation->is_turn_possible[INDICATOR_LEFT];
            situation->is_T_junction_exit_right_available = situation->is_turn_possible[INDICATOR_RIGHT];
            situation->is_T_junction_exit_available = situation->is_T_junction_exit_left_available || situation->is_T_junction_exit_right_available;
        }
    }
    
    situation->is_my_turn_at_stop_sign_fcfs = intersection_upcoming && intersection_upcoming->state == FOUR_WAY_STOP && intersection_upcoming->cars_at_stop_sign_fcfs_queue[0] == car->id;

    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t3 = _bsa_get_time_us();
    #endif

    // Lead vehicle lookup
    Car* all_cars = sim->cars;
    int my_rank = car->lane_rank;
    if (my_rank > 0) {
        // We are not the first car in the lane, so we can find the lead vehicle.
        situation->lead = &all_cars[lane->cars_ids[my_rank - 1]];
    } else {
        // We are the first car in the lane, so there is no lead vehicle, but maybe on next lane.
        const Lane* lane_next = lane_get_connection_straight(lane, map);
        if (!lane_next || lane_next->num_cars == 0) {
            // possible at T-junctions, choose left or right based on which has a closer vehicle
            const Lane* lane_next_left = lane_get_connection_left(lane, map);
            const Lane* lane_next_right = lane_get_connection_right(lane, map);

            if (lane_next_left && lane_next_right) {
                const Car* car_left = (lane_next_left->num_cars > 0) ? &all_cars[lane_next_left->cars_ids[lane_next_left->num_cars - 1]] : NULL;
                const Car* car_right = (lane_next_right->num_cars > 0) ? &all_cars[lane_next_right->cars_ids[lane_next_right->num_cars - 1]] : NULL;
                
                Meters dist_left = car_left ? car_left->lane_progress_meters : __FLT_MAX__;
                Meters dist_right = car_right ? car_right->lane_progress_meters : __FLT_MAX__;

                lane_next = (dist_left < dist_right) ? lane_next_left : lane_next_right;
            } else if (lane_next_left) {
                lane_next = lane_next_left;
            } else if (lane_next_right) {
                lane_next = lane_next_right;
            }
        }
        Car* lane_next_last_vehicle = (lane_next && lane_next->num_cars > 0) ? &all_cars[lane_next->cars_ids[lane_next->num_cars - 1]] : NULL;
        situation->lead = lane_next_last_vehicle;
    }

    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t4 = _bsa_get_time_us();
    #endif

    situation->braking_distance = car_compute_braking_distance(car);

    LOG_TRACE("Situational awareness built for car %d\n", car->id);
    situation->is_valid = true; // Mark situational awareness as valid
    situation->timestamp = sim_get_time(sim); // Set the timestamp of the situational awareness

    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t5 = _bsa_get_time_us();
    _bsa_basic_state_us    += _bsa_t0a - _bsa_t0;
    _bsa_lane_context_us   += _bsa_t1  - _bsa_t0a;
    _bsa_traffic_lights_us += _bsa_t1a - _bsa_t1;
    _bsa_merge_exit_us     += _bsa_t2  - _bsa_t1a;
    _bsa_target_lanes_us   += _bsa_t2a - _bsa_t2;
    _bsa_tjunction_us      += _bsa_t3  - _bsa_t2a;
    _bsa_lead_us           += _bsa_t4  - _bsa_t3;
    _bsa_brakingdist_us    += _bsa_t5  - _bsa_t4;
    _bsa_call_count++;
    if (_bsa_call_count % BSA_INTERVAL == 0) {
        double n = (double)_bsa_call_count;
        double total = (_bsa_basic_state_us + _bsa_lane_context_us + _bsa_traffic_lights_us +
                        _bsa_merge_exit_us + _bsa_target_lanes_us + _bsa_tjunction_us +
                        _bsa_lead_us + _bsa_brakingdist_us) / n;
        printf("[BENCH sa_build] Avg over %d calls (us):\n"
               "  BasicState=%.3f | LaneCtx=%.3f | TrafLights=%.3f | MergeExit=%.3f\n"
               "  TargetLanes=%.3f | TJunc=%.3f\n"
               "  Lead=%.3f | BrakeDist=%.3f\n"
               "  Total=%.3f\n",
               _bsa_call_count,
               _bsa_basic_state_us / n, _bsa_lane_context_us / n,
               _bsa_traffic_lights_us / n, _bsa_merge_exit_us / n,
               _bsa_target_lanes_us / n, _bsa_tjunction_us / n,
               _bsa_lead_us / n, _bsa_brakingdist_us / n,
               total);
    }
    #endif
}
