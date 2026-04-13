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
static double _bsa_init_us = 0;
static double _bsa_connections_us = 0;
static double _bsa_laneinfo_us = 0;
static double _bsa_nearbyveh_us = 0;
static double _bsa_leadfollow_us = 0;
static double _bsa_brakingdist_us = 0;
static int _bsa_call_count = 0;
#define BSA_INTERVAL 500000
#endif

static void _find_vehicle_behind_and_at_and_ahead_of_point(Map* map, Car* all_cars, const Lane* lane, Meters point, Meters at_buffer_half, Car** car_behind_out, Meters* car_behind_distance_out, Car** car_at_out, Meters* car_at_distance_out, Car** car_ahead_out, Meters* car_ahead_distance_out, bool consider_prev_lanes, bool consider_next_lanes) {
    *car_behind_out = NULL;
    *car_behind_distance_out = __FLT_MAX__;
    *car_at_out = NULL;
    *car_at_distance_out = __FLT_MAX__;
    *car_ahead_out = NULL;
    *car_ahead_distance_out = __FLT_MAX__;

    if (!lane) return;

    // Precompute thresholds used in every loop iteration
    const Meters point_plus_half = point + at_buffer_half;
    const Meters point_minus_half = point - at_buffer_half;

    const int num_cars = lane->num_cars;
    for (int i = 0; i < num_cars; i++) {  // traversing from leading vehicle to last vehicle on the lane
        Car* car = &all_cars[lane->cars_ids[i]]; // direct array access, no bounds check (lane car IDs always valid)

        Meters car_position = car->lane_progress_meters;
        Meters car_half_length = car->dimensions.y * 0.5;

        // Logic:
        // if car's rear is ahead of point + at_buffer_half, then it is ahead of the point.
        // if not, if the center-to-center distance is less than sum of half lengths, then it in collision with the at_buffer.
        // if not, if the car's front is behind point - at_buffer_half, then it is behind the point.

        Meters car_rear = car_position - car_half_length;
        if (point_plus_half < car_rear) {
            *car_ahead_out = car;
            *car_ahead_distance_out = car_rear - point_plus_half; // always >= 0 by the condition above
        } else if (car_position + car_half_length < point_minus_half) {
            *car_behind_out = car;
            *car_behind_distance_out = point_minus_half - (car_position + car_half_length); // always >= 0
            break; // since we are traversing from leading vehicle to last vehicle on the lane, we can stop searching here.
        } else {
            if (!*car_at_out) {
                *car_at_out = car;
                *car_at_distance_out = car_position - point;
            }
        }
    }

    // if we did not find a car ahead, then we should pick the trailing car in the next lane in the same direction, if it exists.
    if (!*car_ahead_out && consider_next_lanes) {
        LOG_TRACE("No car ahead found, checking next lane");
        Lane* next_lane = lane_get_connection_straight(lane, map);
        if (!next_lane || next_lane->num_cars == 0) {
            Lane* left_lane = lane_get_connection_left(lane, map);
            Lane* right_lane = lane_get_connection_right(lane, map);
            if (left_lane && right_lane) {
                Meters dist_left = __FLT_MAX__;
                if (left_lane->num_cars > 0) {
                    Car* car = &all_cars[left_lane->cars_ids[left_lane->num_cars - 1]];
                    dist_left = car->lane_progress_meters - car->dimensions.y * 0.5;
                }
                Meters dist_right = __FLT_MAX__;
                if (right_lane->num_cars > 0) {
                    Car* car = &all_cars[right_lane->cars_ids[right_lane->num_cars - 1]];
                    dist_right = car->lane_progress_meters - car->dimensions.y * 0.5;
                }
                next_lane = (dist_left < dist_right) ? left_lane : right_lane;
            } else if (left_lane) {
                next_lane = left_lane;
            } else if (right_lane) {
                next_lane = right_lane;
            }
        }
        if (next_lane && next_lane->num_cars > 0) {
            Car* car = &all_cars[next_lane->cars_ids[next_lane->num_cars - 1]]; // last car in the next lane
            Meters car_half_length = car->dimensions.y * 0.5;
            Meters point_to_end_of_lane = lane->length - point;
            Meters point_to_center_distance = point_to_end_of_lane + car->lane_progress_meters;

            if (point_to_center_distance <= car_half_length + at_buffer_half) {
                if (!*car_at_out) {
                    *car_at_out = car;
                    *car_at_distance_out = point_to_center_distance;
                }
            } else {
                *car_ahead_out = car;
                *car_ahead_distance_out = point_to_center_distance - (car_half_length + at_buffer_half);
            }
        }
    }
    // if we did not find a car behind, then we should pick the leading car in the previous lane in the same direction, if it exists
    if (!*car_behind_out && consider_prev_lanes) {
        LOG_TRACE("No car behind found, checking previous lane");
        Lane* prev_lane = lane_get_connection_incoming_straight(lane, map);
        if (!prev_lane || prev_lane->num_cars == 0) {
            Lane* left_lane = lane_get_connection_incoming_left(lane, map);
            Lane* right_lane = lane_get_connection_incoming_right(lane, map);
            if (left_lane && right_lane) {
                Meters dist_left = __FLT_MAX__;
                if (left_lane->num_cars > 0) {
                    Car* car = &all_cars[left_lane->cars_ids[0]];
                    dist_left = left_lane->length - car->lane_progress_meters - car->dimensions.y * 0.5;
                }
                Meters dist_right = __FLT_MAX__;
                if (right_lane->num_cars > 0) {
                    Car* car = &all_cars[right_lane->cars_ids[0]];
                    dist_right = right_lane->length - car->lane_progress_meters - car->dimensions.y * 0.5;
                }
                prev_lane = (dist_left < dist_right) ? left_lane : right_lane;
            } else if (left_lane) {
                prev_lane = left_lane;
            } else if (right_lane) {
                prev_lane = right_lane;
            }
        }
        if (prev_lane && prev_lane->num_cars > 0) {
            LOG_TRACE("Found previous lane with %d cars", prev_lane->num_cars);
            Car* car = &all_cars[prev_lane->cars_ids[0]]; // first car in the previous lane
            LOG_TRACE("Found car %d in previous lane", car_get_id(car));
            Meters car_half_length = car->dimensions.y * 0.5;
            Meters end_of_prev_lane_to_car_behind = prev_lane->length - car->lane_progress_meters;
            Meters point_to_center_distance = point + end_of_prev_lane_to_car_behind;

            if (point_to_center_distance <= car_half_length + at_buffer_half) {
                LOG_TRACE("Car %d is in collision with the at_buffer", car_get_id(car));
                if (!*car_at_out) {
                    *car_at_out = car;
                    *car_at_distance_out = -point_to_center_distance;
                }
            } else {
                LOG_TRACE("Car %d is behind the point", car_get_id(car));
                *car_behind_out = car;
                *car_behind_distance_out = point_to_center_distance - (car_half_length + at_buffer_half);
            }
        }
    }
}


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


    LOG_TRACE("\nBuilding situational awareness for car %d:", car->id);
    Map* map = sim_get_map(sim);
    Meters car_length = car_get_length(car);
    Meters car_half_length = car_get_length(car) / 2.0; // Half the length of the car
    const Lane* lane = car_get_lane(car, map);
    const Road* road = lane_get_road(lane, map);    // Road we are on. May be NULL if we are already on an intersection.
    const Intersection* intersection = lane_get_intersection(lane, map); // Intersection we are on. May be NULL if we are not on an intersection.
    double lane_progress = car_get_lane_progress(car);
    Meters lane_progress_m = car_get_lane_progress_meters(car);

    MetersPerSecond car_speed = car_get_speed(car);
    situation->stopped = (fabs(car_speed) < STOP_SPEED_THRESHOLD);
    situation->reversing = (car_speed < -STOP_SPEED_THRESHOLD);
    situation->going_forward = (car_speed > STOP_SPEED_THRESHOLD);
    situation->almost_stopped = (fabs(car_speed) < ALMOST_STOP_SPEED_THRESHOLD);
    situation->speed = car_speed;

    Meters lane_length = lane_get_length(lane);
    Direction lane_dir = lane->direction;
    Meters distance_to_end_of_lane = lane_length - lane_progress_m;
    Meters distance_to_end_of_lane_from_leading_edge = distance_to_end_of_lane - car_half_length; // Distance to the end of the lane from the leading edge of the car.
    const Intersection* intersection_upcoming = road ? road_leads_to_intersection(road, map) : NULL;   // may be null if we are already on an intersection or this road does not lead to an intersection.

    situation->is_approaching_dead_end = lane_get_num_connections(lane) == 0;
    if (lane->is_at_intersection) {
        situation->intersection = intersection;
        situation->is_an_intersection_upcoming = false;
        situation->is_on_intersection = true;
        situation->is_stopped_at_intersection = false;
    } else {
        situation->is_on_intersection = false;
        situation->intersection = intersection_upcoming; // may be NULL
        situation->is_an_intersection_upcoming = intersection_upcoming != NULL;
        situation->is_stopped_at_intersection = intersection_upcoming && (fabs(to_mph(car_get_speed(car))) < 0.1) && (distance_to_end_of_lane_from_leading_edge < 50.0);
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
    situation->is_merge_possible = lane_is_merge_available(lane);
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

    if (situation->is_an_intersection_upcoming && situation->intersection->is_T_junction) {
        Direction my_dir = lane->direction;
        bool i_am_merging = (situation->intersection->is_T_junction_north_south && (my_dir == DIRECTION_EAST || my_dir == DIRECTION_WEST)) ||
                            (!situation->intersection->is_T_junction_north_south && (my_dir == DIRECTION_NORTH || my_dir == DIRECTION_SOUTH));
        if (i_am_merging) {
            situation->is_T_junction_exit_available = false;
            situation->is_T_junction_exit_left_available = false;
            situation->is_T_junction_exit_right_available = false;
            situation->is_T_junction_merge_available = true;
            situation->is_T_junction_merge_left_available = situation->is_turn_possible[INDICATOR_LEFT];
            situation->is_T_junction_merge_right_available = situation->is_turn_possible[INDICATOR_RIGHT];
        } else {
            // I am exiting the T-junction
            situation->is_T_junction_merge_available = false;
            situation->is_T_junction_merge_left_available = false;
            situation->is_T_junction_merge_right_available = false;
            situation->is_T_junction_exit_left_available = situation->is_turn_possible[INDICATOR_LEFT];
            situation->is_T_junction_exit_right_available = situation->is_turn_possible[INDICATOR_RIGHT];
            situation->is_T_junction_exit_available = situation->is_T_junction_exit_left_available || situation->is_T_junction_exit_right_available;
        }
    }
    
    situation->is_my_turn_at_stop_sign_fcfs = intersection_upcoming && intersection_upcoming->state == FOUR_WAY_STOP && intersection_upcoming->cars_at_stop_sign_fcfs_queue[0] == car->id;

    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t3 = _bsa_get_time_us();
    #endif

    // build nearby vehicles information:

    Car* all_cars = sim_get_cars(sim);
    Meters car_half_length_for_buffer = car_length * 0.5; // precompute half buffer for _find calls

    // start with what happens on the current lane:
    Car* car_ahead = NULL;
    Meters car_ahead_distance = __FLT_MAX__;
    Car* car_behind = NULL;
    Meters car_behind_distance = __FLT_MAX__;
    Meters car_at_distance = __FLT_MAX__;
    Car* car_at = NULL;
    _find_vehicle_behind_and_at_and_ahead_of_point(map, all_cars, lane, lane_progress_m, car_half_length_for_buffer, &car_behind, &car_behind_distance, &car_at, &car_at_distance, &car_ahead, &car_ahead_distance, true, true);
    situation->nearby_vehicles.ahead[INDICATOR_NONE] = car_ahead;
    situation->nearby_vehicles.behind[INDICATOR_NONE] = car_behind;
    if (car_at == car) {
        car_at = NULL;
    }
    situation->nearby_vehicles.colliding[INDICATOR_NONE] = car_at;
    
    // what happens if we switch lane left:
    const Lane* target_lane_left = situation->lane_target_for_indicator[INDICATOR_LEFT];
    Car* car_ahead_left = NULL;
    Meters car_ahead_distance_left = __FLT_MAX__;
    Car* car_behind_left = NULL;
    Meters car_behind_distance_left = __FLT_MAX__;
    Car* car_at_left = NULL;
    Meters car_at_distance_left = __FLT_MAX__;
    // Inline hypothetical position: avoid redundant merge/exit lane lookups
    Meters hypothetical_position;
    if (!target_lane_left) {
        hypothetical_position = 0;
    } else if (target_lane_left == situation->merges_into_lane) {
        hypothetical_position = lane_progress_m + lane->merges_into_start * target_lane_left->length;
    } else if (target_lane_left == situation->exit_lane) {
        hypothetical_position = (lane_progress - lane->exit_lane_start) * lane->length;
    } else {
        hypothetical_position = lane_progress * target_lane_left->length;
    }
    _find_vehicle_behind_and_at_and_ahead_of_point(map, all_cars, target_lane_left, hypothetical_position, car_half_length_for_buffer, &car_behind_left, &car_behind_distance_left, &car_at_left, &car_at_distance_left, &car_ahead_left, &car_ahead_distance_left, true, true);
    situation->nearby_vehicles.ahead[INDICATOR_LEFT] = car_ahead_left;
    situation->nearby_vehicles.behind[INDICATOR_LEFT] = car_behind_left;
    situation->nearby_vehicles.colliding[INDICATOR_LEFT] = car_at_left;

    // what happens if we switch lane right:
    const Lane* target_lane_right = situation->lane_target_for_indicator[INDICATOR_RIGHT];
    Car* car_ahead_right = NULL;
    Meters car_ahead_distance_right = __FLT_MAX__;
    Car* car_behind_right = NULL;
    Meters car_behind_distance_right = __FLT_MAX__;
    Car* car_at_right = NULL;
    Meters car_at_distance_right = __FLT_MAX__;
    if (!target_lane_right) {
        hypothetical_position = 0;
    } else if (target_lane_right == situation->merges_into_lane) {
        hypothetical_position = lane_progress_m + lane->merges_into_start * target_lane_right->length;
    } else if (target_lane_right == situation->exit_lane) {
        hypothetical_position = (lane_progress - lane->exit_lane_start) * lane->length;
    } else {
        hypothetical_position = lane_progress * target_lane_right->length;
    }
    _find_vehicle_behind_and_at_and_ahead_of_point(map, all_cars, target_lane_right, hypothetical_position, car_half_length_for_buffer, &car_behind_right, &car_behind_distance_right, &car_at_right, &car_at_distance_right, &car_ahead_right, &car_ahead_distance_right, true, true);
    situation->nearby_vehicles.ahead[INDICATOR_RIGHT] = car_ahead_right;
    situation->nearby_vehicles.behind[INDICATOR_RIGHT] = car_behind_right;
    situation->nearby_vehicles.colliding[INDICATOR_RIGHT] = car_at_right;

    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t4 = _bsa_get_time_us();
    #endif

    int my_rank = car->lane_rank;
    if (my_rank > 0) {
        // We are not the first car in the lane, so we can find the lead vehicle.
        situation->nearby_vehicles.lead = &all_cars[lane->cars_ids[my_rank - 1]];
    } else {
        // We are the first car in the lane, so there is no lead vehicle, but maybe on next lane.
        Lane* lane_next = lane_get_connection_straight(lane, map);
        if (!lane_next || lane_next->num_cars == 0) {
            // possible at T-junctions, choose left or right based on which has a closer vehicle
            Lane* lane_next_left = lane_get_connection_left(lane, map);
            Lane* lane_next_right = lane_get_connection_right(lane, map);

            if (lane_next_left && lane_next_right) {
                Car* car_left = (lane_next_left->num_cars > 0) ? &all_cars[lane_next_left->cars_ids[lane_next_left->num_cars - 1]] : NULL;
                Car* car_right = (lane_next_right->num_cars > 0) ? &all_cars[lane_next_right->cars_ids[lane_next_right->num_cars - 1]] : NULL;
                
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
        situation->nearby_vehicles.lead = lane_next_last_vehicle;
    }

    if (my_rank < lane->num_cars - 1) {
        // We are not the last car in the lane, so we can find the following vehicle.
        situation->nearby_vehicles.following = &all_cars[lane->cars_ids[my_rank + 1]];
    } else {
        // We are the last car in the lane, so there is no following vehicle, but maybe on previous lane.
        Lane* lane_prev = lane_get_connection_incoming_straight(lane, map);
        if (!lane_prev || lane_prev->num_cars == 0) {
            Lane* lane_prev_left = lane_get_connection_incoming_left(lane, map);
            Lane* lane_prev_right = lane_get_connection_incoming_right(lane, map);

            if (lane_prev_left && lane_prev_right) {
                Car* car_left = (lane_prev_left->num_cars > 0) ? &all_cars[lane_prev_left->cars_ids[0]] : NULL;
                Car* car_right = (lane_prev_right->num_cars > 0) ? &all_cars[lane_prev_right->cars_ids[0]] : NULL;
                
                Meters dist_left = car_left ? (lane_progress_m + (lane_prev_left->length - car_left->lane_progress_meters)) : __FLT_MAX__;
                Meters dist_right = car_right ? (lane_progress_m + (lane_prev_right->length - car_right->lane_progress_meters)) : __FLT_MAX__;

                lane_prev = (dist_left < dist_right) ? lane_prev_left : lane_prev_right;
            } else if (lane_prev_left) {
                lane_prev = lane_prev_left;
            } else if (lane_prev_right) {
                lane_prev = lane_prev_right;
            }
        }
        Car* lane_prev_first_vehicle = (lane_prev && lane_prev->num_cars > 0) ? &all_cars[lane_prev->cars_ids[0]] : NULL;
        situation->nearby_vehicles.following = lane_prev_first_vehicle;
    }

    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t5 = _bsa_get_time_us();
    #endif

    situation->braking_distance = car_compute_braking_distance(car);

    LOG_TRACE("Situational awareness built for car %d\n", car->id);
    situation->is_valid = true; // Mark situational awareness as valid
    situation->timestamp = sim_get_time(sim); // Set the timestamp of the situational awareness

    #ifdef BENCHMARK_SA_BUILD
    double _bsa_t6 = _bsa_get_time_us();
    _bsa_init_us        += _bsa_t1 - _bsa_t0;
    _bsa_connections_us += _bsa_t2 - _bsa_t1;
    _bsa_laneinfo_us    += _bsa_t3 - _bsa_t2;
    _bsa_nearbyveh_us   += _bsa_t4 - _bsa_t3;
    _bsa_leadfollow_us  += _bsa_t5 - _bsa_t4;
    _bsa_brakingdist_us += _bsa_t6 - _bsa_t5;
    _bsa_call_count++;
    if (_bsa_call_count % BSA_INTERVAL == 0) {
        double n = (double)_bsa_call_count;
        double total = (_bsa_init_us + _bsa_connections_us + _bsa_laneinfo_us + _bsa_nearbyveh_us + _bsa_leadfollow_us + _bsa_brakingdist_us) / n;
        printf("[BENCH sa_build] Avg over %d calls: Init=%.3f us | Connections=%.3f us | LaneInfo=%.3f us | NearbyVeh=%.3f us | LeadFollow=%.3f us | BrakingDist=%.3f us | Total=%.3f us\n",
            _bsa_call_count,
            _bsa_init_us / n,
            _bsa_connections_us / n,
            _bsa_laneinfo_us / n,
            _bsa_nearbyveh_us / n,
            _bsa_leadfollow_us / n,
            _bsa_brakingdist_us / n,
            total);
    }
    #endif
}
