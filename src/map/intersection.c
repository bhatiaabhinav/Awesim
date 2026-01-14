#include "map.h"
#include "utils.h"
#include "logging.h"
#include "car.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

const Seconds TRAFFIC_STATE_DURATIONS[NUM_TRAFFIC_CONTROL_CYCLIC_STATES] = {
    GREEN_DURATION,
    YELLOW_DURATION,
    MINOR_RED_EXTENSION,
    GREEN_DURATION,
    YELLOW_DURATION,
    MINOR_RED_EXTENSION,
};

// Setters

void intersection_set_name(Intersection* self, const char* name) {
    if (name) {
        snprintf(self->name, sizeof(self->name), "%s", name);
    } else {
        self->name[0] = '\0'; // Clear the name if NULL
    }
}

void intersection_set_speed_limit(Intersection* self, MetersPerSecond speed_limit) {
    self->speed_limit = speed_limit;
}

void intersection_set_grip(Intersection* self, double grip) {
    self->grip = (grip > 0.0) ? ((grip < 1.0) ? grip : 1.0) : 0.0;
}


// Getters

int intersection_get_id(const Intersection* self) {
    return self->id;
}
const char* intersection_get_name(const Intersection* self) {
    return self->name;
}
int intersection_get_num_lanes(const Intersection* self) {
    return self->num_lanes;
}
Coordinates intersection_get_center(const Intersection* self) {
    return self->center;
}

IntersectionState intersection_get_state(const Intersection* self) {
    return self->state;
}

Seconds intersection_get_countdown(const Intersection* self) {
    return self->countdown;
}

bool intersection_get_left_lane_turns_left_only(const Intersection* self) {
    return self->left_lane_turns_left_only;
}

bool intersection_get_right_lane_turns_right_only(const Intersection* self) {
    return self->right_lane_turns_right_only;
}

Meters intersection_get_turn_radius(const Intersection* self) {
    return self->turn_radius;
}

Dimensions intersection_get_dimensions(const Intersection* self) {
    return self->dimensions;
}

Lane* intersection_get_lane(const Intersection* self, Map* map, int index) {
    if (index < 0 || index >= self->num_lanes) {
        return NULL;
    }
    LaneId lane_id = self->lane_ids[index];
    return map_get_lane(map, lane_id);
}

Road* intersection_get_road_eastbound_from(const Intersection* self, Map* map) {
    return map_get_road(map, self->road_eastbound_from_id);
}

Road* intersection_get_road_eastbound_to(const Intersection* self, Map* map) {
    return map_get_road(map, self->road_eastbound_to_id);
}

Road* intersection_get_road_westbound_from(const Intersection* self, Map* map) {
    return map_get_road(map, self->road_westbound_from_id);
}

Road* intersection_get_road_westbound_to(const Intersection* self, Map* map) {
    return map_get_road(map, self->road_westbound_to_id);
}

Road* intersection_get_road_northbound_from(const Intersection* self, Map* map) {
    return map_get_road(map, self->road_northbound_from_id);
}

Road* intersection_get_road_northbound_to(const Intersection* self, Map* map) {
    return map_get_road(map, self->road_northbound_to_id);
}

Road* intersection_get_road_southbound_from(const Intersection* self, Map* map) {
    return map_get_road(map, self->road_southbound_from_id);
}

Road* intersection_get_road_southbound_to(const Intersection* self, Map* map) {
    return map_get_road(map, self->road_southbound_to_id);
}


// Intersection Creaters


Intersection* intersection_create_and_form_connections(
    Map* map,
    Road* road_eastbound_from,
    Road* road_eastbound_to,
    Road* road_westbound_from,
    Road* road_westbound_to,
    Road* road_northbound_from,
    Road* road_northbound_to,
    Road* road_southbound_from,
    Road* road_southbound_to,
    bool is_four_way_stop,
    bool left_lane_turns_left_only,
    bool right_lane_turns_right_only,
    MetersPerSecond speed_limit,
    double grip
) {
    LOG_TRACE("Creating intersection with roads names: "
              "%s (from), %s (to), %s (from), %s (to), "
              "%s (from), %s (to), %s (from), %s (to), "
              "Parameters: is_four_way_stop=%d (may be disabled for T-junctions), left_lane_turns_left_only=%d, right_lane_turns_right_only=%d, speed_limit=%.2f, grip=%.2f",
              road_eastbound_from->name, road_eastbound_to->name,
              road_westbound_from->name, road_westbound_to->name,
              road_northbound_from->name, road_northbound_to->name,
              road_southbound_from->name, road_southbound_to->name,
              is_four_way_stop, left_lane_turns_left_only, right_lane_turns_right_only, speed_limit, grip);
    Road* roads_from[4] = {
        [DIRECTION_EAST] = road_eastbound_from,
        [DIRECTION_WEST] = road_westbound_from,
        [DIRECTION_NORTH] = road_northbound_from,
        [DIRECTION_SOUTH] = road_southbound_from,
    };

    Road* roads_to[4] = {
        [DIRECTION_EAST] = road_eastbound_to,
        [DIRECTION_WEST] = road_westbound_to,
        [DIRECTION_NORTH] = road_northbound_to,
        [DIRECTION_SOUTH] = road_southbound_to,
    };

    for (int dir_id = 0; dir_id < 4; dir_id++) {
        if (roads_from[dir_id]->num_lanes != roads_to[dir_id]->num_lanes) {
            LOG_ERROR("Roads from and to must have the same number of lanes. Roads are %s (from) and %s (to).",
                      roads_from[dir_id]->name, roads_to[dir_id]->name);
            return NULL;
        }
        if (roads_from[dir_id]->width != roads_to[dir_id]->width) {
            LOG_ERROR("Roads from and to must have the same width. Roads are %s (from) and %s (to).",
                      roads_from[dir_id]->name, roads_to[dir_id]->name);
            return NULL;
        }
    }

    Intersection* intersection = map_get_new_intersection(map);
    snprintf(intersection->name, sizeof(intersection->name), "Intersection %d", intersection->id);
    intersection->road_eastbound_from_id = road_eastbound_from->id;
    intersection->road_eastbound_to_id = road_eastbound_to->id;
    intersection->road_westbound_from_id = road_westbound_from->id;
    intersection->road_westbound_to_id = road_westbound_to->id;
    intersection->road_northbound_from_id = road_northbound_from->id;
    intersection->road_northbound_to_id = road_northbound_to->id;
    intersection->road_southbound_from_id = road_southbound_from->id;
    intersection->road_southbound_to_id = road_southbound_to->id;
    intersection->left_lane_turns_left_only = left_lane_turns_left_only;
    intersection->right_lane_turns_right_only = right_lane_turns_right_only;
    intersection->speed_limit = speed_limit;
    intersection->grip = grip;
    intersection->lane_width = road_northbound_from->lane_width; // Assuming all roads have the same lane width

    for (int dir_id = 0; dir_id < 4; dir_id++) {
        if (roads_from[dir_id]->num_lanes > MAX_NUM_LANES_PER_ROAD) {
            LOG_ERROR("Road %s has %d lanes, which exceeds the maximum of %d supported by intersections.", roads_from[dir_id]->name, roads_from[dir_id]->num_lanes, MAX_NUM_LANES_PER_ROAD);
        }
    }

    // If any one road (and its adjacent opposite) is insignificantly short, then it's a T-junction
    intersection->is_T_junction = false;
    intersection->is_T_junction_north_south = false;
    for (int dir_id = 0; dir_id < 4; dir_id++) {
        if (road_get_length(roads_from[dir_id]) <= LANE_LENGTH_EPSILON) {
            // assert that adjacent opposite road is also short
            if (road_get_length(roads_to[direction_opposite(dir_id)]) > LANE_LENGTH_EPSILON) {
                LOG_ERROR("T-junction detected but the parallel opposite direction road is not short enough. Roads are %s (from) and %s (to).",
                          roads_from[dir_id]->name, roads_to[direction_opposite(dir_id)]->name);
                return NULL;
            }
            LOG_TRACE("T-junction detected due to short road %s (from) and %s (to).",
                      roads_from[dir_id]->name, roads_to[direction_opposite(dir_id)]->name);
            intersection->is_T_junction = true;
            intersection->is_T_junction_north_south = (dir_id == DIRECTION_EAST || dir_id == DIRECTION_WEST); // East-West road is short
            if (is_four_way_stop) {
                LOG_TRACE("Disabling four-way stop for T-junction.");
                is_four_way_stop = false;
            }
            break;
        }
    }

    if (is_four_way_stop) {
        // assert that all roads are one-lane
        for (int dir_id = 0; dir_id < 4; dir_id++) {
            if (roads_from[dir_id]->num_lanes != 1) {
                LOG_ERROR("Four-way stop intersections must have exactly one lane in each direction. Road %s has %d lanes.",
                          roads_from[dir_id]->name, roads_from[dir_id]->num_lanes);
                return NULL;
            }
        }
        intersection->state = FOUR_WAY_STOP;
        intersection->countdown = 0;
    } else {
        intersection->state = rand_int_range(0, NUM_TRAFFIC_CONTROL_CYCLIC_STATES - 1);
        intersection->countdown = TRAFFIC_STATE_DURATIONS[intersection->state];
    }
    for (int i = 0; i < 4; i++) {
        intersection->cars_at_stop_sign_fcfs_queue[i] = ID_NULL;
    }
    LOG_TRACE("Intersection initialized with ID %d, name '%s', state %d, countdown %.2f. Forming lane connections.",
              intersection->id, intersection->name, intersection->state, intersection->countdown);

    int lanes_count = 0;
    Meters turn_radius = 1e6; 

    for (int dir_id = 0; dir_id < 4; dir_id++) {
        Road* road_from = roads_from[dir_id];

        // Right turn
        Lane* lane_from = road_get_rightmost_lane(road_from, map);
        if (lane_get_length(lane_from) <= LANE_LENGTH_EPSILON) {
            lane_from = NULL;
        }
        Road* road_to = roads_to[direction_perp_cw(dir_id)];
        Lane* lane_to = road_get_rightmost_lane(road_to, map);
        if (lane_get_length(lane_to) <= LANE_LENGTH_EPSILON) {
            lane_to = NULL;
        }
        if (lane_from && lane_to) {
            LOG_TRACE("Creating right turn lane from %s (Road %s) to %s (Road %s)",
                    lane_from->name, road_from->name, lane_to->name, road_to->name);
            Lane* right_turn_lane = quarter_arc_lane_create_from_start_end(map, 
                lane_from->end_point, lane_to->start_point, DIRECTION_CW,
                lane_from->width, speed_limit, grip, DEGRADATIONS_ZERO);
            lane_set_connection_right(lane_from, right_turn_lane);
            lane_set_connection_straight(right_turn_lane, lane_to);
            lane_set_connection_incoming_right(lane_to, right_turn_lane);
            lane_set_connection_incoming_straight(right_turn_lane, lane_from);
            lane_set_intersection(right_turn_lane, intersection);
            intersection->lane_ids[lanes_count++] = right_turn_lane->id;
            turn_radius = fmin(turn_radius, right_turn_lane->radius);
        }

        // Left turn
        lane_from = road_get_leftmost_lane(road_from, map);
        if (lane_get_length(lane_from) <= LANE_LENGTH_EPSILON) {
            lane_from = NULL;
        }
        road_to = roads_to[direction_perp_ccw(dir_id)];
        lane_to = road_get_leftmost_lane(road_to, map);
        if (lane_get_length(lane_to) <= LANE_LENGTH_EPSILON) {
            lane_to = NULL;
        }
        if (lane_from && lane_to) {
            LOG_TRACE("Creating left turn lane from %s (Road %s) to %s (Road %s)",
                    lane_from->name, road_from->name, lane_to->name, road_to->name);
            Lane* left_turn_lane = quarter_arc_lane_create_from_start_end(map, 
                lane_from->end_point, lane_to->start_point, DIRECTION_CCW,
                lane_from->width, speed_limit, grip, DEGRADATIONS_ZERO);
            lane_set_connection_left(lane_from, left_turn_lane);
            lane_set_connection_straight(left_turn_lane, lane_to);
            lane_set_connection_incoming_left(lane_to, left_turn_lane);
            lane_set_connection_incoming_straight(left_turn_lane, lane_from);
            lane_set_intersection(left_turn_lane, intersection);
            intersection->lane_ids[lanes_count++] = left_turn_lane->id;
            turn_radius = fmin(turn_radius, left_turn_lane->radius);
        }

        // Straight lanes
        road_to = roads_to[dir_id];
        LOG_TRACE("Creating straight lanes for direction %d from %s (Road %s) to %s (Road %s)",
                  dir_id, road_from->name, road_from->name, road_to->name, road_to->name);
        for (int k = 0; k < road_from->num_lanes; k++) {
            if (left_lane_turns_left_only && k == 0 && road_from->num_lanes > 1) continue;
            if (right_lane_turns_right_only && k == road_from->num_lanes - 1 && road_from->num_lanes > 1) continue;

            Lane* lane_from = road_get_lane(road_from, map, k);
            if (lane_get_length(lane_from) <= LANE_LENGTH_EPSILON) {
                continue;
            }
            Lane* lane_to = road_get_lane(road_to, map, k);
            if (lane_get_length(lane_to) <= LANE_LENGTH_EPSILON) {
                continue;
            }
            LOG_TRACE("Creating straight lane no. %d from %s (Road %s) to %s (Road %s)",
                k, lane_from->name, road_from->name, lane_to->name, road_to->name);
            Lane* straight_lane = linear_lane_create_from_start_end(map, 
                lane_from->end_point, lane_to->start_point,
                lane_from->width, speed_limit, grip, DEGRADATIONS_ZERO);

            lane_set_connection_straight(lane_from, straight_lane);
            lane_set_connection_straight(straight_lane, lane_to);
            lane_set_connection_incoming_straight(lane_to, straight_lane);
            lane_set_connection_incoming_straight(straight_lane, lane_from);
            lane_set_intersection(straight_lane, intersection);
            intersection->lane_ids[lanes_count++] = straight_lane->id;
        }
    }

    intersection->num_lanes = lanes_count;

    intersection->center = vec_midpoint(vec_midpoint(
        road_northbound_from->end_point, road_northbound_to->start_point), vec_midpoint(
        road_southbound_from->end_point, road_southbound_to->start_point));

    Meters gap_vertical = vec_distance(
        road_northbound_from->end_point, road_northbound_to->start_point);
    Meters gap_horizontal = vec_distance(
        road_eastbound_from->end_point, road_eastbound_to->start_point);

    intersection->turn_radius = turn_radius;
    intersection->dimensions = dimensions_create(gap_horizontal, gap_vertical);

    return intersection;
}

Intersection* intersection_create_from_crossing_roads_and_update_connections(
    Map* map,
    Road* eastbound,
    Road* westbound,
    Road* northbound,
    Road* southbound,
    bool is_four_way_stop,
    Meters turn_radius,
    bool left_lane_turns_left_only,
    bool right_lane_turns_right_only,
    MetersPerSecond speed_limit,
    double grip
) {
    LOG_TRACE("Creating intersection from crossing roads: %s, %s, %s, %s",
        eastbound->name, westbound->name, northbound->name, southbound->name);
    Meters gap_h = (northbound->width + southbound->width) + 2 * turn_radius;
    Meters gap_v = (eastbound->width + westbound->width) + 2 * turn_radius;
    LOG_TRACE("Gap horizontal: %.2f, Gap vertical: %.2f", gap_h, gap_v);

    double NS_center_x = (northbound->center.x + southbound->center.x) / 2;
    LOG_TRACE("Splitting eastbound road at NS center x: %.2f with gap_h: %.2f", NS_center_x, gap_h);
    Road* east_to = straight_road_split_at_and_update_connections(eastbound, map, fabs(NS_center_x - eastbound->start_point.x), gap_h);
    Road* east_from = eastbound;
    LOG_TRACE("Eastbound from: %s, Eastbound to: %s", east_from->name, east_to->name);
    LOG_TRACE("Splitting westbound road at NS center x: %.2f with gap_h: %.2f", NS_center_x, gap_h);
    Road* west_to = straight_road_split_at_and_update_connections(westbound, map, fabs(NS_center_x - westbound->start_point.x), gap_h);
    Road* west_from = westbound;
    LOG_TRACE("Westbound from: %s, Westbound to: %s", west_from->name, west_to->name);

    Lane* east_from_left = road_get_leftmost_lane(east_from, map);
    Lane* west_to_left = road_get_leftmost_lane(west_to, map);
    lane_set_adjacent_left(east_from_left, west_to_left);
    lane_set_adjacent_left(west_to_left, east_from_left);
    Lane* west_from_left = road_get_leftmost_lane(west_from, map);
    Lane* east_to_left = road_get_leftmost_lane(east_to, map);
    lane_set_adjacent_left(west_from_left, east_to_left);
    lane_set_adjacent_left(east_to_left, west_from_left);
    LOG_TRACE("Updated lane adjacencies: East from left to West to left, West from left to East to left");

    double EW_center_y = (eastbound->center.y + westbound->center.y) / 2;
    LOG_TRACE("Splitting northbound road at EW center y: %.2f with gap_v: %.2f", EW_center_y, gap_v);
    Road* north_to = straight_road_split_at_and_update_connections(northbound, map, fabs(EW_center_y - northbound->start_point.y), gap_v);
    Road* north_from = northbound;
    LOG_TRACE("Northbound from: %s, Northbound to: %s", north_from->name, north_to->name);
    LOG_TRACE("Splitting southbound road at EW center y: %.2f with gap_v: %.2f", EW_center_y, gap_v);
    Road* south_to = straight_road_split_at_and_update_connections(southbound, map, fabs(EW_center_y - southbound->start_point.y), gap_v);
    Road* south_from = southbound;
    LOG_TRACE("Southbound from: %s, Southbound to: %s", south_from->name, south_to->name);

    Lane* north_from_left = road_get_leftmost_lane(north_from, map);
    Lane* south_to_left = road_get_leftmost_lane(south_to, map);
    lane_set_adjacent_left(north_from_left, south_to_left);
    lane_set_adjacent_left(south_to_left, north_from_left);
    Lane* south_from_left = road_get_leftmost_lane(south_from, map);
    Lane* north_to_left = road_get_leftmost_lane(north_to, map);
    lane_set_adjacent_left(south_from_left, north_to_left);
    lane_set_adjacent_left(north_to_left, south_from_left);
    LOG_TRACE("Updated lane adjacencies: North from left to South to left, South from left to North to left");

    LOG_TRACE("Calling intersection_create_and_form_connections with roads names: "
              "E from %s to %s, "
              "W from %s to %s, "
              "N from %s to %s, "
              "S from %s to %s",
              east_from->name, east_to->name,
              west_from->name, west_to->name,
              north_from->name, north_to->name,
              south_from->name, south_to->name);

    return intersection_create_and_form_connections(
        map,
        east_from, east_to,
        west_from, west_to,
        north_from, north_to,
        south_from, south_to,
        is_four_way_stop,
        left_lane_turns_left_only,
        right_lane_turns_right_only,
        speed_limit, grip
    );
}

void intersection_update(Intersection* controller, Simulation* sim, Seconds dt) {
    if (controller->is_T_junction) {
        // For T-junctions, orientation direction is always green and the other is always stop and go-yeild
        if (controller->is_T_junction_north_south) {
            controller->state = T_JUNC_NS;
        } else {
            controller->state = T_JUNC_EW;
        }
        return;
    }
    if (controller->state == FOUR_WAY_STOP) {
        // track the first-come-first-serve queue here. A car has reached an intersection if it the distance from its leading edge to the end of the lane is less than STOP_LINE_BUFFER_METERS + meters(0.1).
        Map* map = sim_get_map(sim);
        
        RoadId inbound_road_ids[4] = {
            controller->road_eastbound_from_id,
            controller->road_westbound_from_id,
            controller->road_northbound_from_id,
            controller->road_southbound_from_id
        };

        // 1. Clean up queue: remove cars that are no longer waiting at the stop sign of this intersection
        for (int i = 0; i < 4; i++) {
            CarId queued_car_id = controller->cars_at_stop_sign_fcfs_queue[i];
            if (queued_car_id == ID_NULL) continue;
            
            Car* car = sim_get_car(sim, queued_car_id);
            bool should_remove = true;
            
            if (car) {
                // Check if car is still on one of the inbound lanes
                for (int dir = 0; dir < 4; dir++) {
                    Road* road = map_get_road(map, inbound_road_ids[dir]);
                    if (!road) continue;

                    for (int l = 0; l < road->num_lanes; l++) {
                        if (car->lane_id == road->lane_ids[l]) {
                            should_remove = false; // Still on an inbound lane
                            break;
                        }
                    }
                    if (!should_remove) break;
                }
            } else {
                // Car disappeared (e.g. removed from sim)
                should_remove = true;
            }

            if (should_remove) {
                // Shift remaining queue
                for (int j = i; j < 3; j++) {
                    controller->cars_at_stop_sign_fcfs_queue[j] = controller->cars_at_stop_sign_fcfs_queue[j+1];
                }
                controller->cars_at_stop_sign_fcfs_queue[3] = ID_NULL;
                i--; // re-check this slot as it's now filled by the next item
            }
        }

        // 2. Add new arrivals to queue
        for (int dir_id = 0; dir_id < 4; dir_id++) {
            Road* road = map_get_road(map, inbound_road_ids[dir_id]);
            if (!road) continue;

            // For 4-way stop, we assume 1 lane per direction.
            if (road->num_lanes > 0) {
                LaneId lane_id = road->lane_ids[0];
                Lane* lane = map_get_lane(map, lane_id);
                if (lane && lane->num_cars > 0) {
                    CarId car_id = lane->cars_ids[0]; // foremost car
                    Car* car = sim_get_car(sim, car_id);
                    if (car) {
                        Meters distance = lane->length - car->lane_progress_meters - (car_get_length(car) / 2.0);
                        if (distance < STOP_LINE_BUFFER_METERS + meters(0.1)) {
                            // Car is at stop sign. Check if it's already in queue.
                            bool already_queued = false;
                            for (int k = 0; k < 4; k++) {
                                if (controller->cars_at_stop_sign_fcfs_queue[k] == car_id) {
                                    already_queued = true;
                                    break;
                                }
                            }
                            
                            if (!already_queued) {
                                // Add to the first empty slot
                                // LOG_DEBUG("Adding car ID %d to four-way stop queue of intersection ID %d, which is on lane ID %d at distance %.2f meters", car_id, controller->id, lane_id, distance);
                                for (int k = 0; k < 4; k++) {
                                    if (controller->cars_at_stop_sign_fcfs_queue[k] == ID_NULL) {
                                        controller->cars_at_stop_sign_fcfs_queue[k] = car_id;
                                        break;
                                }
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }
    controller->countdown -= dt;
    if (controller->countdown <= 0) {
        controller->state = (controller->state + 1) % NUM_TRAFFIC_CONTROL_CYCLIC_STATES;
        controller->countdown = TRAFFIC_STATE_DURATIONS[controller->state];
    }
}


void print_traffic_state(const Intersection* intersection) {
    const char* state_names[] = {
        "NS: Green, EW: Red",
        "NS: Yellow, EW: Red",
        "All: Red (before EW Green)",
        "NS: Red, EW: Green",
        "NS: Red, EW: Yellow",
        "All: Red (before NS Green)"
    };
    printf("%s, Countdown: %.2f\n",
           state_names[intersection->state],
           intersection->countdown);
}



Intersection* road_leads_to_intersection(const Road* road, Map* map) {
    if (!road) {
        LOG_ERROR("Road is NULL, cannot determine upcoming intersection.");
        return NULL;
    }
    const Lane* lane = road_get_leftmost_lane(road, map);
    for (int i = 0; i < 3; i++) {
        if (lane->connections[i] != ID_NULL) {
            Lane* connected_lane = map_get_lane(map, lane->connections[i]);
            if (connected_lane->is_at_intersection) {
                return map_get_intersection(map, connected_lane->intersection_id);
            }
        }
    }
    return NULL;
}

Intersection* road_comes_from_intersection(const Road* road, Map* map) {
    if (!road) {
        LOG_ERROR("Road is NULL, cannot determine originating intersection.");
        return NULL;
    }
    const Lane* lane = road_get_leftmost_lane(road, map);
    for (int i = 0; i < 3; i++) {
        if (lane->connections_incoming[i] != ID_NULL) {
            Lane* incoming_lane = map_get_lane(map, lane->connections_incoming[i]);
            if (incoming_lane->is_at_intersection) {
                return map_get_intersection(map, incoming_lane->intersection_id);
            }
        }
    }
    return NULL;
}


Car* intersection_get_foremost_vehicle(const Intersection* self, Simulation* sim) {
    Map* map = sim_get_map(sim);
    Car* foremost_car = NULL;
    double min_distance = 1e9; // Initialize with a large value

    RoadId incoming_road_ids[] = {
        self->road_eastbound_from_id,
        self->road_westbound_from_id,
        self->road_northbound_from_id,
        self->road_southbound_from_id
    };

    for (int i = 0; i < 4; i++) {
        RoadId road_id = incoming_road_ids[i];
        if (road_id == ID_NULL) continue;

        Road* road = map_get_road(map, road_id);
        if (!road) continue;

        for (int k = 0; k < road->num_lanes; k++) {
            LaneId lane_id = road->lane_ids[k];
            Lane* lane = map_get_lane(map, lane_id);
            if (!lane || lane->num_cars == 0) continue;

            // The cars are sorted by progress, so the first car (index 0) is the closest to the end of the lane.
            CarId car_id = lane->cars_ids[0];
            Car* car = sim_get_car(sim, car_id);
            if (!car) continue;

            // Distance to the end of the lane (which connects to the intersection)
            // adjusted for half car length to account for the leading edge.
            double distance = lane->length - car_get_lane_progress_meters(car) - (car_get_length(car) / 2.0);

            if (distance < min_distance) {
                min_distance = distance;
                foremost_car = car;
            }
        }
    }

    return foremost_car;
}


// Checks where there is any car *on* the intersection (i.e., in any of its lanes)
bool intersection_is_any_car_on_intersection(const Intersection* self, Simulation* sim) {
    Map* map = sim_get_map(sim);

    for (int i = 0; i < self->num_lanes; i++) {
        LaneId lane_id = self->lane_ids[i];
        Lane* lane = map_get_lane(map, lane_id);
        if (!lane) continue;

        if (lane->num_cars > 0) {
            return true; // Found at least one car on this intersection lane
        }
    }

    return false; // No cars found on any intersection lanes
}