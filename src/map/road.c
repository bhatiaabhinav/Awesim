#include "map.h"
#include "logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h> 
#include <string.h>


//
// Road Setters
//

void road_set_name(Road* self, const char* name) {
    if (name) {
        snprintf(self->name, sizeof(self->name), "%s", name);
    } else {
        self->name[0] = '\0'; // Clear the name if NULL
    }
}

void road_set_speed_limit(Road* self, MetersPerSecond speed_limit) {
    self->speed_limit = speed_limit;
}

void road_set_grip(Road* self, double grip) {
    self->grip = (grip > 0.0) ? ((grip < 1.0) ? grip : 1.0) : 0.0;
}



//
// Road Getters
//

int road_get_id(const Road* self) { return self->id;}

RoadType road_get_type(const Road* self) { return self->type;}

const char* road_get_name(const Road* self) { return self->name; }

int road_get_num_lanes(const Road* self) { return self->num_lanes; }

MetersPerSecond road_get_speed_limit(const Road* self) { return self->speed_limit; }

double road_get_grip(const Road* self) { return self->grip; }

Meters road_get_lane_width(const Road* self) { return self->lane_width; }

Direction road_get_direction(const Road* self) { return self->direction; }

Meters road_get_length(const Road* self) { return self->length; }

Meters road_get_width(const Road* self) { return self->width; }

Coordinates road_get_start_point(const Road* self) { return self->start_point; }

Coordinates road_get_end_point(const Road* self) { return self->end_point; }

Coordinates road_get_mid_point(const Road* self) { return self->mid_point; }

Coordinates road_get_center(const Road* self) {
    return self->center;
}


// Fancier functions


Lane* road_get_lane(const Road* self, Map* map, int index) {
    if (index < 0 || index >= self->num_lanes) {
        return NULL;
    }
    LaneId lane_id = self->lane_ids[index];
    return map_get_lane(map, lane_id);
}

Lane* road_get_leftmost_lane(const Road* self, Map* map) {
    if (self->num_lanes == 0) {
        return NULL;
    }
    LaneId lane_id = self->lane_ids[0];
    return map_get_lane(map, lane_id);
}

Lane* road_get_rightmost_lane(const Road* self, Map* map) {
    if (self->num_lanes == 0) {
        return NULL;
    }
    LaneId lane_id = self->lane_ids[self->num_lanes - 1];
    return map_get_lane(map, lane_id);
}

bool road_is_merge_available(const Road* self, Map* map) {
    const Lane* leftmost_lane = road_get_leftmost_lane(self, map);
    return leftmost_lane && lane_is_merge_available(leftmost_lane);
}

bool road_is_exit_road_available(const Road* self, Map* map, double progress) {
    const Lane* rightmost_lane = road_get_rightmost_lane(self, map);
    return rightmost_lane && lane_is_exit_lane_available(rightmost_lane, progress);
}

bool road_is_exit_road_eventually_available(const Road* self, Map* map, double progress) {
    const Lane* rightmost_lane = road_get_rightmost_lane(self, map);
    return rightmost_lane && lane_is_exit_lane_eventually_available(rightmost_lane, progress);
}

Road* turn_road_get_from(const Road* self, Map* map) {
    if (self->type != TURN) return NULL;
    return map_get_road(map, self->road_from_id);
}

Road* turn_road_get_to(const Road* self, Map* map) {
    if (self->type != TURN) return NULL;
    return map_get_road(map, self->road_to_id);
}

int road_find_index_of_lane(const Road* self, LaneId lane_id) {
    for (int i = 0; i < self->num_lanes; i++) {
        if (self->lane_ids[i] == lane_id) {
            return i;
        }
    }
    return -1;
}



// Road Creaters


Road* straight_road_create_from_center_dir_len(Map* map, Coordinates center, Direction direction, Meters length, int num_lanes, Meters lane_width, MetersPerSecond speed_limit, double grip) {
    Road* road = map_get_new_road(map);
    snprintf(road->name, sizeof(road->name), "Straight Road %d", road->id);
    LOG_TRACE("Creating straight road with ID %d at center (%.2f, %.2f), direction %d, length %.2f, num_lanes %d, lane_width %.2f, speed_limit %.2f, grip %.2f",
              road->id, center.x, center.y, direction, length, num_lanes, lane_width, speed_limit, grip);
    road->type = STRAIGHT;
    road->num_lanes = num_lanes;
    road->length = length;
    road->direction = direction;
    road->lane_width = lane_width;
    road->speed_limit = speed_limit;
    road->grip = grip;
    road->center = center;
    road->radius = 0.0; // Straight roads have no radius
    road->start_angle = 0.0; // Straight roads have no start angle
    road->end_angle = 0.0; // Straight roads have no end angle
    road->quadrant = QUADRANT_TOP_RIGHT; // Default quadrant for straight roads

    LineSegment seg = line_segment_from_center(center, direction_to_vector(direction), length);
    road->start_point = seg.start;
    road->mid_point = center;
    road->end_point = seg.end;
    road->width = num_lanes * lane_width;

    Vec2D perp_vector = direction_perpendicular_vector(direction);

    for (int i = 0; i < num_lanes; i++) {
        double offset = (((float)num_lanes - 1) / 2 - i) * lane_width;
        Coordinates lane_center = vec_add(center, vec_scale(perp_vector, offset));
        Lane* lane = linear_lane_create_from_center_dir_len(map, lane_center, direction, length, lane_width, speed_limit, grip, DEGRADATIONS_ZERO);
        road->lane_ids[i] = lane->id;
        lane_set_road(lane, road);
    }

    LOG_TRACE("Setting up adjacents for %d lanes", num_lanes);
    for (int i = 0; i < num_lanes; i++) {
        Lane* lane = road_get_lane(road, map, i);
        const Lane* left = i == 0 ? NULL : road_get_lane(road, map, i - 1);
        const Lane* right = i == num_lanes - 1 ? NULL : road_get_lane(road, map, i + 1);
        lane_set_adjacents(lane, left, right);
    }

    return road;
}

Road* straight_road_create_from_start_end(Map* map, Coordinates start, Coordinates end, int num_lanes, Meters lane_width, MetersPerSecond speed_limit, double grip) {
    LOG_TRACE("Creating straight road from start (%.2f, %.2f) to end (%.2f, %.2f) with %d lanes, lane width %.2f, speed limit %.2f, grip %.2f",
              start.x, start.y, end.x, end.y, num_lanes, lane_width, speed_limit, grip);
    if (!(approxeq(start.x, end.x, 1e-6) || approxeq(start.y, end.y, 1e-6))) {
        LOG_ERROR("Start and end points must be aligned either horizontally or vertically");
        return NULL;
    }

    Direction dir = (approxeq(start.x, end.x, 1e-6))
        ? (start.y < end.y ? DIRECTION_NORTH : DIRECTION_SOUTH)
        : (start.x < end.x ? DIRECTION_EAST : DIRECTION_WEST);

    Meters length = vec_distance(start, end);
    Coordinates center = vec_midpoint(start, end);
    LOG_TRACE("Direction: %d, Length: %.2f, Center: (%.2f, %.2f)", dir, length, center.x, center.y);

    return straight_road_create_from_center_dir_len(map, center, dir, length, num_lanes, lane_width, speed_limit, grip);
}

Road* straight_road_make_opposite(Map* map, const Road* road) {
    Direction opp_dir = direction_opposite(road->direction);
    Meters lane_width = road->lane_width;
    int num_lanes = road->num_lanes;

    Vec2D offset;
    switch (road->direction) {
        case DIRECTION_EAST:  offset = vec_create(0,  num_lanes * lane_width); break;
        case DIRECTION_WEST:  offset = vec_create(0, -num_lanes * lane_width); break;
        case DIRECTION_NORTH: offset = vec_create(-num_lanes * lane_width, 0); break;
        case DIRECTION_SOUTH: offset = vec_create( num_lanes * lane_width, 0); break;
        default: offset = vec_create(0, 0); break;
    }

    Coordinates center = vec_add(road->center, offset);
    return straight_road_create_from_center_dir_len(map, center, opp_dir, road->length, num_lanes, lane_width, road->speed_limit, road->grip);
}

Road* straight_road_make_opposite_and_update_adjacents(Map* map, Road* road) {
    assert(road->type == STRAIGHT && "Cannot make opposite of a non-straight road");
    Road* opposite = straight_road_make_opposite(map, road);
    Lane* a = road_get_leftmost_lane(road, map);
    Lane* b = road_get_leftmost_lane(opposite, map);
    lane_set_adjacent_left(a, b);
    lane_set_adjacent_left(b, a);
    return opposite;
}


Road* straight_road_split_at_and_update_connections(Road* road, Map* map, const Meters position, const Meters gap) {
    assert(road->type == STRAIGHT && "Cannot split a non-straight road");
    if (position <= 0 || position >= road->length) {
        LOG_ERROR("Position out of bounds");
        return NULL;
    }
    if (gap < 0) {
        LOG_ERROR("Gap must be non-negative");
        return NULL;
    }
    LOG_TRACE("Splitting road '%s' at position %.2f with gap %.2f", road->name, position, gap);

    Vec2D dir_vector = direction_to_vector(road->direction);
    Coordinates start_point = road->start_point;
    Coordinates end_point = road->end_point;
    Coordinates split_point = vec_add(start_point, vec_scale(dir_vector, position));
    LOG_TRACE("Dir vector: (%.2f, %.2f), Start point: (%.2f, %.2f), End point: (%.2f, %.2f), Split point: (%.2f, %.2f)",
              dir_vector.x, dir_vector.y, start_point.x, start_point.y, end_point.x, end_point.y, split_point.x, split_point.y);

    Coordinates from_end = vec_sub(split_point, vec_scale(dir_vector, gap / 2));
    Coordinates to_start = vec_add(split_point, vec_scale(dir_vector, gap / 2));
    LOG_TRACE("From end: (%.2f, %.2f), To start: (%.2f, %.2f)", from_end.x, from_end.y, to_start.x, to_start.y);

    // Create second segment
    Road* second_segment = straight_road_create_from_start_end(map,
        to_start, end_point, road->num_lanes, road->lane_width,
        road->speed_limit, road->grip);
    LOG_TRACE("Created second segment with ID %d, start point (%.2f, %.2f), end point (%.2f, %.2f)",
              second_segment->id, to_start.x, to_start.y, end_point.x, end_point.y);

    // If the road connects to an intersection, inform it of the new segment
    Intersection* intersection = road_leads_to_intersection(road, map);
    if (intersection) {
        LOG_TRACE("The first segment of road '%s' leads to intersection '%s'. Updating intersection connections.",
                  road->name, intersection->name);
        if (intersection->road_eastbound_from_id == road->id) {
            intersection->road_eastbound_from_id = second_segment->id;
        } else if (intersection->road_westbound_from_id == road->id) {
            intersection->road_westbound_from_id = second_segment->id;
        } else if (intersection->road_northbound_from_id == road->id) {
            intersection->road_northbound_from_id = second_segment->id;
        } else if (intersection->road_southbound_from_id == road->id) {
            intersection->road_southbound_from_id = second_segment->id;
        }
    }

    // Update lane connections
    LOG_TRACE("Trasferring lane connections from original segment (%s) to second segment (%s)", road->name, second_segment->name);
    for (int i = 0; i < road->num_lanes; i++) {
        Lane* lane = road_get_lane(road, map, i);
        Lane* lane_second = road_get_lane(second_segment, map, i);
        memcpy(lane_second->connections, lane->connections, sizeof(lane->connections));
        LOG_TRACE("Lane no. %d (id=%d) connections transferred: left %d, straight %d, right %d",
                  i, lane->id, lane_second->connections[0], lane_second->connections[1], lane_second->connections[2]);
        lane_set_merges_into(lane_second, lane_get_merge_into(lane, map), lane->merges_into_start, lane->merges_into_end);
        LOG_TRACE("Lane no. %d (id=%d) transferred merges (into %d from %.2f to %.2f)",
                  i, lane->id, lane_second->merges_into_id, lane_second->merges_into_start, lane_second->merges_into_end);
        
        // Update incoming connections of the lane connected to the original segment
        Lane* lane_next_straight = lane_get_connection_straight(lane, map);
        if (lane_next_straight) {
            lane_set_connection_incoming_straight(lane_next_straight, lane_second);
            LOG_TRACE("Incoming straight lane for lane id %d set to lane no. %d (id=%d)",
                      lane_next_straight->id, i, lane->id);
        }
        Lane* lane_next_right = lane_get_connection_right(lane, map);
        if (lane_next_right) {
            assert(lane_get_connection_incoming_straight(lane_next_right, map) == lane && "Our assumption that the next right lane's straight incoming connection is the original lane is not valid");
            lane_set_connection_incoming_straight(lane_next_right, lane_second);
            LOG_TRACE("Incoming right lane for lane id %d set to lane no. %d (id=%d)",
                      lane_next_right->id, i, lane->id);
        }
        Lane* lane_next_left = lane_get_connection_left(lane, map);
        if (lane_next_left) {
            assert(lane_get_connection_incoming_straight(lane_next_left, map) == lane && "Our assumption that the next left lane's straight incoming connection is the original lane is not valid");
            lane_set_connection_incoming_straight(lane_next_left, lane_second);
            LOG_TRACE("Incoming left lane for lane id %d set to lane no. %d (id=%d)",
                      lane_next_left->id, i, lane->id);
        }

        lane_set_connection_left(lane, NULL);
        lane_set_connection_right(lane, NULL);
        lane_set_connection_straight(lane, NULL);
        lane_set_merges_into(lane, NULL, 0, 0);
    }

    // Modify original segment's endpoint, center, and lanes
    road->center = vec_midpoint(start_point, from_end);
    road->mid_point = road->center;
    road->end_point = from_end;
    road->length = vec_distance(start_point, from_end);
    LOG_TRACE("Updated original segment's end point to (%.2f, %.2f), center to (%.2f, %.2f), length to %.2f",
              from_end.x, from_end.y, road->center.x, road->center.y, road->length);

    for (int i = 0; i < road->num_lanes; i++) {
        Lane* lane = road_get_lane(road, map, i);
        lane->end_point = vec_add(lane->start_point, vec_scale(dir_vector, road->length));
        lane->center = vec_midpoint(lane->start_point, lane->end_point);
        lane->mid_point = lane->center;
        lane->length = road->length;
        lane->merges_into_id = ID_NULL;
        LOG_TRACE("Updated lane no. %d (id = %d) end point to (%.2f, %.2f), center to (%.2f, %.2f), length to %.2f",
                  i, lane->id, lane->end_point.x, lane->end_point.y, lane->center.x, lane->center.y, lane->length);
    }

    return second_segment;
}


Road* turn_create_and_set_connections_and_adjacents(Map* map, Road* from, Road* to, Direction direction, MetersPerSecond speed_limit, double grip, Road* opposite_turn) {

    Road* turn = map_get_new_road(map);
    snprintf(turn->name, sizeof(turn->name), "Turn %d", turn->id);
    turn->type = TURN;
    int num_lanes = from->num_lanes;
    turn->num_lanes = num_lanes;
    turn->speed_limit = speed_limit;
    turn->grip = grip;
    turn->lane_width = from->lane_width;
    turn->direction = direction;
    turn->road_from_id = from->id;
    turn->road_to_id = to->id;


    assert(from->num_lanes == to->num_lanes);

    for (int i = 0; i < from->num_lanes; i++) {
        Lane* from_lane = road_get_lane(from, map, i);
        Lane* to_lane = road_get_lane(to, map, i);
        Lane* new_lane = quarter_arc_lane_create_from_start_end(map, from_lane->end_point, to_lane->start_point, direction, from_lane->width, speed_limit, grip, DEGRADATIONS_ZERO);
        lane_set_connection_straight(from_lane, new_lane);
        lane_set_connection_incoming_straight(new_lane, from_lane);
        lane_set_connection_straight(new_lane, to_lane);
        lane_set_connection_incoming_straight(to_lane, new_lane);
        turn->lane_ids[i] = new_lane->id;
        lane_set_road(new_lane, turn);
    }

    for (int i = 0; i < num_lanes; i++) {
        Lane* lane = road_get_lane(turn, map, i);
        const Lane* left = i == 0 ? NULL : road_get_lane(turn, map, i - 1);
        const Lane* right = i == num_lanes - 1 ? NULL : road_get_lane(turn, map, i + 1);
        lane_set_adjacents(lane, left, right);
    }

    if (opposite_turn) {
        if (opposite_turn->type != TURN) {
            LOG_ERROR("Expected opposite_turn to be of type TURN, but got %d. From road name is '%s', to road name is '%s'. Opposite turn name is '%s'.",
                      opposite_turn->type, from->name, to->name, opposite_turn->name);
            return NULL;
        }
        Lane* a = road_get_lane(turn, map, 0);
        Lane* b = road_get_lane(opposite_turn, map, 0);
        lane_set_adjacent_left(a, b);
        lane_set_adjacent_left(b, a);
    }

    Lane* first_lane = road_get_lane(turn, map, 0);
    Lane* last_lane = road_get_lane(turn, map, num_lanes - 1);

    turn->start_point = from->end_point;
    turn->end_point = to->start_point;
    turn->mid_point = vec_midpoint(first_lane->mid_point, last_lane->mid_point);
    turn->center = first_lane->center;
    turn->radius = vec_distance(turn->center, turn->end_point);
    turn->start_angle = first_lane->start_angle;
    turn->end_angle = first_lane->end_angle;
    turn->quadrant = first_lane->quadrant;
    turn->width = num_lanes * turn->lane_width;
    turn->length = M_PI * turn->radius / 2.0;
    return turn;
}
