#include "road.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

void straight_road_free(StraightRoad* self) {
    for (int i = 0; i < self->base.num_lanes; i++) {
        lane_free((Lane*)self->base.lanes[i]);
    }
    free(self);
}

StraightRoad* straight_road_create_from_center_dir_len(Coordinates center, Direction direction, Meters length, int num_lanes, Meters lane_width, MetersPerSecond speed_limit, double grip) {
    StraightRoad* road = malloc(sizeof(StraightRoad));
    if (!road) {
        fprintf(stderr, "Memory allocation failed for StraightRoad\n");
        exit(EXIT_FAILURE);
    }

    road->base.type = STRAIGHT;
    road->base.num_lanes = num_lanes;
    road->length = length;
    road->direction = direction;
    road->lane_width = lane_width;
    road->base.speed_limit = speed_limit;
    road->base.grip = grip;
    road->base.center = center;

    LineSegment seg = line_segment_from_center(center, direction_to_vector(direction), length);
    road->start_point = seg.start;
    road->end_point = seg.end;
    road->width = num_lanes * lane_width;

    Vec2D perp_vector = direction_perpendicular_vector(direction);

    for (int i = 0; i < num_lanes; i++) {
        double offset = (((float)num_lanes - 1) / 2 - i) * lane_width;
        Coordinates lane_center = vec_add(center, vec_scale(perp_vector, offset));
        LinearLane* lane = linear_lane_create_from_center_dir_len(lane_center, direction, length, lane_width, speed_limit, grip, DEGRADATIONS_ZERO);
        road->base.lanes[i] = &lane->base;
        lane_set_road(&lane->base, (Road*)road);
    }

    for (int i = 0; i < num_lanes; i++) {
        Lane* lane = (Lane*)road->base.lanes[i];
        const Lane* left = i == 0 ? NULL : road->base.lanes[i - 1];
        const Lane* right = i == num_lanes - 1 ? NULL : road->base.lanes[i + 1];
        lane_set_adjacents(lane, left, right);
    }

    return road;
}

StraightRoad* straight_road_create_from_start_end(Coordinates start, Coordinates end, int num_lanes, Meters lane_width, MetersPerSecond speed_limit, double grip) {
    assert(approxeq(start.x, end.x, 1e-6) || approxeq(start.y, end.y, 1e-6));

    Direction dir = (approxeq(start.x, end.x, 1e-6))
        ? (start.y < end.y ? DIRECTION_NORTH : DIRECTION_SOUTH)
        : (start.x < end.x ? DIRECTION_EAST : DIRECTION_WEST);

    Meters length = vec_distance(start, end);
    Coordinates center = vec_midpoint(start, end);

    return straight_road_create_from_center_dir_len(center, dir, length, num_lanes, lane_width, speed_limit, grip);
}

StraightRoad* straight_road_make_opposite(const StraightRoad* self) {
    Direction opp_dir = direction_opposite(self->direction);
    Meters lane_width = self->lane_width;
    int num_lanes = self->base.num_lanes;

    Vec2D offset;
    switch (self->direction) {
        case DIRECTION_EAST:  offset = vec_create(0,  num_lanes * lane_width); break;
        case DIRECTION_WEST:  offset = vec_create(0, -num_lanes * lane_width); break;
        case DIRECTION_NORTH: offset = vec_create(-num_lanes * lane_width, 0); break;
        case DIRECTION_SOUTH: offset = vec_create( num_lanes * lane_width, 0); break;
        default: offset = vec_create(0, 0); break;
    }

    Coordinates center = vec_add(self->base.center, offset);
    return straight_road_create_from_center_dir_len(center, opp_dir, self->length, num_lanes, lane_width, self->base.speed_limit, self->base.grip);
}

StraightRoad* straight_road_make_opposite_and_set_adjacent_left(StraightRoad* self) {
    StraightRoad* opposite = straight_road_make_opposite(self);

    Lane* a = (Lane*)self->base.lanes[0];
    Lane* b = (Lane*)opposite->base.lanes[0];
    lane_set_adjacent_left(a, b);
    lane_set_adjacent_left(b, a);

    return opposite;
}


StraightRoad* straight_road_split_at_and_update_connections(StraightRoad* road, const Meters position, const Meters gap) {
    if (position <= 0 || position >= road->length) {
        fprintf(stderr, "Position out of bounds\n");
        return NULL;
    }
    if (gap < 0) {
        fprintf(stderr, "Gap must be non-negative\n");
        return NULL;
    }

    Vec2D dir_vector = direction_to_vector(road->direction);
    Coordinates start_point = road->start_point;
    Coordinates end_point = road->end_point;
    Coordinates split_point = vec_add(start_point, vec_scale(dir_vector, position));

    Coordinates from_end = vec_sub(split_point, vec_scale(dir_vector, gap / 2));
    Coordinates to_start = vec_add(split_point, vec_scale(dir_vector, gap / 2));

    // Create second segment
    StraightRoad* second_segment = straight_road_create_from_start_end(
        to_start, end_point, road->base.num_lanes, road->lane_width,
        road->base.speed_limit, road->base.grip);

    // If the road connects to an intersection, inform it of the new segment
    Intersection* intersection = (Intersection*)road_leads_to_intersection((Road*)road);
    if (intersection) {
        if (intersection->road_eastbound_from == road) {
            intersection->road_eastbound_from = second_segment;
        } else if (intersection->road_westbound_from == road) {
            intersection->road_westbound_from = second_segment;
        } else if (intersection->road_northbound_from == road) {
            intersection->road_northbound_from = second_segment;
        } else if (intersection->road_southbound_from == road) {
            intersection->road_southbound_from = second_segment;
        }
    }

    // Update lane connections
    for (int i = 0; i < road->base.num_lanes; i++) {
        Lane* lane = (Lane*)road->base.lanes[i];
        Lane* lane_second = (Lane*)second_segment->base.lanes[i];

        lane_connect_to_left(lane_second, lane->for_left_connects_to);
        lane_connect_to_right(lane_second, lane->for_right_connects_to);
        lane_connect_to_straight(lane_second, lane->for_straight_connects_to);
        lane_set_merges_into(lane_second, lane->merges_into, lane->merges_into_start, lane->merges_into_end);

        lane_connect_to_left(lane, NULL);
        lane_connect_to_right(lane, NULL);
        lane_connect_to_straight(lane, NULL);
        lane_set_merges_into(lane, NULL, 0, 0);
    }

    // Modify original segment's endpoint, center, and lanes
    StraightRoad* first_tmp = straight_road_create_from_start_end(
        start_point, from_end, road->base.num_lanes, road->lane_width,
        road->base.speed_limit, road->base.grip);

    road->base.center = first_tmp->base.center;
    road->end_point = first_tmp->end_point;
    road->length = first_tmp->length;

    for (int i = 0; i < road->base.num_lanes; i++) {
        Lane* lane = (Lane*)road->base.lanes[i];
        Lane* lane_tmp = (Lane*)first_tmp->base.lanes[i];
        lane->center = lane_tmp->center;
        lane->end_point = lane_tmp->end_point;
        lane->length = lane_tmp->length;
        lane->merges_into = NULL;
    }

    free(first_tmp);
    return second_segment;
}
