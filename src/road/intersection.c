#include "road.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>


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


Intersection* intersection_create_and_form_connections(
    StraightRoad* road_eastbound_from,
    StraightRoad* road_eastbound_to,
    StraightRoad* road_westbound_from,
    StraightRoad* road_westbound_to,
    StraightRoad* road_northbound_from,
    StraightRoad* road_northbound_to,
    StraightRoad* road_southbound_from,
    StraightRoad* road_southbound_to,
    bool is_four_way_stop,
    bool left_lane_turns_left_only,
    bool right_lane_turns_right_only,
    MetersPerSecond speed_limit,
    double grip
) {
    StraightRoad* roads_from[4] = {
        road_northbound_from, road_eastbound_from,
        road_southbound_from, road_westbound_from
    };
    StraightRoad* roads_to[4] = {
        road_northbound_to, road_eastbound_to,
        road_southbound_to, road_westbound_to
    };

    for (int dir_id = 0; dir_id < 4; dir_id++) {
        assert(roads_from[dir_id]->base.num_lanes == roads_to[dir_id]->base.num_lanes);
        assert(roads_from[dir_id]->width == roads_to[dir_id]->width);
    }

    Intersection* intersection = malloc(sizeof(Intersection));
    if (!intersection) {
        fprintf(stderr, "Memory allocation failed for Intersection\n");
        exit(EXIT_FAILURE);
    }

    intersection->base.id = road_id_counter++;
    snprintf(intersection->base.name, sizeof(intersection->base.name), "Intersection %d", intersection->base.id);
    intersection->base.type = INTERSECTION;
    intersection->road_eastbound_from = road_eastbound_from;
    intersection->road_eastbound_to = road_eastbound_to;
    intersection->road_westbound_from = road_westbound_from;
    intersection->road_westbound_to = road_westbound_to;
    intersection->road_northbound_from = road_northbound_from;
    intersection->road_northbound_to = road_northbound_to;
    intersection->road_southbound_from = road_southbound_from;
    intersection->road_southbound_to = road_southbound_to;
    intersection->left_lane_turns_left_only = left_lane_turns_left_only;
    intersection->right_lane_turns_right_only = right_lane_turns_right_only;
    intersection->base.speed_limit = speed_limit;
    intersection->base.grip = grip;

    if (is_four_way_stop) {
        intersection->state = FOUR_WAY_STOP;
        intersection->countdown = 0;
    } else {
        intersection->state = rand_int_range(0, NUM_TRAFFIC_CONTROL_STATES - 1);
        intersection->countdown = TRAFFIC_STATE_DURATIONS[intersection->state];
    }

    int lanes_count = 0;

    for (int dir_id = 0; dir_id < 4; dir_id++) {
        Road* road_from = (Road*)roads_from[dir_id];

        // Right turn
        Lane* lane_from = (Lane*)road_get_rightmost_lane(road_from);
        Road* road_to = (Road*)roads_to[(dir_id + 1) % 4];
        const Lane* lane_to = road_get_rightmost_lane(road_to);
        Lane* right_turn_lane = (Lane*)quarter_arc_lane_create_from_start_end(
            lane_from->end_point, lane_to->start_point, DIRECTION_CW,
            lane_from->width, speed_limit, grip, DEGRADATIONS_ZERO);
        lane_connect_to_right(lane_from, right_turn_lane);
        lane_connect_to_straight(right_turn_lane, lane_to);
        lane_set_road(right_turn_lane, (Road*)intersection);
        intersection->base.lanes[lanes_count++] = right_turn_lane;

        // Left turn
        lane_from = (Lane*)road_get_leftmost_lane(road_from);
        road_to = (Road*)roads_to[(dir_id + 3) % 4];
        lane_to = road_get_leftmost_lane(road_to);
        Lane* left_turn_lane = (Lane*)quarter_arc_lane_create_from_start_end(
            lane_from->end_point, lane_to->start_point, DIRECTION_CCW,
            lane_from->width, speed_limit, grip, DEGRADATIONS_ZERO);
        lane_connect_to_left(lane_from, left_turn_lane);
        lane_connect_to_straight(left_turn_lane, lane_to);
        lane_set_road(left_turn_lane, (Road*)intersection);
        intersection->base.lanes[lanes_count++] = left_turn_lane;

        // Straight lanes
        for (int k = 0; k < road_from->num_lanes; k++) {
            Road* road_to = (Road*)roads_to[dir_id];

            if (left_lane_turns_left_only && k == 0 && road_from->num_lanes > 1) continue;
            if (right_lane_turns_right_only && k == road_from->num_lanes - 1 && road_from->num_lanes > 1) continue;

            Lane* lane_from = (Lane*)road_from->lanes[k];
            const Lane* lane_to = road_to->lanes[k];

            Lane* straight_lane = (Lane*)linear_lane_create_from_start_end(
                lane_from->end_point, lane_to->start_point,
                lane_from->width, speed_limit, grip, DEGRADATIONS_ZERO);

            lane_connect_to_straight(lane_from, straight_lane);
            lane_connect_to_straight(straight_lane, lane_to);
            lane_set_road(straight_lane, (Road*)intersection);
            intersection->base.lanes[lanes_count++] = straight_lane;
        }
    }

    intersection->base.num_lanes = lanes_count;

    intersection->base.center = vec_midpoint(
        road_northbound_from->end_point, road_northbound_to->start_point);

    Meters gap_vertical = vec_distance(
        road_northbound_from->end_point, road_northbound_to->start_point);
    Meters gap_horizontal = vec_distance(
        road_eastbound_from->end_point, road_eastbound_to->start_point);

    Meters radius_v = (gap_vertical - (road_eastbound_from->width + road_westbound_to->width)) / 2;
    Meters radius_h = (gap_horizontal - (road_northbound_from->width + road_southbound_to->width)) / 2;
    assert(approxeq(radius_v, radius_h, 1e-6));

    intersection->turn_radius = radius_v;
    intersection->dimensions = dimensions_create(gap_horizontal, gap_vertical);

    return intersection;
}

Intersection* intersection_create_from_crossing_roads_and_update_connections(
    StraightRoad* eastbound,
    StraightRoad* westbound,
    StraightRoad* northbound,
    StraightRoad* southbound,
    bool is_four_way_stop,
    Meters turn_radius,
    bool left_lane_turns_left_only,
    bool right_lane_turns_right_only,
    MetersPerSecond speed_limit,
    double grip
) {
    Meters gap_h = (northbound->width + southbound->width) + 2 * turn_radius;
    Meters gap_v = (eastbound->width + westbound->width) + 2 * turn_radius;

    double NS_center_x = (northbound->base.center.x + southbound->base.center.x) / 2;
    StraightRoad* east_to = straight_road_split_at_and_update_connections(eastbound, fabs(NS_center_x - eastbound->start_point.x), gap_h);
    StraightRoad* east_from = eastbound;
    StraightRoad* west_to = straight_road_split_at_and_update_connections(westbound, fabs(NS_center_x - westbound->start_point.x), gap_h);
    StraightRoad* west_from = westbound;

    Lane* east_from_left = (Lane*)east_from->base.lanes[0];
    Lane* west_to_left = (Lane*)west_to->base.lanes[0];
    lane_set_adjacent_left(east_from_left, west_to_left);
    lane_set_adjacent_left(west_to_left, east_from_left);
    Lane* west_from_left = (Lane*)west_from->base.lanes[0];
    Lane* east_to_left = (Lane*)east_to->base.lanes[0];
    lane_set_adjacent_left(west_from_left, east_to_left);
    lane_set_adjacent_left(east_to_left, west_from_left);

    double EW_center_y = (eastbound->base.center.y + westbound->base.center.y) / 2;
    StraightRoad* north_to = straight_road_split_at_and_update_connections(northbound, fabs(EW_center_y - northbound->start_point.y), gap_v);
    StraightRoad* north_from = northbound;
    StraightRoad* south_to = straight_road_split_at_and_update_connections(southbound, fabs(EW_center_y - southbound->start_point.y), gap_v);
    StraightRoad* south_from = southbound;

    Lane* north_from_left = (Lane*)north_from->base.lanes[0];
    Lane* south_to_left = (Lane*)south_to->base.lanes[0];
    lane_set_adjacent_left(north_from_left, south_to_left);
    lane_set_adjacent_left(south_to_left, north_from_left);
    Lane* south_from_left = (Lane*)south_from->base.lanes[0];
    Lane* north_to_left = (Lane*)north_to->base.lanes[0];
    lane_set_adjacent_left(south_from_left, north_to_left);
    lane_set_adjacent_left(north_to_left, south_from_left);

    return intersection_create_and_form_connections(
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

void intersection_update(Intersection* controller, Seconds dt) {
    if (controller->state == FOUR_WAY_STOP) return;
    controller->countdown -= dt;
    if (controller->countdown <= 0) {
        controller->state = (controller->state + 1) % NUM_TRAFFIC_CONTROL_STATES;
        controller->countdown = TRAFFIC_STATE_DURATIONS[controller->state];
    }
}

void intersection_free(Intersection* self) {
    for (int i = 0; i < self->base.num_lanes; i++) {
        lane_free((Lane*)self->base.lanes[i]);
    }
    free(self);
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

