#include "road.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

Turn* turn_create_and_set_connections_and_adjacents(StraightRoad* from, StraightRoad* to, Direction direction, MetersPerSecond speed_limit, double grip, Turn* opposite_turn) {
    Turn* turn = malloc(sizeof(Turn));
    if (!turn) {
        fprintf(stderr, "Memory allocation failed for Turn\n");
        exit(EXIT_FAILURE);
    }

    turn->base.type = TURN;
    turn->road_from = from;
    turn->road_to = to;
    turn->direction = direction;
    turn->start_point = from->end_point;
    turn->end_point = to->start_point;
    turn->base.speed_limit = speed_limit;
    turn->base.grip = grip;

    assert(from->base.num_lanes == to->base.num_lanes);

    int count = 0;
    for (int i = 0; i < from->base.num_lanes; i++) {
        Lane* from_lane = (Lane*)from->base.lanes[i];
        const Lane* to_lane = to->base.lanes[i];
        Lane* new_lane = (Lane*)quarter_arc_lane_create_from_start_end(from_lane->end_point, to_lane->start_point, direction, from_lane->width, speed_limit, grip, DEGRADATIONS_ZERO);
        lane_connect_to_straight(from_lane, new_lane);
        lane_connect_to_straight(new_lane, to_lane);
        turn->base.lanes[count++] = new_lane;
        lane_set_road(new_lane, (Road*)turn);
    }
    turn->base.num_lanes = count;

    for (int i = 0; i < count; i++) {
        Lane* lane = (Lane*)turn->base.lanes[i];
        const Lane* left = i == 0 ? NULL : turn->base.lanes[i - 1];
        const Lane* right = i == count - 1 ? NULL : turn->base.lanes[i + 1];
        lane_set_adjacents(lane, left, right);
    }

    if (opposite_turn) {
        Lane* a = (Lane*)turn->base.lanes[0];
        Lane* b = (Lane*)opposite_turn->base.lanes[0];
        lane_set_adjacent_left(a, b);
        lane_set_adjacent_left(b, a);
    }

    turn->base.center = vec_midpoint(turn->start_point, turn->end_point);
    return turn;
}

void turn_free(Turn* self) {
    for (int i = 0; i < self->base.num_lanes; i++) {
        lane_free((Lane*)self->base.lanes[i]);
    }
    free(self);
}
