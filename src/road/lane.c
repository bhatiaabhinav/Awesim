#include "road.h"
#include "utils.h"
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>


//
// Lane Core Initialization
//

void lane_init(Lane* lane, const LaneType type, const Direction direction, const Meters width, const MetersPerSecond speed_limit, const double grip, const Degradations degradations) {
    lane->type = type;
    lane->direction = direction;
    lane->width = width > 0.0 ? width : 0.0;
    lane->speed_limit = speed_limit;
    lane->grip = grip > 0.0 ? (grip < 1.0 ? grip : 1.0) : 0.0;
    lane->degradations = degradations;

    lane->for_straight_connects_to = NULL;
    lane->for_left_connects_to = NULL;
    lane->for_right_connects_to = NULL;
    lane->adjacent_left = NULL;
    lane->adjacent_right = NULL;
    lane->merges_into = NULL;
    lane->merges_into_start = 0.0;
    lane->merges_into_end = 0.0;
    lane->exit_lane = NULL;
    lane->exit_lane_start = 0.0;
    lane->exit_lane_end = 0.0;

    lane->num_cars = 0;
    lane->road = NULL;

    lane->start_point = vec_create(0.0, 0.0);
    lane->end_point = vec_create(0.0, 0.0);
    lane->center = vec_create(0.0, 0.0);
    lane->length = 0.0;
}

//
// Lane Connections
//


void lane_connect_to_straight(Lane* self, const Lane* lane) {
    if (!lane) return;
    assert(vec_approx_equal(self->end_point, lane->start_point, 1e-6));
    self->for_straight_connects_to = lane;
}

void lane_connect_to_left(Lane* self, const Lane* lane) {
    if (!lane) return;
    assert(vec_approx_equal(self->end_point, lane->start_point, 1e-6));
    self->for_left_connects_to = lane;
}

void lane_connect_to_right(Lane* self, const Lane* lane) {
    if (!lane) return;
    assert(vec_approx_equal(self->end_point, lane->start_point, 1e-6));
    self->for_right_connects_to = lane;
}

void lane_set_adjacents(Lane* self, const Lane* left, const Lane* right) {
    self->adjacent_left = left;
    self->adjacent_right = right;
}

void lane_set_adjacent_left(Lane* self, const Lane* left) {
    self->adjacent_left = left;
}

void lane_set_adjacent_right(Lane* self, const Lane* right) {
    self->adjacent_right = right;
}

void lane_set_merges_into(Lane* self, const Lane* lane, const double start, const double end) {
    self->merges_into = lane;
    self->merges_into_start = start;
    self->merges_into_end = end;
}

void lane_set_exit_lane(Lane* self, const Lane* lane, const double start, const double end) {
    self->exit_lane = lane;
    self->exit_lane_start = start;
    self->exit_lane_end = end;
}

//
// Lane Queries
//

int lane_num_connections(const Lane* self) {
    int count = 0;
    if (self->for_straight_connects_to) count++;
    if (self->for_left_connects_to) count++;
    if (self->for_right_connects_to) count++;
    return count;
}

const Lane* first_connection(const Lane* self) {
    if (self->for_straight_connects_to) return self->for_straight_connects_to;
    if (self->for_left_connects_to) return self->for_left_connects_to;
    if (self->for_right_connects_to) return self->for_right_connects_to;
    return NULL;
}

const Lane* last_connection(const Lane* self) {
    if (self->for_right_connects_to) return self->for_right_connects_to;
    if (self->for_left_connects_to) return self->for_left_connects_to;
    if (self->for_straight_connects_to) return self->for_straight_connects_to;
    return NULL;
}

//
// Lane Car Management
//

void lane_add_car(Lane* self, const Car* car) {
    if (self->num_cars < MAX_CARS_PER_LANE) {
        self->cars[self->num_cars++] = car;
    } else {
        fprintf(stderr, "Lane is full\n");
    }
}

void lane_remove_car(Lane* self, const Car* car) {
    for (int i = 0; i < self->num_cars; i++) {
        if (self->cars[i] == car) {
            for (int j = i; j < self->num_cars - 1; j++) {
                self->cars[j] = self->cars[j + 1];
            }
            self->num_cars--;
            return;
        }
    }
    fprintf(stderr, "Car not found in lane\n");
}

//
// Road Link
//

void lane_set_road(Lane* self, const Road* road) {
    self->road = road;
}

//
// Merge / Exit Checks
//

bool lane_check_merge_available(const Lane* self) {
    return self->merges_into != NULL;
}

bool lane_check_exit_lane_available(const Lane* self, const double progress) {
    return self->exit_lane != NULL &&
           self->exit_lane_start <= progress &&
           progress <= self->exit_lane_end;
}

//
// Type-based Freeing
//

void lane_free(Lane* self) {
    switch (self->type) {
        case LINEAR_LANE:
            linear_lane_free((LinearLane*)self);
            break;
        case QUARTER_ARC_LANE:
            quarter_arc_lane_free((QuarterArcLane*)self);
            break;
        default:
            fprintf(stderr, "Unknown lane type\n");
            exit(1);
    }
}

//
// Getters
//

LaneType lane_get_type(const Lane* self) { return self->type; }
Direction lane_get_direction(const Lane* self) { return self->direction; }
Meters lane_get_width(const Lane* self) { return self->width; }
MetersPerSecond lane_get_speed_limit(const Lane* self) { return self->speed_limit; }
double lane_get_grip(const Lane* self) { return self->grip; }
Coordinates lane_get_start_point(const Lane* self) { return self->start_point; }
Coordinates lane_get_end_point(const Lane* self) { return self->end_point; }
Coordinates lane_get_center(const Lane* self) { return self->center; }
Meters lane_get_length(const Lane* self) { return self->length; }
int lane_get_num_cars(const Lane* self) { return self->num_cars; }
const Car* lane_get_car(const Lane* self, int index) {
    if (!self || index < 0 || index >= self->num_cars) {
        fprintf(stderr, "Invalid index or lane is NULL\n");
        return NULL;
    }
    return self->cars[index];
}
const Road* lane_get_road(const Lane* self) { return self->road; }
Degradations lane_get_degradations(const Lane* self) { return self->degradations; }

//
// Setters
//

void lane_set_speed_limit(Lane* self, MetersPerSecond speed_limit) {
    self->speed_limit = speed_limit;
}

void lane_set_grip(Lane* self, double grip) {
    self->grip = grip > 0.0 ? (grip < 1.0 ? grip : 1.0) : 0.0;
}

void lane_set_degradations(Lane* self, Degradations degradations) {
    self->degradations = degradations;
}



// --- Linear Lane ---

LinearLane* linear_lane_create_from_center_dir_len(
    Coordinates center,
    Direction direction,
    Meters length,
    Meters width,
    MetersPerSecond speed_limit,
    double grip,
    Degradations degradations
) {
    LinearLane* lane = malloc(sizeof(LinearLane));
    if (!lane) {
        fprintf(stderr, "Memory allocation failed for LinearLane\n");
        exit(1);
    }

    lane_init(&lane->base, LINEAR_LANE, direction, width, speed_limit, grip, degradations);
    lane->base.center = center;
    assert(length > 0.0 && "Length must be positive");
    lane->base.length = length;

    LineSegment seg = line_segment_from_center(center, direction_to_vector(direction), length);
    lane->base.start_point = seg.start;
    lane->base.end_point = seg.end;

    return lane;
}

LinearLane* linear_lane_create_from_start_end(
    Coordinates start,
    Coordinates end,
    Meters width,
    MetersPerSecond speed_limit,
    double grip,
    Degradations degradations
) {
    assert((approxeq(start.x, end.x, 1e-6) || approxeq(start.y, end.y, 1e-6)) && "Linear lane must be vertical or horizontal.");

    Direction direction;
    if (approxeq(start.x, end.x, 1e-6)) {
        direction = start.y < end.y ? DIRECTION_NORTH : DIRECTION_SOUTH;
    } else {
        direction = start.x < end.x ? DIRECTION_EAST : DIRECTION_WEST;
    }

    Meters length = vec_distance(start, end);
    Coordinates center = vec_midpoint(start, end);

    return linear_lane_create_from_center_dir_len(center, direction, length, width, speed_limit, grip, degradations);
}

void linear_lane_free(LinearLane* self) {
    free(self);
}


// --- Quarter Arc Lane ---

QuarterArcLane* quarter_arc_lane_create_from_start_end(
    Coordinates start_point,
    Coordinates end_point,
    Direction direction,
    Meters width,
    MetersPerSecond speed_limit,
    double grip,
    Degradations degradations
) {
    assert(direction == DIRECTION_CW || direction == DIRECTION_CCW);

    QuarterArcLane* lane = malloc(sizeof(QuarterArcLane));
    if (!lane) {
        fprintf(stderr, "Memory allocation failed for QuarterArcLane\n");
        exit(1);
    }

    lane_init(&lane->base, QUARTER_ARC_LANE, direction, width, speed_limit, grip, degradations);
    lane->base.start_point = start_point;
    lane->base.end_point = end_point;

    Vec2D chord = vec_sub(end_point, start_point);
    Radians rotate_by = direction == DIRECTION_CCW ? M_PI / 4 : -M_PI / 4;
    Vec2D z = vec_div(vec_rotate(chord, rotate_by), sqrt(2));
    lane->base.center = vec_add(start_point, z);
    lane->radius = vec_distance(end_point, lane->base.center);
    lane->base.length = M_PI * lane->radius / 2.0;

    z = vec_sub(vec_midpoint(start_point, end_point), lane->base.center);
    if (z.x > 0 && z.y > 0) {
        lane->quadrant = QUADRANT_TOP_RIGHT;
    } else if (z.x < 0 && z.y > 0) {
        lane->quadrant = QUADRANT_TOP_LEFT;
    } else if (z.x < 0 && z.y < 0) {
        lane->quadrant = QUADRANT_BOTTOM_LEFT;
    } else {
        lane->quadrant = QUADRANT_BOTTOM_RIGHT;
    }

    if (direction == DIRECTION_CCW) {
        switch (lane->quadrant) {
            case QUADRANT_TOP_RIGHT:    lane->start_angle = 0.0;               lane->end_angle = M_PI / 2; break;
            case QUADRANT_TOP_LEFT:     lane->start_angle = M_PI / 2;          lane->end_angle = M_PI; break;
            case QUADRANT_BOTTOM_LEFT:  lane->start_angle = M_PI;              lane->end_angle = 3 * M_PI / 2; break;
            case QUADRANT_BOTTOM_RIGHT: lane->start_angle = 3 * M_PI / 2;      lane->end_angle = 2 * M_PI; break;
        }
    } else { // CW
        switch (lane->quadrant) {
            case QUADRANT_TOP_RIGHT:    lane->start_angle = M_PI / 2;          lane->end_angle = 0.0; break;
            case QUADRANT_TOP_LEFT:     lane->start_angle = M_PI;              lane->end_angle = M_PI / 2; break;
            case QUADRANT_BOTTOM_LEFT:  lane->start_angle = 3 * M_PI / 2;      lane->end_angle = M_PI; break;
            case QUADRANT_BOTTOM_RIGHT: lane->start_angle = 2 * M_PI;          lane->end_angle = 3 * M_PI / 2; break;
        }
    }

    return lane;
}

void quarter_arc_lane_free(QuarterArcLane* self) {
    free(self);
}
