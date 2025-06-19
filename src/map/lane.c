#include "map.h"
#include "utils.h"
#include "logging.h"
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>

const Degradations DEGRADATIONS_ZERO = {0};

//
// Lane Setters
//

void lane_set_connection_incoming_left(Lane* self, const Lane* lane) {
    if (!lane) {
        self->connections_incoming[0] = -1; // No incoming left connection
        return;
    }
    assert(vec_approx_equal(lane->end_point, self->start_point, 1e-6));
    self->connections_incoming[0] = lane->id; // incoming left connection
}

void lane_set_connection_incoming_straight(Lane* self, const Lane* lane) {
    if (!lane) {
        self->connections_incoming[1] = -1; // No incoming straight connection
        return;
    }
    assert(vec_approx_equal(lane->end_point, self->start_point, 1e-6));
    self->connections_incoming[1] = lane->id; // incoming straight connection
}

void lane_set_connection_incoming_right(Lane* self, const Lane* lane) {
    if (!lane) {
        self->connections_incoming[2] = -1; // No incoming right connection
        return;
    }
    assert(vec_approx_equal(lane->end_point, self->start_point, 1e-6));
    self->connections_incoming[2] = lane->id; // incoming right connection
}

void lane_set_connection_left(Lane* self, const Lane* lane) {
    if (!lane) {
        self->connections[0] = -1; // No left connection
        return;
    }
    assert(vec_approx_equal(self->end_point, lane->start_point, 1e-6));
    self->connections[0] = lane->id; // left connection
}

void lane_set_connection_straight(Lane* self, const Lane* lane) {
    if (!lane) {
        self->connections[1] = -1; // No straight connection
        return;
    }
    assert(vec_approx_equal(self->end_point, lane->start_point, 1e-6));
    self->connections[1] = lane->id; // straight connection
}

void lane_set_connection_right(Lane* self, const Lane* lane) {
    if (!lane) {
        self->connections[2] = -1; // No right connection
        return;
    }
    assert(vec_approx_equal(self->end_point, lane->start_point, 1e-6));
    self->connections[2] = lane->id; // right connection
}

void lane_set_adjacents(Lane* self, const Lane* left, const Lane* right) {
    self->adjacents[0] = left ? left->id : -1; // left adjacent
    lane_set_adjacent_right(self, right); // right adjacent
}

void lane_set_adjacent_left(Lane* self, const Lane* left) {
    self->adjacents[0] = left ? left->id : -1; // left adjacent
}

void lane_set_adjacent_right(Lane* self, const Lane* right) {
    if (right) {
        assert(right->direction == self->direction && "Right adjacent lane must have the same direction as the current lane.");
        self->adjacents[1] = right->id; // right adjacent
    } else {
        self->adjacents[1] = -1; // No right adjacent
    }
}

void lane_set_merges_into(Lane* self, const Lane* lane, const double start, const double end) {
    self->merges_into_id = lane ? lane->id : -1; // merges into lane
    self->merges_into_start = start;
    self->merges_into_end = end;
}

void lane_set_exit_lane(Lane* self, const Lane* lane, const double start, const double end) {
    self->exit_lane_id = lane ? lane->id : -1; // exit lane
    self->exit_lane_start = start;
    self->exit_lane_end = end;
}

void lane_set_speed_limit(Lane* self, MetersPerSecond speed_limit) {
    self->speed_limit = speed_limit;
}

void lane_set_grip(Lane* self, double grip) {
    self->grip = grip > 0.0 ? (grip < 1.0 ? grip : 1.0) : 0.0;
}

void lane_set_degradations(Lane* self, Degradations degradations) {
    self->degradations = degradations;
}

void lane_set_road(Lane* self, const Road* road) {
    if (road && lane_is_at_intersection(self)) {
        LOG_ERROR("Cannot set road (id=%d) for lane (id=%d), which is already specified to part of an intersection (id=%d).", road->id, self->id, self->intersection_id);
        return;
    }
    self->road_id = road ? road->id : -1; // Set the road id, or -1 if no road
}

void lane_set_intersection(Lane* self, const Intersection* intersection) {
    self->intersection_id = intersection ? intersection->id : -1; // Set the intersection id, or -1 if no intersection
    self->is_at_intersection = (intersection != NULL);
}

void lane_set_name(Lane* self, const char* name) {
    if (name) {
        snprintf(self->name, sizeof(self->name), "%s", name);
    } else {
        self->name[0] = '\0'; // Clear the name if NULL
    }
}


//
// Getters
//

int lane_get_id(const Lane* self) { return self->id; }
LaneType lane_get_type(const Lane* self) { return self->type; }
Direction lane_get_direction(const Lane* self) { return self->direction; }
Meters lane_get_width(const Lane* self) { return self->width; }
MetersPerSecond lane_get_speed_limit(const Lane* self) { return self->speed_limit; }
double lane_get_grip(const Lane* self) { return self->grip; }
Coordinates lane_get_start_point(const Lane* self) { return self->start_point; }
Coordinates lane_get_end_point(const Lane* self) { return self->end_point; }
Coordinates lane_get_mid_point(const Lane* self) { return self->mid_point; }
Coordinates lane_get_center(const Lane* self) { return self->center; }
Meters lane_get_length(const Lane* self) { return self->length; }
Degradations lane_get_degradations(const Lane* self) { return self->degradations; }

LaneId lane_get_connection_incoming_left_id(const Lane* self) { return self->connections_incoming[0]; }
Lane* lane_get_connection_incoming_left(const Lane* self, Map* map) {
    LaneId left_incoming_id = lane_get_connection_incoming_left_id(self);
    if (left_incoming_id < 0) return NULL;
    return map_get_lane(map, left_incoming_id);
}
LaneId lane_get_connection_incoming_straight_id(const Lane* self) { return self->connections_incoming[1]; }
Lane* lane_get_connection_incoming_straight(const Lane* self, Map* map) {
    LaneId straight_incoming_id = lane_get_connection_incoming_straight_id(self);
    if (straight_incoming_id < 0) return NULL;
    return map_get_lane(map, straight_incoming_id);
}
LaneId lane_get_connection_incoming_right_id(const Lane* self) { return self->connections_incoming[2]; }
Lane* lane_get_connection_incoming_right(const Lane* self, Map* map) {
    LaneId right_incoming_id = lane_get_connection_incoming_right_id(self);
    if (right_incoming_id < 0) return NULL;
    return map_get_lane(map, right_incoming_id);
}
LaneId lane_get_connection_left_id(const Lane* self) { return self->connections[0]; }
Lane* lane_get_connection_left(const Lane* self, Map* map) {
    LaneId left_id = lane_get_connection_left_id(self);
    if (left_id < 0) return NULL;
    return map_get_lane(map, left_id);
}
LaneId lane_get_connection_straight_id(const Lane* self) { return self->connections[1]; }
Lane* lane_get_connection_straight(const Lane* self, Map* map) {
    LaneId straight_id = lane_get_connection_straight_id(self);
    if (straight_id < 0) return NULL;
    return map_get_lane(map, straight_id);
}
LaneId lane_get_connection_right_id(const Lane* self) { return self->connections[2]; }
Lane* lane_get_connection_right(const Lane* self, Map* map) {
    LaneId right_id = lane_get_connection_right_id(self);
    if (right_id < 0) return NULL;
    return map_get_lane(map, right_id);
}
LaneId lane_get_merge_into_id(const Lane* self) { return self->merges_into_id; }
Lane* lane_get_merge_into(const Lane* self, Map* map) {
    LaneId merge_id = lane_get_merge_into_id(self);
    if (merge_id < 0) return NULL;
    return map_get_lane(map, merge_id);
}
LaneId lane_get_exit_lane_id(const Lane* self) { return self->exit_lane_id; }
Lane* lane_get_exit_lane(const Lane* self, Map* map) {
    LaneId exit_id = lane_get_exit_lane_id(self);
    if (exit_id < 0) return NULL;
    return map_get_lane(map, exit_id);
}
LaneId lane_get_adjacent_left_id(const Lane* self) { return self->adjacents[0]; }
Lane* lane_get_adjacent_left(const Lane* self, Map* map) {
    LaneId left_id = lane_get_adjacent_left_id(self);
    if (left_id < 0) return NULL;
    return map_get_lane(map, left_id);
}
LaneId lane_get_adjacent_right_id(const Lane* self) { return self->adjacents[1]; }
Lane* lane_get_adjacent_right(const Lane* self, Map* map) {
    LaneId right_id = lane_get_adjacent_right_id(self);
    if (right_id < 0) return NULL;
    return map_get_lane(map, right_id);
}
RoadId lane_get_road_id(const Lane* self) { return self->road_id; }
Road* lane_get_road(const Lane* self, Map* map) {
    RoadId road_id = lane_get_road_id(self);
    if (road_id < 0) return NULL;
    return map_get_road(map, road_id);
}
IntersectionId lane_get_intersection_id(const Lane* self) { return self->intersection_id; }
bool lane_is_at_intersection(const Lane* self) { return self->is_at_intersection; }
Intersection* lane_get_intersection(const Lane* self, Map* map) {
    IntersectionId intersection_id = lane_get_intersection_id(self);
    if (intersection_id < 0) return NULL;
    return map_get_intersection(map, intersection_id);
}
int lane_get_num_cars(const Lane* self) { return self->num_cars; }
CarId lane_get_car_id(const Lane* self, int index) {
    if (!self || index < 0 || index >= self->num_cars || index >= MAX_CARS_PER_LANE) {
        LOG_ERROR("Invalid index %d for lane (id=%d) with %d cars, max %d.", index, self->id, self->num_cars, MAX_CARS_PER_LANE);
        return ID_NULL;
    }
    return self->cars_ids[index];
}
Car* lane_get_car(const Lane* self, Simulation* sim, int index) {
    CarId car_id = lane_get_car_id(self, index);
    if (car_id == ID_NULL) {
        return NULL;
    }
    return sim_get_car(sim, car_id);
}


//
// Fancier Functions
//

int lane_get_num_connections_incoming(const Lane* self) {
    int count = 0;
    if (self->connections_incoming[0] >= 0) count++; // left incoming
    if (self->connections_incoming[1] >= 0) count++; // straight incoming
    if (self->connections_incoming[2] >= 0) count++; // right incoming
    return count;
}

int lane_get_num_connections(const Lane* self) {
    int count = 0;
    if (self->connections[0] >= 0) count++; // left connection
    if (self->connections[1] >= 0) count++; // straight connection
    if (self->connections[2] >= 0) count++; // right connection
    return count;
}

bool lane_is_merge_available(const Lane* self) {
    return self->merges_into_id != ID_NULL;
}

bool lane_is_exit_lane_available(const Lane* self, const double progress) {
    return self->exit_lane_id != ID_NULL &&
           self->exit_lane_start <= progress &&
           progress <= self->exit_lane_end;
}

bool lane_is_exit_lane_eventually_available(const Lane* self, double progress) {
    return self->exit_lane_id && (self->exit_lane_start > progress);
}


// --- Lane Creaters ---


Lane* lane_create(Map* map, const LaneType type, const Direction direction, const Meters width, const MetersPerSecond speed_limit, const double grip, const Degradations degradations) {
    Lane* lane = map_get_new_lane(map);
    LOG_TRACE("Initializing lane with id %d, type %d, direction %d, width %.2f, speed limit %.2f, grip %.2f, degradations %d",
              lane->id, type, direction, width, speed_limit, grip, degradations);
    lane->type = type;
    snprintf(lane->name, sizeof(lane->name), "Lane %d", lane->id);
    lane->direction = direction;
    lane->width = width > 0.0 ? width : 0.0;
    lane->speed_limit = speed_limit;
    lane->grip = grip > 0.0 ? (grip < 1.0 ? grip : 1.0) : 0.0;
    lane->degradations = degradations;

    lane->connections_incoming[0] = -1; // left incoming
    lane->connections_incoming[1] = -1; // straight incoming
    lane->connections_incoming[2] = -1; // right incoming
    lane->connections[0] = -1; // left
    lane->connections[1] = -1; // straight
    lane->connections[2] = -1; // right
    lane->adjacents[0] = -1; // left
    lane->adjacents[1] = -1; // right
    lane->merges_into_id = -1;  // no merge by default
    lane->merges_into_start = 0.0;
    lane->merges_into_end = 0.0;
    lane->exit_lane_id = -1;    // no exit lane by default
    lane->exit_lane_start = 0.0;
    lane->exit_lane_end = 0.0;

    lane->num_cars = 0;

    lane->start_point = vec_create(0.0, 0.0);
    lane->end_point = vec_create(0.0, 0.0);
    lane->center = vec_create(0.0, 0.0);
    lane->mid_point = lane->center;
    lane->length = 0.0;

    lane->radius = 0.0;
    lane->start_angle = 0.0;
    lane->end_angle = 0.0;
    lane->quadrant = QUADRANT_TOP_RIGHT;

    lane->road_id = -1; // no road by default
    lane->intersection_id = -1; // no intersection by default
    lane->is_at_intersection = false;

    return lane;
}


// --- Linear Lane ---

Lane* linear_lane_create_from_center_dir_len(
    Map* map,
    Coordinates center,
    Direction direction,
    Meters length,
    Meters width,
    MetersPerSecond speed_limit,
    double grip,
    Degradations degradations
) {
    LOG_TRACE("Creating linear lane at center (%.2f, %.2f) with direction %d, length %.2f, width %.2f, speed limit %.2f, grip %.2f",
              center.x, center.y, direction, length, width, speed_limit, grip);
    Lane* lane = lane_create(map, LINEAR_LANE, direction, width, speed_limit, grip, degradations);
    lane->center = center;
    lane->mid_point = center; // For linear lanes, the midpoint is the same as the center.
    assert(length > 0.0 && "Length must be positive");
    lane->length = length;

    LineSegment seg = line_segment_from_center(center, direction_to_vector(direction), length);
    lane->start_point = seg.start;
    lane->end_point = seg.end;
    return lane;
}

Lane* linear_lane_create_from_start_end(
    Map* map,
    Coordinates start,
    Coordinates end,
    Meters width,
    MetersPerSecond speed_limit,
    double grip,
    Degradations degradations
) {
    LOG_TRACE("Creating linear lane from start (%.2f, %.2f) to end (%.2f, %.2f) with width %.2f, speed limit %.2f, grip %.2f",
              start.x, start.y, end.x, end.y, width, speed_limit, grip);
    // assert((approxeq(start.x, end.x, 1e-6) || approxeq(start.y, end.y, 1e-6)) && "Linear lane must be vertical or horizontal.");
    if (!(approxeq(start.x, end.x, 1e-6) || approxeq(start.y, end.y, 1e-6))) {
        LOG_ERROR("Start and end points must be aligned either horizontally or vertically. You provided start (%.2f, %.2f) and end (%.2f, %.2f).",
                  start.x, start.y, end.x, end.y);
        exit(EXIT_FAILURE);
    }

    Direction direction;
    if (approxeq(start.x, end.x, 1e-6)) {
        direction = start.y < end.y ? DIRECTION_NORTH : DIRECTION_SOUTH;
    } else {
        direction = start.x < end.x ? DIRECTION_EAST : DIRECTION_WEST;
    }

    Meters length = vec_distance(start, end);
    Coordinates center = vec_midpoint(start, end);

    Lane* lane = linear_lane_create_from_center_dir_len(map, center, direction, length, width, speed_limit, grip, degradations);

    return lane;
}

// --- Quarter Arc Lane ---

Lane* quarter_arc_lane_create_from_start_end(
    Map* map,
    Coordinates start_point,
    Coordinates end_point,
    Direction direction,
    Meters width,
    MetersPerSecond speed_limit,
    double grip,
    Degradations degradations
) {
    assert(direction == DIRECTION_CW || direction == DIRECTION_CCW);

    Lane* lane = lane_create(map, QUARTER_ARC_LANE, direction, width, speed_limit, grip, degradations);
    lane->start_point = start_point;
    lane->end_point = end_point;

    Vec2D chord = vec_sub(end_point, start_point);
    Radians rotate_by = direction == DIRECTION_CCW ? M_PI / 4 : -M_PI / 4;
    Vec2D z = vec_div(vec_rotate(chord, rotate_by), sqrt(2));
    lane->center = vec_add(start_point, z);
    lane->radius = vec_distance(end_point, lane->center);
    lane->length = M_PI * lane->radius / 2.0;

    z = vec_sub(vec_midpoint(start_point, end_point), lane->center);
    lane->mid_point = vec_add(lane->center, vec_scale(vec_normalize(z), lane->radius));
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



//
// Lane Car Management
//

// add a car to the lane, maintaining descending order by progress
void lane_add_car(Lane* self, Car* car, Simulation* sim) {
    CarId car_id = car_get_id(car);
    if (self->num_cars >= MAX_CARS_PER_LANE) {
        LOG_ERROR("Lane (id=%d) is full, cannot add car (id=%d). Max cars allowed: %d", self->id, car_id, MAX_CARS_PER_LANE);
        return;
    }

    float new_progress = car_get_lane_progress(car);
    int insert_pos = self->num_cars;

    // Find correct position to insert (keep descending order: highest progress first)
    for (int i = 0; i < self->num_cars; i++) {
        Car* car_i = sim_get_car(sim, self->cars_ids[i]);
        if (new_progress > car_get_lane_progress(car_i)) {
            insert_pos = i;
            break;
        }
    }

    // Shift elements to the right and update ranks
    for (int i = self->num_cars; i > insert_pos; i--) {
        self->cars_ids[i] = self->cars_ids[i - 1];
        Car* moved_car = sim_get_car(sim, self->cars_ids[i]);
        car_set_lane_rank(moved_car, i);
    }

    self->cars_ids[insert_pos] = car_id;
    car_set_lane_rank(car, insert_pos);
    self->num_cars++;
}

// move a car (after updated progress) within the lane, maintaining descending order by progress, assuming other cars are properly ordered
void lane_move_car(Lane* self, Car* car, Simulation* sim) {
    CarId car_id = car_get_id(car);
    float new_progress = car_get_lane_progress(car);

    int index = car_get_lane_rank(car);  // use known rank

    if (index < 0 || index >= self->num_cars || self->cars_ids[index] != car_id) {
        LOG_ERROR("Car (id=%d) has invalid rank in lane (id=%d). Cannot move.", car_id, self->id);
        return;
    }

    // Early exit if order is already valid
    if ((index == 0 || new_progress <= car_get_lane_progress(sim_get_car(sim, self->cars_ids[index - 1]))) &&
        (index == self->num_cars - 1 || new_progress >= car_get_lane_progress(sim_get_car(sim, self->cars_ids[index + 1])))) {
        return;
    }

    // Bubble left (toward front) if progress increased
    while (index > 0) {
        int prev_idx = index - 1;
        Car* prev_car = sim_get_car(sim, self->cars_ids[prev_idx]);
        if (new_progress > car_get_lane_progress(prev_car)) {
            self->cars_ids[index] = self->cars_ids[prev_idx];
            car_set_lane_rank(prev_car, index);
            index--;
            self->cars_ids[index] = car_id;
        } else {
            break;
        }
    }

    // Bubble right (toward back) if progress decreased
    while (index < self->num_cars - 1) {
        int next_idx = index + 1;
        Car* next_car = sim_get_car(sim, self->cars_ids[next_idx]);
        if (new_progress < car_get_lane_progress(next_car)) {
            self->cars_ids[index] = self->cars_ids[next_idx];
            car_set_lane_rank(next_car, index);
            index++;
            self->cars_ids[index] = car_id;
        } else {
            break;
        }
    }

    car_set_lane_rank(car, index);
}

// remove a car from the lane, maintaining descending order by progress
void lane_remove_car(Lane* self, Car* car, Simulation* sim) {
    CarId car_id = car_get_id(car);
    int index = car_get_lane_rank(car);

    if (index < 0 || index >= self->num_cars || self->cars_ids[index] != car_id) {
        LOG_ERROR("Car (id=%d) has invalid rank in lane (id=%d). Cannot remove.", car_id, self->id);
        return;
    }

    // Shift left and update ranks
    for (int i = index; i < self->num_cars - 1; i++) {
        self->cars_ids[i] = self->cars_ids[i + 1];
        Car* moved_car = sim_get_car(sim, self->cars_ids[i]);
        car_set_lane_rank(moved_car, i);
    }
    self->cars_ids[self->num_cars - 1] = ID_NULL; // Clear last position

    self->num_cars--;

    car_set_lane_rank(car, -1); // Reset rank in car
}
