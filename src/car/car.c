#include "utils.h"
#include "car.h"
#include "logging.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>


void car_init(Car* car, Dimensions3D dimensions, CarCapabilities capabilities, Liters fuel_tank_capacity, CarPersonality preferences) {
    car->dimensions = dimensions;
    car->fuel_tank_capacity = fuel_tank_capacity;
    car->capabilities = capabilities;
    car->preferences = preferences;
    car->lane_id = ID_NULL;
    car->lane_progress = 0.0;
    car->lane_progress_meters = 0.0;
    car->lane_rank = -1;
    car->speed = 0.0;
    car->damage = 0.0;
    car->fuel_level = fuel_tank_capacity; // Start with a full tank
    car->acceleration = 0.0;
    car->indicator_turn = INDICATOR_NONE;
    car->indicator_lane = INDICATOR_NONE;
    car->request_indicated_lane = false;
    car->request_indicated_turn = false;
    car->auto_turn_off_indicators = true; // Default to true, can be changed later
    car->prev_lane_id = ID_NULL; // Initialize previous lane ID to NULL
    car->recent_forward_movement = 0.0; // Initialize recent forward movement to 0
}

void car_free(Car* self) {
    if (self) {
        free(self);
    }
}

CarId car_get_id(const Car* self) { return self->id; }
Dimensions3D car_get_dimensions(const Car* self) { return self->dimensions; }
Meters car_get_length(const Car* self) { return self->dimensions.y; }
Meters car_get_width(const Car* self) { return self->dimensions.x; }
Meters car_get_height(const Car* self) { return self->dimensions.z; }
Liters car_get_fuel_tank_capacity(const Car* self) { return self->fuel_tank_capacity; }
CarCapabilities car_get_capabilities(const Car* self) { return self->capabilities; }
CarPersonality car_get_preferences(const Car* self) { return self->preferences; }
LaneId car_get_lane_id(const Car* self) { return self->lane_id; }
Lane* car_get_lane(const Car* self, Map* map) {
    if (self->lane_id < 0 || self->lane_id >= map->num_lanes) {
        LOG_ERROR("Car (id=%d) is not on a valid lane (id=%d).", self->id, self->lane_id);
        return NULL;
    }
    return map_get_lane(map, self->lane_id);
}
double car_get_lane_progress(const Car* self) { return self->lane_progress; }
Meters car_get_lane_progress_meters(const Car* self) { return self->lane_progress_meters; }
int car_get_lane_rank(const Car* self) { return self->lane_rank; }
MetersPerSecond car_get_speed(const Car* self) { return self->speed; }
double car_get_damage(const Car* self) { return self->damage; }
Liters car_get_fuel_level(const Car* self) { return self->fuel_level; }
Coordinates car_get_center(const Car* self) { return self->center; }
Radians car_get_orientation(const Car* self) { return self->orientation; }
Coordinates* car_get_corners(const Car* self) { return (Coordinates*)self->corners; }
Coordinates car_get_corner_front_right(const Car* self) { return self->corners[0]; }
Coordinates car_get_corner_front_left(const Car* self) { return self->corners[1]; }
Coordinates car_get_corner_rear_left(const Car* self) { return self->corners[2]; }
Coordinates car_get_corner_rear_right(const Car* self) { return self->corners[3]; }

MetersPerSecondSquared car_get_acceleration(const Car* self) { return self->acceleration; }
CarIndicator car_get_indicator_turn(const Car* self) { return self->indicator_turn; }
CarIndicator car_get_indicator_lane(const Car* self) { return self->indicator_lane; }
bool car_get_request_indicated_lane(const Car* self) { return self->request_indicated_lane; }
bool car_get_request_indicated_turn(const Car* self) { return self->request_indicated_turn; }
bool car_get_auto_turn_off_indicators(const Car* self) { return self->auto_turn_off_indicators; }


void car_set_fuel_tank_capacity(Car* self, Liters capacity) {
    if (capacity <= 0.0) {
        LOG_WARN("Fuel tank capacity must be positive. You are trying to set it to %f. Ignoring.", capacity);
        return;
    }
    self->fuel_tank_capacity = capacity;
    // If the current fuel level exceeds the new capacity, clip it.
    if (self->fuel_level > capacity) {
        LOG_WARN("Current fuel level (%f liters) exceeds new fuel tank capacity (%f liters). Clipping fuel level.", self->fuel_level, capacity);
        self->fuel_level = capacity;
    }
}

void car_set_lane(Car* self, const Lane* lane) {
    self->prev_lane_id = self->lane_id; // Store the lane ID at the previous timestep
    self->lane_id = lane->id;
}

void car_set_lane_progress(Car* self, double progress, Meters progress_meters, Meters recent_forward_movement) {
    if (progress < 0.0 || progress > 1.0) {
        LOG_WARN("Car %d, lane %d : Lane progress must be between 0.0 and 1.0. You are trying to set it to %f. Clipping.", self->id, self->lane_id, progress);
    }
    progress = fclamp(progress, 0.0, 1.0);
    self->lane_progress = progress;
    if (progress_meters < 0.0) {
        progress_meters = 0.0;
    }
    self->lane_progress_meters = progress_meters;
    self->recent_forward_movement = recent_forward_movement;
}

void car_set_lane_rank(Car* self, int rank) {
    self->lane_rank = rank;
}

void car_set_speed(Car* self, MetersPerSecond speed) {
    if (speed > self->capabilities.top_speed) {
        LOG_WARN("Speed exceeds car's top speed. You are trying to set it to %f. Clipping.", speed);
        self->speed = self->capabilities.top_speed;
    } else {
        self->speed = speed;
    }
}

void car_set_acceleration(Car* self, MetersPerSecondSquared acceleration) {
    CarAccelProfile* profile = &self->capabilities.accel_profile;
    if (self->speed >= 0.0) {
        if (acceleration > profile->max_acceleration) acceleration = profile->max_acceleration;     // gas
        if (acceleration < -profile->max_deceleration) acceleration = -profile->max_deceleration;   // braking
    } else {    // reverse gear
        if (acceleration < -profile->max_acceleration_reverse_gear) acceleration = -profile->max_acceleration_reverse_gear;                                                     // gas with reverse gear
        if (acceleration > profile->max_deceleration) acceleration = profile->max_deceleration;     // braking
    }
    self->acceleration = acceleration;
}

void car_set_damage(Car* self, const double damage) {
    if (damage < 0.0 || damage > 1.0) {
        LOG_WARN("Damage must be between 0.0 and 1.0. You are trying to set it to %f. Clipping.", damage);
    }
    self->damage = fclamp(damage, 0.0, 1.0);
}

void car_set_fuel_level(Car* self, Liters fuel_level) {
    if (fuel_level < 0.0) {
        LOG_WARN("Fuel level cannot be negative. You are trying to set it to %f. Clipping.", fuel_level);
    }
    if (fuel_level > self->fuel_tank_capacity) {
        LOG_WARN("Fuel level cannot exceed fuel tank capacity (%f liters). You are trying to set it to %f. Clipping.", self->fuel_tank_capacity, fuel_level);
    }
    self->fuel_level = fclamp(fuel_level, 0.0, self->fuel_tank_capacity);
}

void car_set_indicator_turn(Car* self, CarIndicator indicator) {
    self->indicator_turn = indicator;
}

void car_set_indicator_lane(Car* self, CarIndicator indicator) {
    self->indicator_lane = indicator;
}

void car_set_request_indicated_lane(Car* self, bool request) {
    self->request_indicated_lane = request;
}
void car_set_request_indicated_turn(Car* self, bool request) {
    self->request_indicated_turn = request;
}

void car_set_indicator_turn_and_request(Car* self, CarIndicator indicator) {
    car_set_indicator_turn(self, indicator);
    if (indicator != INDICATOR_NONE) {
        car_set_request_indicated_turn(self, true);
    } else {
        car_set_request_indicated_turn(self, false);
    }
}

void car_set_indicator_lane_and_request(Car* self, CarIndicator indicator) {
    car_set_indicator_lane(self, indicator);
    if (indicator != INDICATOR_NONE) {
        car_set_request_indicated_lane(self, true);
    } else {
        car_set_request_indicated_lane(self, false);
    }
}

void car_set_auto_turn_off_indicators(Car* self, bool auto_turn_off) {
    self->auto_turn_off_indicators = auto_turn_off;
}


Meters car_get_recent_forward_movement(const Car* self) {
    return self->recent_forward_movement;
}

LaneId car_get_prev_lane_id(const Car* self) {
    return self->prev_lane_id;
}

void car_reset_all_control_variables(Car* self) {
    car_set_indicator_lane(self, INDICATOR_NONE);
    car_set_request_indicated_lane(self, false);
    car_set_indicator_turn(self, INDICATOR_NONE);
    car_set_request_indicated_turn(self, false);
    car_set_acceleration(self, 0);
}


void car_update_geometry(Simulation* sim, Car* car) {
    Lane* lane = car_get_lane(car, sim_get_map(sim));
    if (lane->type == LINEAR_LANE) {
        Vec2D start = lane->start_point;
        Vec2D end = lane->end_point;
        Vec2D delta = vec_sub(end, start);
        car->center = vec_add(start, vec_scale(delta, car->lane_progress));
        car->orientation = angle_normalize(atan2(delta.y, delta.x));
    } else if (lane->type == QUARTER_ARC_LANE) {
        double theta = lane->start_angle + car->lane_progress * (lane->end_angle - lane->start_angle);
        theta = angle_normalize(theta);
        car->center.x = lane->center.x + lane->radius * cos(theta);
        car->center.y = lane->center.y + lane->radius * sin(theta);
        if (lane->direction == DIRECTION_CCW) {
            car->orientation = angles_add(theta, M_PI / 2); // CCW: tangent is +90 deg
        } else {
            car->orientation = angles_add(theta, -M_PI / 2); // CW: tangent is -90 deg
        }
    } else {
        LOG_ERROR("Cannot update geometry for car %d: unknown lane type.", car->id);
        return;
    }

    double width = car->dimensions.x;
    double length = car->dimensions.y;
    // Define car corners in local coordinates. Car is facing right (0 radians).
    Vec2D local_corners[4] = {
        {length / 2, -width / 2},   // Front-right (0)
        {length / 2, width / 2},  // Front-left (1)
        {-length / 2, width / 2}, // Rear-left (2)
        {-length / 2, -width / 2}   // Rear-right (3)
    };
    // Transform corners to world coordinates
    for (int i = 0; i < 4; i++) {
        car->corners[i] = vec_add(car_get_center(car), vec_rotate(local_corners[i], car->orientation));
    }
}


bool car_is_colliding(const Car* car1, const Car* car2) {

    // very cheap filter: check if the distance between the centers is larger than 15 meters
    // Bus diagonal is ~12.25m. Two buses touching would have centers ~12.25m apart.
    // We use 15m (225 sq) to include a safety margin and allow for slightly larger future vehicles.
    double dx = car1->center.x - car2->center.x;
    double dy = car1->center.y - car2->center.y;
    double dist_sq = dx * dx + dy * dy;
    if (dist_sq > 225) { // 15^2 = 225
        return false;
    }

    RotatedRect2D car1_rect = {
        car1->center,
        (Dimensions) {car1->dimensions.y, car1->dimensions.x},
        car1->orientation
    };
    RotatedRect2D car2_rect = {
        car2->center,
        (Dimensions) {car2->dimensions.y, car2->dimensions.x},
        car2->orientation
    };
    return rotated_rects_intersect(car1_rect, car2_rect);
}
