#include "utils.h"
#include "car.h"
#include "logging.h"
#include <stdlib.h>
#include <stdio.h>


int car_id_counter = 0; // Global ID counter for cars

void car_init(Car* car, Dimensions dimensions, CarCapabilities capabilities, CarPersonality preferences) {
    car->id = car_id_counter++;
    car->dimensions = dimensions;
    car->capabilities = capabilities;
    car->preferences = preferences;
    car->lane_id = ID_NULL;
    car->lane_progress = 0.0;
    car->lane_progress_meters = 0.0;
    car->lane_rank = -1;
    car->speed = 0.0;
    car->damage = 0.0;
    car->acceleration = 0.0;
    car->indicator_turn = INDICATOR_NONE;
    car->indicator_lane = INDICATOR_NONE;
}

void car_free(Car* self) {
    if (self) {
        free(self);
    }
}

CarId car_get_id(const Car* self) { return self->id; }
Dimensions car_get_dimensions(const Car* self) { return self->dimensions; }
Meters car_get_length(const Car* self) { return self->dimensions.y; }
Meters car_get_width(const Car* self) { return self->dimensions.x; }
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
MetersPerSecondSquared car_get_acceleration(const Car* self) { return self->acceleration; }
CarIndictor car_get_indicator_turn(const Car* self) { return self->indicator_turn; }
CarIndictor car_get_indicator_lane(const Car* self) { return self->indicator_lane; }

void car_set_lane(Car* self, const Lane* lane) {
    self->lane_id = lane->id;
}

void car_set_lane_progress(Car* self, double progress, Meters progress_meters) {
    if (progress < 0.0 || progress > 1.0) {
        LOG_WARN("Car %d, lane %d : Lane progress must be between 0.0 and 1.0. You are trying to set it to %f. Clipping.", self->id, self->lane_id, progress);
    }
    progress = fclamp(progress, 0.0, 1.0);
    self->lane_progress = progress;
    if (progress_meters < 0.0) {
        progress_meters = 0.0;
    }
    self->lane_progress_meters = progress_meters;
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

void car_set_indicator_turn(Car* self, CarIndictor indicator) {
    self->indicator_turn = indicator;
}

void car_set_indicator_lane(Car* self, CarIndictor indicator) {
    self->indicator_lane = indicator;
}