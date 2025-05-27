#include "utils.h"
#include "car.h"
#include "logging.h"
#include <stdlib.h>
#include <stdio.h>


int car_id_counter = 0; // Global ID counter for cars

Car* car_create(Dimensions dimensions, CarCapabilities capabilities, CarPersonality preferences) {
    Car* car = (Car*)malloc(sizeof(Car));
    if (!car) {
        LOG_ERROR("Memory allocation failed for Car");
        exit(EXIT_FAILURE);
    }
    car->id = car_id_counter++;
    car->dimensions = dimensions;
    car->capabilities = capabilities;
    car->preferences = preferences;
    car->lane = NULL;
    car->lane_progress = 0.0;
    car->speed = 0.0;
    car->damage = 0.0;
    car->acceleration = 0.0;
    car->indicator_turn = INDICATOR_NONE;
    car->indicator_lane = INDICATOR_NONE;
    return car;
}

void car_free(Car* self) {
    if (self) {
        free(self);
    }
}

Dimensions car_get_dimensions(const Car* self) { return self->dimensions; }
Meters car_get_length(const Car* self) { return self->dimensions.y; }
Meters car_get_width(const Car* self) { return self->dimensions.x; }
CarCapabilities car_get_capabilities(const Car* self) { return self->capabilities; }
CarPersonality car_get_preferences(const Car* self) { return self->preferences; }
const Lane* car_get_lane(const Car* self) { return self->lane; }
double car_get_lane_progress(const Car* self) { return self->lane_progress; }
Meters car_get_lane_progress_meters(const Car* self) {
    return self->lane ? self->lane->length * self->lane_progress : 0.0;
}
MetersPerSecond car_get_speed(const Car* self) { return self->speed; }
double car_get_damage(const Car* self) { return self->damage; }
MetersPerSecondSquared car_get_acceleration(const Car* self) { return self->acceleration; }
CarIndictor car_get_indicator_turn(const Car* self) { return self->indicator_turn; }
CarIndictor car_get_indicator_lane(const Car* self) { return self->indicator_lane; }

void car_set_lane(Car* self, const Lane* lane) {
    self->lane = lane;
}

void car_set_lane_progress(Car* self, double progress) {
    if (progress < 0.0 || progress > 1.0) {
        // fprintf(stderr, "WARN: Lane progress must be between 0.0 and 1.0. You are trying to set it to %f. Clipping.\n", progress);
    }
    progress = fclamp(progress, 0.0, 1.0);
    self->lane_progress = progress;
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