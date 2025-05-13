#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include "car.h"


MetersPerSecond acceleration_mode_settling_time(AccelerationMode mode) {
    // based on 60kmph to 0kmph stopping time
    switch (mode) {
        case CHILL_MODE:        // a ~ 2.5 m/s^2    k = 0.36.
            return 6.66;
        case STANDARD_MODE:     // a ~ 5.0 m/s^2    k = 1.44
            return 3.33;
        case SPORT_MODE:        // a ~ 7.5 m/s^2    k = 3.24
            return 2.22;
        case EMERGENCY_MODE:    // a ~ 10.0 m/s^2   k = 5.76
            return 1.66;
        default:
            return 0.0; // Invalid mode
    }
}

double pid_k_for_stopping_time(const Seconds settling_time) {
    return 16.0 / (settling_time * settling_time);
}

double acceleration_mode_pid_k(AccelerationMode mode) {
    double t = acceleration_mode_settling_time(mode);
    return pid_k_for_stopping_time(t);
}


Car* car_create(const Dimensions dimensions, const CarCapabilities capabilities, const PreferenceProfile preferences) {
    Car* car = (Car*)malloc(sizeof(Car));
    if (!car) {
        fprintf(stderr, "Memory allocation failed for Car\n");
        exit(1);
    }
    car->dimensions = dimensions;
    car->capabilities = capabilities;
    car->preferences = preferences;
    car->lane = NULL;
    car->lane_progress = 0.0;
    car->speed = 0.0;
    car->acceleration = 0.0;
    car->turn_intent = INTENT_STRAIGHT;
    car->lane_change_intent = INTENT_STRAIGHT;
    car->damage = 0.0;
    car->npc_state = (NPCState){false, false, false, 0, 0, 0};
    return car;
}


// ---------------- setters ----------------

void car_set_lane(Car* self, const Lane* lane) {
    self->lane = lane;

}

void car_set_lane_progress(Car* self, const double progress) {
    if (progress < 0.0 || progress > 1.0) {
        fprintf(stderr, "Lane progress must be between 0.0 and 1.0. You are trying to set it to %f\n", progress);
        self->lane_progress = fmax(0.0, fmin(1.0, progress)); // Clip to [0.0, 1.0]
    } else if (self->lane != NULL) {
        self->lane_progress = progress;
    } else {
        fprintf(stderr, "Car is not in a lane\n");
    }
}

void car_set_speed(Car* self, const MetersPerSecond speed) {
    if (speed > self->capabilities.top_speed) {
        fprintf(stderr, "Speed exceeds car's top speed. Clipping to top speed.\n");
        self->speed = self->capabilities.top_speed;
    } else {
        self->speed = speed;
    }
}

void car_set_acceleration(Car* self, const MetersPerSecondSquared acceleration) {
    if (acceleration > self->capabilities.max_acceleration) {
        fprintf(stderr, "Acceleration exceeds car's maximum acceleration capability. Clipping to max acceleration.\n");
        self->acceleration = self->capabilities.max_acceleration;
    } else if (acceleration < -self->capabilities.max_deceleration) {
        fprintf(stderr, "Deceleration exceeds car's maximum deceleration (braking) capability. Clipping to max deceleration.\n");
        self->acceleration = -self->capabilities.max_deceleration;
    } else {
        self->acceleration = acceleration;
    }
}

void car_set_turn_intent(Car* self, const DirectionIntent intent, bool auto_correct) {
    if (is_turn_possible(self->lane, intent) || intent == INTENT_NA) {
        self->turn_intent = intent;
    } else if (auto_correct) {
        self->turn_intent = get_legitimate_turn_intent_closest_to(self->lane, intent);
    } else {
        fprintf(stderr, "Turn intent %d not possible and auto-correction is disabled.\n", intent);
        exit(1);
    }
}

void car_set_lane_change_intent(Car* self, const DirectionIntent intent) {
    self->lane_change_intent = intent;
}

void car_set_damage(Car* self, const double damage) {
    if (damage < 0.0 || damage > 1.0) {
        fprintf(stderr, "Damage must be between 0.0 and 1.0\n");
        exit(1);
    }
    self->damage = damage;
}


// ---------------- getters ----------------

Meters car_get_length(const Car* self) {
    return self->dimensions.y;
}
Meters car_get_width(const Car* self) {
    return self->dimensions.x;
}
const Lane* car_get_lane(const Car* self) {
    // assert(self->lane != NULL && "Car is not in a lane");
    return self->lane;
}
double car_get_lane_progress(const Car* self) {
    // assert(self->lane_progress >= 0.0 && self->lane_progress <= 1.0 && "Lane progress must be between 0.0 and 1.0. How did this happen?");
    return self->lane_progress;
}
Meters car_get_lane_progress_meters(const Car* self) {
    return car_get_lane(self)->length * car_get_lane_progress(self);
}
MetersPerSecond car_get_speed(const Car* self) {
    // assert(self->speed <= self->capabilities.top_speed && "Speed exceeds car's top speed. How did this happen?");
    return self->speed;
}
MetersPerSecondSquared car_get_acceleration(const Car* self) {
    // assert(self->acceleration <= self->capabilities.max_acceleration && "Acceleration exceeds car's maximum acceleration capability. How did this happen?");
    // assert(self->acceleration >= -self->capabilities.max_deceleration && "Deceleration exceeds car's maximum deceleration (braking) capability. How did this happen?");
    return self->acceleration;
}
DirectionIntent car_get_turn_intent(const Car* self) {
    return self->turn_intent;
}
DirectionIntent car_get_lane_change_intent(const Car* self) {
    return self->lane_change_intent;
}
double car_get_damage(const Car* self) {
    // assert(self->damage >= 0.0 && self->damage <= 1.0 && "Damage must be between 0.0 and 1.0. How did this happen?");
    return self->damage;
}


// ---------------- decision-making functions ----------------





void player_car_make_decision(Car* self, Simulation* sim) {
    printf("Player car decision-making logic not implemented\n");
}


void car_free(Car* self) {
    free(self);
}




