#include "sim.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>


// --- Clock Utility Functions ---

ClockReading clock_reading(const Days days, const Hours hours, const Minutes minutes, const Seconds seconds) {
    ClockReading clock = { days, hours, minutes, seconds, days % 7 };
    return clock;
}

ClockReading clock_reading_from_seconds(const Seconds seconds) {
    if (seconds < 0) {
        fprintf(stderr, "Error: seconds cannot be negative\n");
        exit(EXIT_FAILURE);
    }
    ClockReading clock;
    Seconds seconds_remaining = seconds;
    clock.days = (int)(seconds_remaining / 86400);
    seconds_remaining -= clock.days * 86400;
    clock.hours = (int)(seconds_remaining / 3600);
    seconds_remaining -= clock.hours * 3600;
    clock.minutes = (int)(seconds_remaining / 60);
    seconds_remaining -= clock.minutes * 60;
    clock.seconds = seconds_remaining;
    clock.day_of_week = clock.days % 7; // Assuming day 0 is Monday
    return clock;
}

Seconds clock_reading_to_seconds(const ClockReading clock) {
    return clock.seconds +
           clock.minutes * 60 +
           clock.hours * 3600 +
           clock.days * 86400;
}

ClockReading clock_reading_add_seconds(const ClockReading clock, const Seconds seconds) {
    return clock_reading_from_seconds(clock_reading_to_seconds(clock) + seconds);
}

ClockReading clock_reading_add(const ClockReading clock1, const ClockReading clock2) {
    return clock_reading_from_seconds(
        clock_reading_to_seconds(clock1) + clock_reading_to_seconds(clock2)
    );
}

// --- Simulation Core Functions ---

Simulation* sim_create(Map* map, double dt) {
    Simulation* sim = malloc(sizeof(Simulation));
    if (!sim) {
        fprintf(stderr, "Memory allocation failed for Simulation\n");
        exit(EXIT_FAILURE);
    }

    sim->map = map;
    sim->initial_clock_reading = clock_reading(0, 8, 0, 0);  // 8:00 AM Monday
    sim->num_cars = 0;
    sim->time = 0.0;
    sim->dt = dt;
    sim->weather = WEATHER_SUNNY;
    return sim;
}

void sim_add_car(Simulation* self, Car* car) {
    if (!self || !car) return;
    if (self->num_cars >= MAX_CARS_IN_SIMULATION) {
        fprintf(stderr, "Simulation is full, cannot add more cars\n");
        return;
    }
    self->cars[self->num_cars++] = car;
}

// --- Getters ---

Map* sim_get_map(Simulation* self) {
    return self ? self->map : NULL;
}

ClockReading sim_get_initial_clock_reading(Simulation* self) {
    return self->initial_clock_reading;
}

Car** sim_get_cars(Simulation* self) {
    return self->cars;
}

int sim_get_num_cars(const Simulation* self) {
    return self ? self->num_cars : 0;
}

Car* sim_get_car(Simulation* self, int id) {
    if (!self || id < 0 || id >= self->num_cars) return NULL;
    return self->cars[id];
}

Car* sim_get_agent_car(Simulation* self) {
    return sim_get_car(self, 0);
}

Seconds sim_get_time(Simulation* self) {
    return self ? self->time : 0;
}

Seconds sim_get_dt(const Simulation* self) {
    return self ? self->dt : 0;
}

Weather sim_get_weather(Simulation* self) {
    return self ? self->weather : WEATHER_SUNNY;
}

DayOfWeek sim_get_day_of_week(const Simulation* self) {
    if (!self) return MONDAY;
    return sim_get_clock_reading((Simulation*)self).day_of_week;
}

bool sim_car_exists(const Simulation* self, int id) {
    return self && id >= 0 && id < self->num_cars && self->cars[id] != NULL;
}

// --- Setters ---

void sim_set_initial_clock_reading(Simulation* self, ClockReading clock) {
    if (self) self->initial_clock_reading = clock;
}

void sim_set_weather(Simulation* self, Weather weather) {
    if (self) self->weather = weather;
}

// --- Time & Condition Queries ---

ClockReading sim_get_clock_reading(Simulation* self) {
    return clock_reading_add(self->initial_clock_reading, clock_reading_from_seconds(self->time));
}

bool sim_is_weekend(Simulation* self) {
    DayOfWeek d = sim_get_day_of_week(self);
    return d == SATURDAY || d == SUNDAY;
}

bool sim_is_morning(Simulation* self) {
    int h = sim_get_clock_reading(self).hours;
    return h >= MORNING_START && h < AFTERNOON_START;
}

bool sim_is_afternoon(Simulation* self) {
    int h = sim_get_clock_reading(self).hours;
    return h >= AFTERNOON_START && h < EVENING_START;
}

bool sim_is_evening(Simulation* self) {
    int h = sim_get_clock_reading(self).hours;
    return h >= EVENING_START && h < NIGHT_START;
}

bool sim_is_night(Simulation* self) {
    int h = sim_get_clock_reading(self).hours;
    return h >= NIGHT_START || h < MORNING_START;
}

double sunlight_intensity(Simulation* self) {
    Hours h = sim_get_clock_reading(self).hours;
    if (h < 6 || h > 18) return 0.0;
    return sin(M_PI * (h - 6) / 12);
}

// --- Simulation Ticking ---

void sim_step(Simulation* self) {
    if (self) simulate(self, self->dt);
}

void sim_free(Simulation* self) {
    if (!self) return;
    free(self);
}

void sim_deep_free(Simulation* self) {
    if (!self) return;
    for (int i = 0; i < self->num_cars; i++) {
        car_free(self->cars[i]);
    }
    map_deep_free(self->map);
    free(self);
}
