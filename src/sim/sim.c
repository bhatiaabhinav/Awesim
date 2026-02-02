#include "sim.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>


const char* day_of_week_strings[] = { "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday" };
const char* weather_strings[] = { "Sunny", "Cloudy", "Rainy", "Foggy", "Snowy" };

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

const ClockReading Monday_12_AM = { 0, 0, 0, 0.0, MONDAY };
const ClockReading Monday_1_AM = { 0, 1, 0, 0.0, MONDAY };
const ClockReading Monday_2_AM = { 0, 2, 0, 0.0, MONDAY };
const ClockReading Monday_3_AM = { 0, 3, 0, 0.0, MONDAY };
const ClockReading Monday_4_AM = { 0, 4, 0, 0.0, MONDAY };
const ClockReading Monday_5_AM = { 0, 5, 0, 0.0, MONDAY };
const ClockReading Monday_6_AM = { 0, 6, 0, 0.0, MONDAY };
const ClockReading Monday_7_AM = { 0, 7, 0, 0.0, MONDAY };
const ClockReading Monday_8_AM = { 0, 8, 0, 0.0, MONDAY };
const ClockReading Monday_9_AM = { 0, 9, 0, 0.0, MONDAY };
const ClockReading Monday_10_AM = { 0, 10, 0, 0.0, MONDAY };
const ClockReading Monday_11_AM = { 0, 11, 0, 0.0, MONDAY };

const ClockReading Monday_12_PM = { 0, 12, 0, 0.0, MONDAY };
const ClockReading Monday_1_PM = { 0, 13, 0, 0.0, MONDAY };
const ClockReading Monday_2_PM = { 0, 14, 0, 0.0, MONDAY };
const ClockReading Monday_3_PM = { 0, 15, 0, 0.0, MONDAY };
const ClockReading Monday_4_PM = { 0, 16, 0, 0.0, MONDAY };
const ClockReading Monday_5_PM = { 0, 17, 0, 0.0, MONDAY };
const ClockReading Monday_6_PM = { 0, 18, 0, 0.0, MONDAY };
const ClockReading Monday_7_PM = { 0, 19, 0, 0.0, MONDAY };
const ClockReading Monday_8_PM = { 0, 20, 0, 0.0, MONDAY };
const ClockReading Monday_9_PM = { 0, 21, 0, 0.0, MONDAY };
const ClockReading Monday_10_PM = { 0, 22, 0, 0.0, MONDAY };
const ClockReading Monday_11_PM = { 0, 23, 0, 0.0, MONDAY };

// --- Simulation Core Functions ---

Simulation* sim_malloc() {
    Simulation* sim = (Simulation*)malloc(sizeof(Simulation));
    if (!sim) {
        LOG_ERROR("Failed to allocate memory for Simulation");
        return NULL;
    }
    LOG_DEBUG("Allocated memory for simulation, size %.2f kilobytes.", sizeof(*sim) / 1024.0);
    return sim;
}

void sim_free(Simulation* self) {
    if (self) {
        free(self);
        LOG_DEBUG("Freed memory for simulation");
    } else {
        LOG_ERROR("Attempted to free a NULL Simulation pointer");
    }
}


void sim_init(Simulation* sim) {
    map_init(&sim->map);
    sim->initial_clock_reading = clock_reading(0, 8, 0, 0);  // 8:00 AM Monday
    sim->num_cars = 0;
    sim->time = 0.0;
    sim->dt = 0.02;
    sim->weather = WEATHER_SUNNY;
    sim->is_agent_enabled = false; // Agent car is disabled by default
    sim->is_agent_driving_assistant_enabled = false; // Driving assistant is disabled by default
    sim->agent_goal_lane_id = ID_NULL;
    sim->npc_rogue_factor = 0.0; // NPC cars are completely law-abiding by default

    // Initialize perception noise parameters with realistic defaults
    sim->perception_noise_distance_std_dev_percent = 0.05; // 5% distance error
    sim->perception_noise_speed_std_dev = from_mph(0.5); // 0.5 mph speed error
    sim->perception_noise_dropout_probability = 0.01; // 1% dropout probability
    sim->perception_noise_blind_spot_dropout_base_probability = 0.1; // 5% blind spot dropout probabilityelative positions. The base is the multiplier)

    sim->is_synchronized = false; // Not synchronized by default
    sim->is_paused = false; // Simulation is not paused by default
    sim->simulation_speedup = 1.0; // Default to real-time speed
    sim->should_quit_when_rendering_window_closed = false; // Default behavior is to not quit when rendering window is closed
    sim->render_socket = -1; // Invalid socket handle by default
    sim->is_connected_to_render_server = false; // Not connected to render server by default
    for (int i = 0; i < MAX_CARS_IN_SIMULATION; i++) {
        sim->situational_awarenesses[i].is_valid = false; // Initialize situational awareness for each car
        driving_assistant_reset_settings(&sim->driving_assistants[i], &sim->cars[i]); // Reset driving assistant settings for each car
    }
    sim_set_npc_rogue_factor(sim, 0.0); // Initialize NPC rogue factor and driving styles
}

Car* sim_get_new_car(Simulation* self) {
    if (self->num_cars >= MAX_CARS_IN_SIMULATION) {
        LOG_ERROR("Simulation is full, cannot add more cars");
        return NULL;
    }
    Car* car = &self->cars[self->num_cars++];
    car->id = self->num_cars - 1; // Assign a unique ID
    return car;
}

// --- Getters ---

Map* sim_get_map(Simulation* self) {
    return self ? &self->map : NULL;
}

ClockReading sim_get_initial_clock_reading(Simulation* self) {
    return self->initial_clock_reading;
}

Car* sim_get_cars(Simulation* self) {
    return self->cars;
}

int sim_get_num_cars(const Simulation* self) {
    return self ? self->num_cars : 0;
}

Car* sim_get_car(Simulation* self, CarId id) {
    if (!self || id < 0 || id >= self->num_cars) return NULL;
    return &self->cars[id];
}

SituationalAwareness* sim_get_situational_awareness(Simulation* self, CarId id) {
    if (!self || id < 0 || id >= self->num_cars) return NULL;
    return &self->situational_awarenesses[id];
}

// Procedure* sim_get_ongoing_procedure(Simulation* self, CarId id) {
//     if (!self || id < 0 || id >= self->num_cars) return NULL;
//     return &self->ongoing_procedures[id];
// }

DrivingAssistant* sim_get_driving_assistant(Simulation* self, CarId id) {
    if (!self || id < 0 || id >= self->num_cars) return NULL;
    return &self->driving_assistants[id];
}

double sim_get_npc_rogue_factor(Simulation* self) {
    return self ? self->npc_rogue_factor : 0.0;
}

Seconds sim_get_time(const Simulation* self) {
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

bool sim_is_agent_enabled(Simulation* self) {
    return self ? self->is_agent_enabled : false;
}

bool sim_is_agent_driving_assistant_enabled(Simulation* self) {
    return self ? self->is_agent_driving_assistant_enabled : false;
}

Car* sim_get_agent_car(Simulation* self) {
    if (!self) {
        LOG_ERROR("Attempted to get agent car from a NULL Simulation pointer");
        return NULL;
    }
    if (self->is_agent_enabled) {
        return sim_get_car(self, 0); // Agent car is always at index 0
    } else {
        LOG_DEBUG("Agent is not enabled, returning NULL");
        return NULL; // Agent is not enabled
    }
}

void sim_set_agent_enabled(Simulation* self, bool enabled) {
    if (self) {
        self->is_agent_enabled = enabled;
        LOG_INFO("Agent enabled: %s", enabled ? "true" : "false");
    } else {
        LOG_ERROR("Attempted to set agent enabled on a NULL Simulation pointer");
    }
}

void sim_set_agent_driving_assistant_enabled(Simulation* self, bool enabled) {
    if (self) {
        self->is_agent_driving_assistant_enabled = enabled;
        LOG_INFO("Agent driving assistant enabled: %s", enabled ? "true" : "false");
    } else {
        LOG_ERROR("Attempted to set agent driving assistant enabled on a NULL Simulation pointer");
    }
}

void sim_set_synchronized(Simulation* self, bool is_synchronized, double simulation_speedup) {
    if (self) {
        if (is_synchronized) {
            if (simulation_speedup <= 0) {
                LOG_ERROR("Invalid simulation speedup value: %.2f. It must be positive.", simulation_speedup);
                return;
            }
            self->is_synchronized = is_synchronized;
            self->simulation_speedup = simulation_speedup;
            LOG_INFO("Simulation synchronized with wall time at %.2fx speedup", simulation_speedup);
        } else {
            self->is_synchronized = false;
            self->simulation_speedup = 1.0; // Default to real-time speed
            LOG_INFO("Simulation set to run as fast as possible without synchronization");
        }
    } else {
        LOG_ERROR("Attempted to set synchronization on a NULL Simulation pointer");
    }
}

void sim_set_should_quit_when_rendering_window_closed(Simulation* self, bool should_quit) {
    if (self) {
        self->should_quit_when_rendering_window_closed = should_quit;
        LOG_DEBUG("Set should quit when rendering window closed to %s", should_quit ? "true" : "false");
    } else {
        LOG_ERROR("Attempted to set should quit on a NULL Simulation pointer");
    }
}

// --- Setters ---

void sim_set_dt(Simulation* self, Seconds dt) {
    if (self) {
        if (dt <= 0) {
            LOG_ERROR("Invalid dt value: %.2f. It must be positive.", dt);
            return;
        }
        self->dt = dt;
    } else {
        LOG_ERROR("Attempted to set dt on a NULL Simulation pointer");
    }
}

void sim_set_initial_clock_reading(Simulation* self, ClockReading clock) {
    if (self) self->initial_clock_reading = clock;
}

void sim_set_weather(Simulation* self, Weather weather) {
    if (self) self->weather = weather;
}

void sim_set_npc_rogue_factor(Simulation* self, double rogue_factor) {
    if (self) {
        if (rogue_factor < 0.0 || rogue_factor > 1.0) {
            LOG_ERROR("Invalid NPC rogue factor: %.2f. It must be between 0.0 and 1.0.", rogue_factor);
            return;
        }
        self->npc_rogue_factor = rogue_factor;
        LOG_INFO("Set NPC rogue factor to %.2f", rogue_factor);

        int reckless_limit = (int)(self->num_cars * self->npc_rogue_factor);
        int aggressive_limit = reckless_limit + (int)(self->num_cars * 2.0 * self->npc_rogue_factor);
        if (aggressive_limit > self->num_cars) aggressive_limit = self->num_cars;
        
        int remaining_after_agg = self->num_cars - aggressive_limit;
        int normal_limit = aggressive_limit + (int)(remaining_after_agg * 0.75);

        for (int i = 0; i < self->num_cars; i++) {
            if (i < reckless_limit) {
                LOG_TRACE("Car %d: Assigned driving style RECKLESS\n", i);
                self->driving_assistants[i].smart_das_driving_style = SMART_DAS_DRIVING_STYLE_RECKLESS;
            } else if (i < aggressive_limit) {
                LOG_TRACE("Car %d: Assigned driving style AGGRESSIVE\n", i);
                self->driving_assistants[i].smart_das_driving_style = SMART_DAS_DRIVING_STYLE_AGGRESSIVE;
            } else if (i < normal_limit) {
                LOG_TRACE("Car %d: Assigned driving style NORMAL\n", i);
                self->driving_assistants[i].smart_das_driving_style = SMART_DAS_DRIVING_STYLE_NORMAL;
            } else {
                LOG_TRACE("Car %d: Assigned driving style DEFENSIVE\n", i);
                self->driving_assistants[i].smart_das_driving_style = SMART_DAS_DRIVING_STYLE_DEFENSIVE;
            }
        }
    } else {
        LOG_ERROR("Attempted to set NPC rogue factor on a NULL Simulation pointer");
    }
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

void sim_step(Simulation* self) {
    if (self) sim_integrate(self, self->dt);
}
