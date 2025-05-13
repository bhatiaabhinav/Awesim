#pragma once

#include "utils.h"
#include "road.h"
#include "car.h"
#include "map.h"

#define MAX_CARS_IN_SIMULATION 1024 // Maximum number of cars in the simulation

// Time-of-day phase start hours
#define MORNING_START 6
#define AFTERNOON_START 12
#define EVENING_START 16
#define NIGHT_START 20

typedef int Minutes;
typedef int Hours;
typedef int Days;

// Enum representing days of the week
typedef enum {
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    FRIDAY,
    SATURDAY,
    SUNDAY
} DayOfWeek;

// Structure for time representation as DD:HH:MM:SS 
typedef struct {
    Days days;
    Hours hours;
    Minutes minutes;
    Seconds seconds;
    DayOfWeek day_of_week; // 0 = Monday, 1 = Tuesday, ..., 6 = Sunday
} ClockReading;

// Constructs a ClockReading from components
ClockReading clock_reading(const Days days, const Hours hours, const Minutes minutes, const Seconds seconds);

// Converts seconds to ClockReading (throws error if < 0)
ClockReading clock_reading_from_seconds(const Seconds seconds);

// Converts a ClockReading to seconds
Seconds clock_reading_to_seconds(const ClockReading clock);

// Adds two ClockReadings
ClockReading clock_reading_add(const ClockReading clock1, const ClockReading clock2);

// Adds seconds to a ClockReading
ClockReading clock_reading_add_seconds(const ClockReading clock, const Seconds seconds);

// Weather conditions enum
typedef enum {
    WEATHER_SUNNY,
    WEATHER_CLOUDY,
    WEATHER_RAINY,
    WEATHER_FOGGY,
    WEATHER_SNOWY
} Weather;

// Main simulation structure
typedef struct Simulation {
    Map* map;       // Simulation map
    ClockReading initial_clock_reading; // Start clock
    Car* cars[MAX_CARS_IN_SIMULATION];  // Car list; car[0] = agent
    int num_cars;   // Number of cars in simulation
    Seconds time;   // Time simulated so far
    Seconds dt;     // Simulation timestep
    Weather weather;
} Simulation;

// Create a new simulation instance
Simulation* sim_create(Map* map, double dt);

// Add a car to the simulation
void sim_add_car(Simulation* self, Car* car);

// --- Getters ---
Map* sim_get_map(Simulation* self);
ClockReading sim_get_initial_clock_reading(Simulation* self);
Car** sim_get_cars(Simulation* self);
int sim_get_num_cars(const Simulation* self);
Car* sim_get_car(Simulation* self, int id);
Seconds sim_get_time(Simulation* self);
Seconds sim_get_dt(const Simulation* self);
Weather sim_get_weather(Simulation* self);
ClockReading sim_get_clock_reading(Simulation* self);
DayOfWeek sim_get_day_of_week(const Simulation* self);
Car* sim_get_agent_car(Simulation* self);

// --- Setters ---
void sim_set_initial_clock_reading(Simulation* self, ClockReading clock);
void sim_set_weather(Simulation* self, Weather weather);

// --- Time Queries ---
bool sim_is_weekend(Simulation* self);
bool sim_is_morning(Simulation* self);
bool sim_is_afternoon(Simulation* self);
bool sim_is_evening(Simulation* self);
bool sim_is_night(Simulation* self);

// --- Utility ---
bool sim_car_exists(const Simulation* self, int id);

// Returns sunlight intensity (0–1) based on time of day
double sunlight_intensity(Simulation* self);

// Advance simulation state by a duration
void simulate(Simulation* self, Seconds duration);

// Advance simulation by one timestep (dt)
void sim_step(Simulation* self);

// Cleanup simulation object (does not free map, roads, cars etc.)
void sim_free(Simulation* self);

// Cleanup simulation resources and free all cars, roads, and map
void sim_deep_free(Simulation* self);

