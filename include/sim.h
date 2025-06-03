#pragma once

#include "utils.h"
#include "map.h"
#include "car.h"
#include "map.h"

#define MAX_CARS_IN_SIMULATION 1024 // Maximum number of cars in the simulation
#define MAX_NUM_HAZARDS_EACH_TYPE 1024 // Maximum number of hazards of each type

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

extern const char* day_of_week_strings[];

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

extern const char* weather_strings[];

// Main simulation structure
typedef struct Simulation {
    Map map;       // Simulation map
    ClockReading initial_clock_reading; // Start clock
    Car cars[MAX_CARS_IN_SIMULATION];  // Car list; car[0] = agent
    int num_cars;   // Number of cars in simulation
    Seconds time;   // Time simulated so far
    Seconds dt;     // Simulation engine's time resolution for integration (in seconds)
    Weather weather;
    bool is_agent_enabled; // Whether car 0 is an agent (true) or an NPC (false). False by default. When true, car 0 is the agent car and must be controlled using car_set_acceleration, car_set_indicator_turn, and car_set_indicator_lane functions outside the sim loop.
} Simulation;

 // Allocate memory for a new Simulation instance. The memory is not initialized and contains garbage values.
Simulation* sim_malloc();
 // Free memory for a Simulation instance
void sim_free(Simulation* self);

// Initialize the simulation instance, setting up an empty map and default initial conditions (clock = 8:00 AM on monday, weather = sunny, dt = 0.02 seconds).
void sim_init(Simulation* sim);

// Add a car to the simulation
Car* sim_get_new_car(Simulation* self);

// --- Getters ---
Map* sim_get_map(Simulation* self);
ClockReading sim_get_initial_clock_reading(Simulation* self);
int sim_get_num_cars(const Simulation* self);
Car* sim_get_car(Simulation* self, int id);
Seconds sim_get_time(Simulation* self);
Seconds sim_get_dt(const Simulation* self);
Weather sim_get_weather(Simulation* self);
ClockReading sim_get_clock_reading(Simulation* self);
DayOfWeek sim_get_day_of_week(const Simulation* self);
bool sim_is_agent_enabled(Simulation* self);
// returns NULL if the agent is not enabled
Car* sim_get_agent_car(Simulation* self);

// --- Setters ---
void sim_set_dt(Simulation* self, Seconds dt);
void sim_set_initial_clock_reading(Simulation* self, ClockReading clock);
void sim_set_weather(Simulation* self, Weather weather);
void sim_set_agent_enabled(Simulation* self, bool enabled);

// --- Time Queries ---
bool sim_is_weekend(Simulation* self);
bool sim_is_morning(Simulation* self);
bool sim_is_afternoon(Simulation* self);
bool sim_is_evening(Simulation* self);
bool sim_is_night(Simulation* self);

// --- Utility ---

// Returns sunlight intensity (0–1) based on time of day
double sunlight_intensity(Simulation* self);

// Advance simulation state by a duration
void simulate(Simulation* self, Seconds duration);

// Advance simulation by one timestep (dt)
void sim_step(Simulation* self);


struct TrackIntersectionPoint {
    const Lane* lane1;  // Lane at the intersection point
    const Lane* lane2;  // Second lane at the intersection point
    double progress1;   // Progress along lane1
    double progress2;   // Progress along lane2
    bool exists;        // Whether the intersection point exists
};
typedef struct TrackIntersectionPoint TrackIntersectionPoint;

TrackIntersectionPoint track_intersection_check(const Lane* lane1, const Lane* lane2);
// TrackIntersectionPoint track_intersection_check_linear_linear(const LinearLane* lane1, const LinearLane* lane2);
// TrackIntersectionPoint track_intersection_check_linear_curved(const LinearLane* lane1, const QuarterArcLane* lane2);
// TrackIntersectionPoint track_intersection_check_curved_curved(const QuarterArcLane* lane1, const QuarterArcLane* lane2);


// struct CarCollision {
//     const Car* car1;    // First car involved in the collision
//     const Lane* lane1;  // Lane of the first car
//     double progress1;   // Progress along the lane of the first car
//     const Car* car2;    // Second car involved in the collision
//     const Lane* lane2;  // Lane of the second car
//     double progress2;   // Progress along the lane of the second car
//     Seconds time;       // Time of the collision
// };
// typedef struct CarCollision CarCollision;





// struct DeadEnd {
//     const Lane* lane; // Lane that is a dead end
// };
// typedef struct DeadEnd DeadEnd;

// struct Hazards {
//     const TrackIntersectionPoint* points[MAX_NUM_HAZARDS_EACH_TYPE]; // Array of intersection points
//     const DeadEnd* dead_ends[MAX_NUM_HAZARDS_EACH_TYPE]; // Array of dead ends
//     int num_intersection_points; // Number of intersection points
//     int num_dead_ends; // Number of dead ends
// };
// typedef struct Hazards Hazards;


// Hazards* hazards_create();
// Hazards* hazards_create_from_map(const Map* map);
// void hazards_free(Hazards* self);
// void hazards_add_intersection_point(Hazards* self, const TrackIntersectionPoint* point);
// void hazards_add_dead_end(Hazards* self, const DeadEnd* dead_end);