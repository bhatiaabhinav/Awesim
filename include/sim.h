#pragma once

#include "constants.h"
#include "utils.h"
#include "map.h"
#include "car.h"
#include "map.h"
#include "ai.h"

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

extern const ClockReading Monday_12_AM, Monday_1_AM, Monday_2_AM, Monday_3_AM, Monday_4_AM, Monday_5_AM, Monday_6_AM, Monday_7_AM, Monday_8_AM, Monday_9_AM, Monday_10_AM, Monday_11_AM;
extern const ClockReading Monday_12_PM, Monday_1_PM, Monday_2_PM, Monday_3_PM, Monday_4_PM, Monday_5_PM, Monday_6_PM, Monday_7_PM, Monday_8_PM, Monday_9_PM, Monday_10_PM, Monday_11_PM;

// Weather conditions enum
typedef enum {
    WEATHER_SUNNY,
    WEATHER_CLOUDY,
    WEATHER_RAINY,
    WEATHER_FOGGY,
    WEATHER_SNOWY
} Weather;

extern const char* weather_strings[];


typedef enum {
    TRAFFIC_VIOLATION_RED_LIGHT_RUN,    // while entering the lane on the intersection, was the lane guarded by a red light? If so, red light violation
    TRAFFIC_VIOLATION_RED_YIELD_FAIL_TO_STOP, // while entering the lane on the intersection, was the lane guarded by a red yield sign and did the car fail to stop completely in the stop zone or shortly after (before the leading edge reached the end of the lane)?
    TRAFFIC_VIOLATION_TAILGATING,     // was the car tailgating the lead car (i.e. was the distance to the lead car less than TAILGATING_TIME_HEADWAY_THRESHOLD second headway) while the car's speed was above TAILGATING_MIN_SPEED_MPS and it was not in the process of braking? Detected with probability = TAILGATING_DETECTION_PROBABILITY * dt. 
    TRAFFIC_VIOLATION_STOP_SIGN_FAIL_TO_STOP,    // did the car fail to come to a full stop in the stop sign zone? If so, stop sign violation. The detail field can be used to store the exact lowest speed in the stop sign zone. See car->lowest_speed_in_stop_zone, or situational_awareness->did_stop_in_stop_zone.
    TRAFFIC_VIOLATION_OVERSPEEDING,     // was the car's speed above the lane's speed limit by more than OVERSPEEDING_THRESHOLD. Detected with probability = OVERSPEEDING_DETECTION_PROBABILITY * dt.
    TRAFFIC_VIOLATION_NUM_TYPES
} TrafficViolationType;

typedef struct TrafficViolation {
    CarId car_id;
    TrafficViolationType type;
    Seconds time;
    double detail;  // for red light, exact speed. For stop sign, lowest speed in the stop zone. For overspeeding, how much the speed exceeded the limit. For tailgating, the time headway to the lead car. For red yield sign violation, the lowest speed in the stop zone.
} TrafficViolation;

typedef struct TrafficViolationsLogsQueue {
    bool enabled[MAX_CARS_IN_SIMULATION];                   // Whether logging is enabled for each car ID. If false, no violations will be logged for that car.
    TrafficViolation queue[MAX_CARS_IN_SIMULATION][TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE];     // For each car ID, a circular buffer queue of the most recent traffic violations involving that car.
    int next_index[MAX_CARS_IN_SIMULATION];                // For each car ID, the next index in the circular buffer queue to write to. After writing, this index is incremented and wrapped around using modulo with TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE.
    int num_violations_each_type[MAX_CARS_IN_SIMULATION][TRAFFIC_VIOLATION_NUM_TYPES];              // For each car ID, the total number of violations that have been logged for that car since the beginning of the sim, for each type of violation. These may exceed TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE. Queue clearing functions do not reset this value.
    int queue_num_items[MAX_CARS_IN_SIMULATION];          // For each car ID, the current number of violations stored in the queue, which is at most TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE.
} TrafficViolationsLogsQueue;

void traffic_violations_set_enabled(TrafficViolationsLogsQueue* queue, CarId car_id, bool enabled);
TrafficViolation traffic_violation_get(TrafficViolationsLogsQueue* queue, CarId car_id, int violation_index_from_most_recent);
void traffic_violation_log(TrafficViolationsLogsQueue* queue, CarId car_id, TrafficViolationType type, Seconds time, double detail);
int traffic_violation_get_total_count(TrafficViolationsLogsQueue* queue, CarId car_id);
int traffic_violation_type_get_total_count(TrafficViolationsLogsQueue* queue, CarId car_id, TrafficViolationType type);

void traffic_violations_logs_queue_clear(TrafficViolationsLogsQueue* queue, CarId car_id);


// Main simulation structure
typedef struct Simulation {
    Map map;       // Simulation map
    ClockReading initial_clock_reading; // Start clock
    Car cars[MAX_CARS_IN_SIMULATION];  // Car list; car[0] = agent
    SituationalAwareness situational_awarenesses[MAX_CARS_IN_SIMULATION]; // Situational awareness for each car
    // Procedure ongoing_procedures[MAX_CARS_IN_SIMULATION]; // Ongoing procedures for each car
    DrivingAssistant driving_assistants[MAX_CARS_IN_SIMULATION]; // Driving assistants for each car
    TrafficViolationsLogsQueue traffic_violations_logs_queue; // Queue for logging traffic violations for each car
    int num_cars;   // Number of cars in simulation
    Seconds time;   // Time simulated so far
    Seconds dt;     // Simulation engine's time resolution for integration (in seconds)
    long long step_count; // Number of steps simulated so far
    Weather weather;
    bool is_agent_enabled; // Whether car 0 is an agent (true) or an NPC (false). False by default. When true, car 0 is the agent car and must be controlled using car_set_acceleration, car_set_indicator_turn, and car_set_indicator_lane functions outside the sim loop.
    bool is_agent_driving_assistant_enabled; // Whether the agent car has a driving assistant enabled, the simulator will automatically set the agent car's controls using the driving assistant within the sim loop. The driving assistant must be configured periodically outside the sim loop.
    LaneId agent_goal_lane_id; // Goal lane ID for the agent car, if any.
    double npc_rogue_factor; // Rogue factor for NPC cars, between 0.0 (completely law-abiding) to 1.0 (maximum rogue). Default is 0.0

    // Perception Noise Parameters
    double perception_noise_distance_std_dev_percent; // Standard deviation of distance perception noise as a percentage of distance
    double perception_noise_speed_std_dev; // Standard deviation of speed perception noise in m/s
    double perception_noise_dropout_probability; // Probability of failing to perceive a vehicle
    double perception_noise_blind_spot_dropout_base_probability; // Base probability factor for blind spot dropout

    bool is_synchronized; // Whether the simulation is synchronized with wall time. If true, the simulation will run in real-time, simulating `simulation_speedup` seconds of simulation time per second of wall time. If false, the simulation will run as fast as possible without synchronization.
    bool is_paused; // Whether the simulation is paused. If true, the simulation will not advance time until it is resumed.
    double simulation_speedup; // How many seconds of simulation time to simulate per second of wall time. Default is 1.0, meaning real-time simulation.
    bool should_quit_when_rendering_window_closed; // Whether the simulation should quit when the rendering window is closed. If true, the `simulate` function will return when the rendering window is closed, allowing the simulation to exit gracefully. If false, the simulation will continue running even if the rendering window is closed, allowing for background processing or other tasks to continue.
    int render_socket; // Socket handle for render server connection
    bool is_connected_to_render_server; // Flag indicating connection status
} Simulation;

 // Allocate memory for a new Simulation instance. The memory is not initialized and contains garbage values.
Simulation* sim_malloc();
 // Free memory for a Simulation instance
void sim_free(Simulation* self);

// Initialize the simulation instance, setting up an empty map and default initial conditions (clock = 8:00 AM on monday, weather = sunny, dt = 0.02 seconds, no cars, no agent enabled, not synchronized).
void sim_init(Simulation* sim);

// Add a car to the simulation
Car* sim_get_new_car(Simulation* self);

// Function to connect to render server
bool sim_connect_to_render_server(Simulation* self, const char* server_ip, int port);
// Function to disconnect from render server
void sim_disconnect_from_render_server(Simulation* self);
// The render server may issue these commands to the simulation
typedef enum {
    COMMAND_NONE,
    COMMAND_QUIT,
    COMMAND_SIM_INCREASE_SPEED,
    COMMAND_SIM_DECREASE_SPEED,
    COMMAND_SIM_RESET_SPEED,
    COMMAND_SIM_PAUSE
} SimCommand;

// --- Getters ---
Map* sim_get_map(Simulation* self);
ClockReading sim_get_initial_clock_reading(Simulation* self);
int sim_get_num_cars(const Simulation* self);
Car* sim_get_cars(Simulation* self);
Car* sim_get_car(Simulation* self, CarId id);
SituationalAwareness* sim_get_situational_awareness(Simulation* self, CarId id);
// Procedure* sim_get_ongoing_procedure(Simulation* self, CarId id);
DrivingAssistant* sim_get_driving_assistant(Simulation* self, CarId id);
double sim_get_npc_rogue_factor(Simulation* self);
Seconds sim_get_time(const Simulation* self);
Seconds sim_get_dt(const Simulation* self);
Weather sim_get_weather(Simulation* self);
ClockReading sim_get_clock_reading(Simulation* self);
DayOfWeek sim_get_day_of_week(const Simulation* self);
bool sim_is_agent_enabled(Simulation* self);
bool sim_is_agent_driving_assistant_enabled(Simulation* self);
// returns NULL if the agent is not enabled
Car* sim_get_agent_car(Simulation* self);

// --- Setters ---
void sim_set_dt(Simulation* self, Seconds dt);
void sim_set_initial_clock_reading(Simulation* self, ClockReading clock);
void sim_set_weather(Simulation* self, Weather weather);
void sim_set_agent_enabled(Simulation* self, bool enabled);
void sim_set_agent_driving_assistant_enabled(Simulation* self, bool enabled);
void sim_set_synchronized(Simulation* self, bool is_synchronized, double simulation_speedup);
void sim_set_should_quit_when_rendering_window_closed(Simulation* self, bool should_quit);
void sim_set_npc_rogue_factor(Simulation* self, double rogue_factor);
// --- Time Queries ---
bool sim_is_weekend(Simulation* self);
bool sim_is_morning(Simulation* self);
bool sim_is_afternoon(Simulation* self);
bool sim_is_evening(Simulation* self);
bool sim_is_night(Simulation* self);

// --- Utility ---

// Returns sunlight intensity (0–1) based on time of day
double sunlight_intensity(Simulation* self);

// Compute average fuel consumption for a car over a small time interval, with typical car and environment parameters.
Liters fuel_consumption_compute_typical(MetersPerSecond initial_velocity, MetersPerSecondSquared accel, Seconds delta_time);

// Integrate the simulation for a given duration in (simulated) seconds with dt resolution, causing sim_duration / dt transition updates. This will advance the simulation `time` by `sim_duration` seconds.
void sim_integrate(Simulation* self, Seconds sim_duration);

// Advance simulation by a duration in (simulated) seconds. If `self->is_synchronized` is true, the simulation will run in real-time, simulating `self->simulation_speedup` seconds of simulation time per second of wall time. If `is_synchronized` is false, the simulation will run as fast as possible without synchronization, making this function identical to `sim_integrate(self, sim_duration)`.
void simulate(Simulation* self, Seconds sim_duration);

// Advance simulation by one timestep (dt)
void sim_step(Simulation* self);




// Collisions stuff:

// Sparse collision storage: track up to two current collisions per car.
typedef struct Collisions {
    CarId colliding_cars[MAX_CARS_IN_SIMULATION][COLLISIONS_MAX_PER_CAR]; // Packed: indices [0, num_collisions_per_car) are valid IDs, rest are -1.
    int num_collisions_per_car[MAX_CARS_IN_SIMULATION];
    int total_collisions; // unique pairs currently tracked
} Collisions;

Collisions* collisions_malloc();
void collisions_free(Collisions* collisions);
void collisions_reset(Collisions* collisions);

int collisions_get_total(Collisions* collisions);
int collisions_get_count_for_car(Collisions* collisions, CarId car_id);
bool collisions_are_colliding(Collisions* collisions, CarId car1, CarId car2);
CarId collisions_get_colliding_car(Collisions* collisions, CarId car_id, int collision_index);

void collisions_detect_all(Collisions* collisions, Simulation* sim);
void collisions_detect_for_car(Collisions* collisions, Simulation* sim, CarId car_id);
void collisions_print(Collisions* collisions);

