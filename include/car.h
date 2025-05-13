#pragma once

#include "road.h"

// Acceleration model is used to determine PID control proportionality constant. k = (4/stopping_time)^2
enum AccelerationMode {
    CHILL_MODE,
    STANDARD_MODE,
    SPORT_MODE,
    EMERGENCY_MODE,
};
typedef enum AccelerationMode AccelerationMode;
MetersPerSecond acceleration_mode_settling_time(AccelerationMode mode);
double acceleration_mode_pid_k(AccelerationMode mode);

// Describes the car's personality and personalized reward/cost function and determines the user's happiness only.
struct PreferenceProfile {
    int lane;                           // lane preference. The user likes to stay in this lane and every second not in this lane incurs cost.
    MetersPerSecond turn_speed;         // turn speed preference. Use is upset if the car turns at a speed lower or greater than this speed. cost = (turn_speed - true_speed)^2.
    double scenic;                      // scenic preference between 0.0 and 1.0. User reward for this = scenic x time spent on scenic roads.
    Meters distance_to_front_vehicle;   // distance to front vehicle preference. User is happy if this distance is respected. Cost = preferred distance - true distance when true distance is smaller.
    Meters distance_to_back_vehicle;    // distance to back vehicle preference. User is happy if this distance is respected. Cost = preferred distance - true distance when true distance is smaller.
    bool run_yellow_light;              // run yellow light preference. User is disappointed if the car slows down on seeing yellow light.
    bool turn_right_on_red;             // turn right on red preference. User is upset if the car turns right when it is red, despite this flag.
    double aversion_to_rough_terrain;   // aversion to rough terrain. Cost = aversion x time on rough terrain.
    double aversion_to_traffic_jam;     // aversion to traffic jam. Being stuck in traffic jam carries cost = aversion x time spent in a traffic jam.
    double aversion_to_construction_zone;   // aversion to construction zone.   Encountering a traffic signal causes this much cost.
    double aversion_to_traffic_light;   // aversion to traffic light. Encountering a traffic signal causes this much cost.
    double aversion_to_lane_change;     // aversion to lane change causes this much cost.
    MetersPerSecondSquared max_acceleration;    // max acceleration preference. User is upset if the car accelerates faster than this.
    MetersPerSecondSquared max_deceleration;    // max deceleration preference. User is upset if the car decelerates faster than this.
    MetersPerSecond average_speed_offset;       // user likes to drive at speed limit + this offset for the most part. Cost = (true_speed - preffered_speed)^2.
    MetersPerSecond min_speed_offset;           // user is upset if car drives below speed limit + this offset. Exponential cost for driving slower than this.
    MetersPerSecond max_speed_offset;           // user is scared if car drives above speed limit + this offset. Exponential cost for driving faster than this.
    AccelerationMode acceleration_mode;             // CHILL_MODE, STANDARD_MODE, SPORT_MODE.
};
typedef struct PreferenceProfile PreferenceProfile;

struct CarCapabilities {
    MetersPerSecondSquared max_acceleration;     // max acceleration capability. The simulation engine will not execute an acceleration higher than this number.
    MetersPerSecondSquared max_deceleration;     // max deceleration (braking) capability. The simulation engine will not execute a deceleration (braking) higher than this number. i.e., the car can only specify acceleration between [-max_deceleration, max_acceleration] in its make_decisions function.
    MetersPerSecond top_speed;                  // max speed capability. The simulation will clamp the speed of the car to this value if physics suggests higher speeds.
    Seconds reaction_time_delay;    // TODO: implement this in the sim
};
typedef struct CarCapabilities CarCapabilities;


enum DirectionIntent {
    INTENT_STRAIGHT,        // Go straight or do not change lane
    INTENT_LEFT,            // Turn left or change to the left lane
    INTENT_RIGHT,           // Turn right or change to the right lane
    INTENT_NA,              // No intent: not applicable
};
typedef enum DirectionIntent DirectionIntent;

typedef enum {
    GO_GREEN,
    GO_YELLOW,
    GO_RED,
    GO_STOP,    // typically 4-way stop sign
    GO_ILLEGAL, // could be end of lane, or dead end, or just some illegal thing (other than stop sign or red light)
} GoSituation;

typedef struct {
    GoSituation state;
    Seconds countdown;
} GoSituationWithCountdown;


struct NPCState {
    bool braking;
    bool cruising;
    bool approach_mode;
    MetersPerSecond speed_error;
    Meters distance_error;
    double pid_k_braking;
};
typedef struct NPCState NPCState;

struct Car {
    // ---- Car's fixed properties ---
    Dimensions dimensions;      // width and length of the car
    CarCapabilities capabilities;   // capabilities of the car
    PreferenceProfile preferences;   // Preference profile of the car.

    // --- Car's state variables ---
    const Lane* lane;                 // current lane
    double lane_progress;       // progress in the lane as a fraction of the lane length.
    MetersPerSecond speed;      // current speed. Should be set by the simulation engine.
    MetersPerSecondSquared acceleration;    // current acceleration. Should be determined by the car's decision-making logic when car_make_decision() is called. acceration must be clipped between -max_deceleration_capability and max_acceleration_capability.
    DirectionIntent turn_intent;            // turn intent (left, right, straight) on approaching an intersection. Should be set by the car's decision-making logic when car_make_decision() is called.
    DirectionIntent lane_change_intent;     // lane change intent (left, right, straight). Should be set by the car's decision-making logic when car_make_decision() is called.
    double damage;              // damage level of the car. 0.0 for no damage and 1.0 for total damage.
    NPCState npc_state;               // An integer state useful to track what the NPC is doing. e.g., cruising, changing lanes, turning, slowing for intersection etc.
};
typedef struct Car Car;


Car* car_create(const Dimensions dimensions, const CarCapabilities capabilities, const PreferenceProfile preferences);


// ---- setters -----

// Sets the lane of the car
// The lane will not be modified. `lane_remove_car` and `lane_add_car` should be called by the simulation engine.
void car_set_lane(Car* self, const Lane* lane);

// Sets the lane progress of the car. The progress should be between 0.0 and 1.0.
void car_set_lane_progress(Car* self, const double progress);

// Sets the speed of the car. The speed should be between 0.0 and max_speed_capability.
void car_set_speed(Car* self, const MetersPerSecond speed);

// Sets the acceleration of the car. The acceleration should be between -max_deceleration_capability and max_acceleration_capability. It will be clipped to this range.
void car_set_acceleration(Car* self, const MetersPerSecondSquared acceleration);

// Sets the turn intent of the car. if the turn is not possible, the intent will be auto corrected to the closest possible turn if auto_correct is true. If auto_correct is false, error will be thrown. For auto correcting straight, the right turn will be preferred.
void car_set_turn_intent(Car* self, const DirectionIntent intent, bool auto_correct);

// Sets the lane change intent of the car.
void car_set_lane_change_intent(Car* self, const DirectionIntent intent);

// Sets the damage level of the car. The damage should be between 0.0 and 1.0.
void car_set_damage(Car* self, const double damage);



// ---- getters ----

// Returns the length of the car in meters.
Meters car_get_length(const Car* self);

// Returns the width of the car in meters.
Meters car_get_width(const Car* self);

// Returns the lane of the car.
const Lane* car_get_lane(const Car* self);

// Returns the lane progress of the car.
double car_get_lane_progress(const Car* self);

// Returns car progress x lane length in meters.
Meters car_get_lane_progress_meters(const Car* self);

// Returns the speed of the car.
MetersPerSecond car_get_speed(const Car* self);

// Returns the acceleration of the car. The acceleration should be between -max_deceleration_capability and max_acceleration_capability. A clipped value is returned.
MetersPerSecondSquared car_get_acceleration(const Car* self);

// Returns the turn intent of the car.
DirectionIntent car_get_turn_intent(const Car* self);

// Returns the lane change intent of the car.
DirectionIntent car_get_lane_change_intent(const Car* self);

double car_get_damage(const Car* self);



// --- Decision-making functions -------------
// ------------------------------------------

// ------ forward declarations, i.e., assume such structs and functions exist -----
typedef struct Simulation Simulation;
Seconds sim_get_dt(const Simulation* sim);
Car** sim_get_cars(Simulation* sim);
int sim_get_num_cars(const Simulation* sim);
// ---------------------------------


// Should set car's acceleration, lane_change_intent, turn_intent and must not modify any other variable anywhere. The sim object contains info about other cars, map and time.
void NPC_car_make_decisions(Car* self, const Simulation* sim);

// Utility function. Returns true if switching immediately to `target_lane` would place us dangerously close to another car in that lane, within `buffer_m` meters of lane progress.
bool would_collide_on_lane_change(const Car* self, const Lane* target_lane, const Meters buffer_m);

// Returns true if an adjacent left lane with same travel direction exits.
bool is_left_lane_change_possible(const Lane* lane);

// Returns true if an adjacent right lane with same travel direction exits.
bool is_right_lane_change_possible(const Lane* lane);

// Computes acceleration to reduce distance and speed errors using a critically damped PD controller.
// a = 2 * sqrt(k) * speed_error + k * distance_error.
MetersPerSecondSquared pid_acceleration(const Meters distance_error, const MetersPerSecond speed_error, const double pid_k);

// returns the mininum proportional gain required for critical damping with these errors.
double pid_min_k(const Meters distance_error, const MetersPerSecond speed_error);

// What should be the indicator if the car wants to go from `lane_from` to `lane_to`? 
DirectionIntent turn_intent_given_next_lane(const Lane* lane_from, const Lane* lane_to);

// whether it is possible to turn from `lane_from` towards `turn_intent` based on the lane's connections.
bool is_turn_possible(const Lane* lane_from, DirectionIntent turn_intent);

// If the turn is possible, it will return the same turn intent. If the turn is not possible, this function will return the closest possible turn intent. For input INTENT_RIGHT, it will try INTENT_STRAIGHT first and then INTENT_LEFT. For input INTENT_LEFT, it will try INTENT_STRAIGHT first and then INTENT_RIGHT. For input INTENT_STRAIGHT, it will try INTENT_RIGHT first and then INTENT_LEFT. If no turn is possible, it will return INTENT_NA.
DirectionIntent get_legitimate_turn_intent_closest_to(const Lane* lane_from, DirectionIntent turn_intent);

// While lane does this turn intent lead to, assuming the turn is possible?
const Lane* turn_intent_leads_to_lane(const Lane* lane, DirectionIntent turn_intent);


GoSituationWithCountdown go_situation_compute(const Lane* from_lane, DirectionIntent turn_intent);

// ----------- init and free ------------
void car_free(Car* self);




// ----------------- presets --------------------
// ----------------------------------------------

Dimensions car_dimensions_typical(void); // Typical car (e.g., midsize sedan, 5.8 ft wide, 14.7 ft long)

Dimensions car_dimensions_compact(void); // Compact car (e.g., Honda Civic, 5.6 ft wide, 14 ft long)

Dimensions car_dimensions_midsize(void); // Midsize sedan (e.g., Toyota Camry, 6.0 ft wide, 15.7 ft long)

Dimensions car_dimensions_fullsize(void); // Full-size car (e.g., Dodge Charger, 6.0 ft wide, 18 ft long)

Dimensions car_dimensions_compact_suv(void); // Compact SUV (e.g., Honda CR-V, 6.0 ft wide, 16 ft long)

Dimensions car_dimensions_large_suv(void); // Large SUV (e.g., Chevrolet Traverse, 6.5 ft wide, 17.5 ft long)

Dimensions car_dimensions_pickup(void); // Full-size pickup (e.g., Ford F-150, 7.0 ft wide, 18.4 ft long)

Dimensions car_dimensions_random_preset(void);

CarCapabilities car_capable_max(void); // A car with infinite capabilities

PreferenceProfile preferences_random(void);

PreferenceProfile preferences_random_with_average_speed_offset(const MetersPerSecond average_speed_offset);
