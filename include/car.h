#pragma once

#include "utils.h"
#include "map.h"


// --- Accleration profile ------

// All numbers are absolute (>0).
struct CarAccelProfile {
    MetersPerSecondSquared max_acceleration;
    MetersPerSecondSquared max_acceleration_reverse_gear;
    MetersPerSecondSquared max_deceleration;
};
typedef struct CarAccelProfile CarAccelProfile;

// effective acceleration profile of a car is the minimum of the capable and preferred profiles.
CarAccelProfile car_accel_profile_get_effective(CarAccelProfile capable, CarAccelProfile preferred);



// --- Car capabilities ------

struct CarCapabilities {
    CarAccelProfile accel_profile;
    MetersPerSecond top_speed;
    Seconds reaction_time_delay;
};
typedef struct CarCapabilities CarCapabilities;



// --- Car indicator to indicate lane and turn intent ------
enum CarIndicator {
    INDICATOR_LEFT,         // Turn left or change to the left lane
    INDICATOR_NONE,         // Go straight or do not change lane
    INDICATOR_RIGHT,        // Turn right or change to the right lane
};
typedef enum CarIndicator CarIndicator;



// ---- Car preference profile ------

// Describes the car's personality and personalized reward/cost function and determines the user's happiness only.
struct CarPersonality {
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
    MetersPerSecond average_speed_offset;       // user likes to drive at speed limit + this offset for the most part. Cost = (true_speed - preffered_speed)^2.
    MetersPerSecond min_speed_offset;           // user is upset if car drives below speed limit + this offset. Exponential cost for driving slower than this.
    MetersPerSecond max_speed_offset;           // user is scared if car drives above speed limit + this offset. Exponential cost for driving faster than this.
    CarAccelProfile acceleration_profile;       // comfort, normal, eco, sport, aggressive.
    Seconds time_headway;                // time headway preference. User is happy if the car maintains this time headway from the front vehicle. Cost = (time_headway - true_time_headway)^2.
};
typedef struct CarPersonality CarPersonality;

CarPersonality preferences_sample_random(void);



// -------- Car struct -----------

struct Car {
    // fixed properties:
    CarId id;                         // unique identifier for the car
    Dimensions dimensions;          // width and length of the car
    Liters fuel_tank_capacity;       // fuel tank capacity in liters
    CarCapabilities capabilities;   // capabilities of the car
    CarPersonality preferences;     // Preference profile of the car.

    // state variables:
    LaneId lane_id;                   // current lane
    double lane_progress;          // progress in the lane as a fraction of the lane length.
    Meters lane_progress_meters;   // progress in the lane in meters. Should be set by the simulation engine. Is a product of lane_progress and lane length.
    int lane_rank;              // rank of the car in the lane, where 0 is the first car in the lane (the one closest to the end of the lane), and num_cars-1 is the last car in the lane (the one closest to the start of the lane).
    MetersPerSecond speed;              // current speed. Should be set by the simulation engine.
    double damage;                      // damage level of the car. 0.0 for no damage and 1.0 for total damage.
    Liters fuel_level;                  // fuel level of the car in liters. If fuel level is 0.0, the car cannot accelerate in the direction of motion, but can decelerate (brake). The fuel level will be set by the simulation engine.

    // control variables:
    MetersPerSecondSquared acceleration;    // current acceleration. Should be determined by the car's decision-making logic when car_make_decision() is called.
    CarIndicator indicator_lane;                  // Lane change indicator. This is purely an intent signal and does not request the sim to execute the lane change. Should be set by the car's decision-making logic when car_make_decision() is called.
    CarIndicator indicator_turn;                  // Turn indicator. This is purely an intent signal and does not request the sim to execute the turn. Should be set by the car's decision-making logic when car_make_decision() is called.
    bool request_indicated_lane;        // Whether the car requests to change lane as per the indicator_lane. This is set by the car's decision-making logic when car_make_decision() is called.
    bool request_indicated_turn;        // Whether the car requests to turn as per the indicator_turn. This is set by the car's decision-making logic when car_make_decision() is called.

    bool auto_turn_off_indicators; // Whether the sim should automatically turn off the car's indicators after the requested lane change or turn is completed.

    // history:
    LaneId prev_lane_id; // Previous lane ID, useful to detect lane changes. This is the lane that the car was at the previous simulation timestep.
    Meters recent_forward_movement;
};
typedef struct Car Car;

typedef struct Simulation Simulation;


void car_init(Car* car, Dimensions dimensions,  CarCapabilities capabilities, Liters fuel_tank_capacity, CarPersonality preferences);


// getters:
CarId car_get_id(const Car* self);
Dimensions car_get_dimensions(const Car* self);
Meters car_get_length(const Car* self); 
Meters car_get_width(const Car* self);
Liters car_get_fuel_tank_capacity(const Car* self);
CarCapabilities car_get_capabilities(const Car* self);
CarPersonality car_get_preferences(const Car* self);

LaneId car_get_lane_id(const Car* self);
Lane* car_get_lane(const Car* self, Map* map);
double car_get_lane_progress(const Car* self);
// Returns car progress x lane length in meters.
Meters car_get_lane_progress_meters(const Car* self);
int car_get_lane_rank(const Car* self);
MetersPerSecond car_get_speed(const Car* self);
double car_get_damage(const Car* self);
Liters car_get_fuel_level(const Car* self);

MetersPerSecondSquared car_get_acceleration(const Car* self);
CarIndicator car_get_indicator_turn(const Car* self);
CarIndicator car_get_indicator_lane(const Car* self);
bool car_get_request_indicated_lane(const Car* self);
bool car_get_request_indicated_turn(const Car* self);
bool car_get_auto_turn_off_indicators(const Car* self);

// Returns the recent forward movement of the car in the current simulation timestep.
Meters car_get_recent_forward_movement(const Car* self);
// Returns the previous lane ID of the car.
LaneId car_get_prev_lane_id(const Car* self);

// setters:

// Sets the fuel tank capacity of the car.
void car_set_fuel_tank_capacity(Car* self, Liters capacity);
// Sets the lane of the car.
void car_set_lane(Car* self, const Lane* lane);
// Sets the lane progress of the car. The progress should be between 0.0 and 1.0. Will be clipped automatically.
void car_set_lane_progress(Car* self, double progress, Meters progress_meters, Meters recent_forward_movement);
void car_set_lane_rank(Car* self, int rank);
// Sets the speed of the car. The speed should be between 0.0 and max_speed_capability. Will be clipped automatically.
void car_set_speed(Car* self, MetersPerSecond speed);
// Sets the acceleration of the car. The acceleration will be clipped automatically as per the car's capable acceleration profile.
void car_set_acceleration(Car* self, MetersPerSecondSquared acceleration);
// Sets the damage level of the car. The damage should be between 0.0 and 1.0.
void car_set_damage(Car* self, const double damage);
// Sets the fuel level of the car.
void car_set_fuel_level(Car* self, Liters fuel_level);
// Sets the turn indicator signal of the car. This is purely an intent signal and does not request the sim to execute the turn.
void car_set_indicator_turn(Car* self, CarIndicator indicator);
// Sets the lane change indicator signal of the car. This is purely an intent signal and does not request the sim to execute the lane change.
void car_set_indicator_lane(Car* self, CarIndicator indicator);
// Sets the request to change lane as per the indicator_lane. This is set by the car's decision-making logic when car_make_decision() is called.
void car_set_request_indicated_lane(Car* self, bool request);
// Sets the request to turn as per the indicator_turn. This is set by the car's decision-making logic when car_make_decision() is called.
void car_set_request_indicated_turn(Car* self, bool request);
// Sets the turn indicator signal of the car and requests the sim to execute the turn.
void car_set_indicator_turn_and_request(Car* self, CarIndicator indicator);
// Sets the lane change indicator signal of the car and requests the sim to execute the lane change.
void car_set_indicator_lane_and_request(Car* self, CarIndicator indicator);
// Sets whether the sim should automatically turn off the car's indicators after the requested lane change or turn is completed.
void car_set_auto_turn_off_indicators(Car* self, bool auto_turn_off);
// Resets all control variables of the car to their default values: all indicators off, no requests, acceleration set to 0.
void car_reset_all_control_variables(Car* self);




// ------- Presets (values defined in car_presets_and_preferences.c) -----------

// acceleration profile capability of various types of vehicles:

extern const CarAccelProfile CAR_ACC_PROFILE_SPORTS_CAR;
extern const CarAccelProfile CAR_ACC_PROFILE_ELECTRIC_SEDAN;
extern const CarAccelProfile CAR_ACC_PROFILE_GAS_SEDAN;
extern const CarAccelProfile CAR_ACC_PROFILE_SUV;
extern const CarAccelProfile CAR_ACC_PROFILE_TRUCK;
extern const CarAccelProfile CAR_ACC_PROFILE_HEAVY_TRUCK;

CarAccelProfile car_accel_profile_get_random_preset(void);

// preferred acceleration profile presets:

extern const CarAccelProfile CAR_ACC_PREF_COMFORT;
extern const CarAccelProfile CAR_ACC_PREF_ECO;
extern const CarAccelProfile CAR_ACC_PREF_NORMAL;
extern const CarAccelProfile CAR_ACC_PREF_SPORT;
extern const CarAccelProfile CAR_ACC_PREF_AGGRESSIVE;

CarAccelProfile car_accel_profile_get_random_preference_preset(void);

// Dimensions of various types of cars:

extern const Dimensions CAR_SIZE_COMPACT;
extern const Dimensions CAR_SIZE_TYPICAL;
extern const Dimensions CAR_SIZE_MIDSIZE;
extern const Dimensions CAR_SIZE_COMPACT_SUV;
extern const Dimensions CAR_SIZE_LARGE_SUV;
extern const Dimensions CAR_SIZE_FULLSIZE;
extern const Dimensions CAR_SIZE_PICKUP;
extern const Dimensions CAR_SIZE_BUS;


Dimensions car_dimensions_get_random_preset(void);