#pragma once

#include "utils.h"
#include "road.h"


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

enum CarIndictor {
    INDICATOR_LEFT,         // Turn left or change to the left lane
    INDICATOR_NONE,         // Go straight or do not change lane
    INDICATOR_RIGHT,        // Turn right or change to the right lane
};
typedef enum CarIndictor CarIndictor;



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
};
typedef struct CarPersonality CarPersonality;

CarPersonality preferences_sample_random(void);



// -------- Car struct -----------

struct Car {
    // fixed properties:
    Dimensions dimensions;          // width and length of the car
    CarCapabilities capabilities;   // capabilities of the car
    CarPersonality preferences;     // Preference profile of the car.

    // state variables:
    const Lane* lane;                   // current lane
    double lane_progress;               // progress in the lane as a fraction of the lane length.
    MetersPerSecond speed;              // current speed. Should be set by the simulation engine.
    double damage;                      // damage level of the car. 0.0 for no damage and 1.0 for total damage.

    // control variables:
    MetersPerSecondSquared acceleration;    // current acceleration. Should be determined by the car's decision-making logic when car_make_decision() is called.
    CarIndictor indicator_lane;                  // Lane change indicator. Should be set by the car's decision-making logic when car_make_decision() is called.
    CarIndictor indicator_turn;                  // Turn indicator. Should be set by the car's decision-making logic when car_make_decision() is called.
};
typedef struct Car Car;


Car* car_create(Dimensions dimensions, CarCapabilities capabilities, CarPersonality preferences);
void car_free(Car* self);


// getters:

Dimensions car_get_dimensions(const Car* self);
Meters car_get_length(const Car* self); 
Meters car_get_width(const Car* self);
CarCapabilities car_get_capabilities(const Car* self);
CarPersonality car_get_preferences(const Car* self);

const Lane* car_get_lane(const Car* self);
double car_get_lane_progress(const Car* self);
// Returns car progress x lane length in meters.
Meters car_get_lane_progress_meters(const Car* self);
MetersPerSecond car_get_speed(const Car* self);
double car_get_damage(const Car* self);

MetersPerSecondSquared car_get_acceleration(const Car* self);
CarIndictor car_get_indicator_turn(const Car* self);
CarIndictor car_get_indicator_lane(const Car* self);


// setters:

// Sets the lane of the car.
void car_set_lane(Car* self, const Lane* lane);
// Sets the lane progress of the car. The progress should be between 0.0 and 1.0. Will be clipped automatically.
void car_set_lane_progress(Car* self, double progress);
// Sets the speed of the car. The speed should be between 0.0 and max_speed_capability. Will be clipped automatically.
void car_set_speed(Car* self, MetersPerSecond speed);
// Sets the acceleration of the car. The acceleration will be clipped automatically as per the car's capable acceleration profile.
void car_set_acceleration(Car* self, MetersPerSecondSquared acceleration);
// Sets the damage level of the car. The damage should be between 0.0 and 1.0.
void car_set_damage(Car* self, const double damage);
// Sets the turn indicator signal of the car
void car_set_indicator_turn(Car* self, CarIndictor indicator);
// Sets the lane change indicator signal of the car
void car_set_indicator_lane(Car* self, CarIndictor indicator);




// ------- Presets -----------

// acceleration profile capability of various types of vehicles:

static const CarAccelProfile CAR_ACC_PROFILE_SPORTS_CAR     = {9.0, 3.0, 12.0};  // Sports car (≈ 0–60 mph in ~3.0s)
static const CarAccelProfile CAR_ACC_PROFILE_ELECTRIC_SEDAN = {6.5, 5.0, 11.0};  // Electric sedan (≈ 0–60 mph in ~4.1s)
static const CarAccelProfile CAR_ACC_PROFILE_GAS_SEDAN      = {4.5, 2.0, 11.0};   // Gas sedan (≈ 0–60 mph in ~6.0s)
static const CarAccelProfile CAR_ACC_PROFILE_SUV            = {4.0, 2.0, 10.0};   // SUV/crossover (≈ 0–60 mph in ~6.7s)
static const CarAccelProfile CAR_ACC_PROFILE_TRUCK          = {3.0, 1.5, 9.0};   // Pickup truck (≈ 0–60 mph in ~8.9s)
static const CarAccelProfile CAR_ACC_PROFILE_HEAVY_TRUCK    = {1.8, 1.0, 8.0};   // Heavy-duty truck (≈ 0–60 mph in ~14.9s)


CarAccelProfile car_accel_profile_get_random_preset(void);


// preferred acceleration profile presets:

static const CarAccelProfile CAR_ACC_PREF_COMFORT    = {2.5, 1.0, 9.0};   // Comfort: soft accel & braking (≈ 0–60 mph in ~10.7s)
static const CarAccelProfile CAR_ACC_PREF_ECO        = {3.0, 1.5, 10.0};   // Eco: efficiency-focused (≈ 0–60 mph in ~8.9s)
static const CarAccelProfile CAR_ACC_PREF_NORMAL     = {4.0, 2.0, 11.0};   // Normal: balanced everyday driving (≈ 0–60 mph in ~6.7s)
static const CarAccelProfile CAR_ACC_PREF_SPORT      = {6.0, 3.0, 11.0};  // Sport: responsive feel (≈ 0–60 mph in ~4.5s)
static const CarAccelProfile CAR_ACC_PREF_AGGRESSIVE = {9.0, 4.0, 12.0};  // Aggressive: max driver intent (≈ 0–60 mph in ~3.0s)

CarAccelProfile car_accel_profile_get_random_preference_preset(void);


// Dimensions of various types of cars:

static const Dimensions CAR_SIZE_COMPACT       = {1.707, 4.267};   // Compact car (e.g., Honda Civic, 5.6 ft wide, 14 ft long)
static const Dimensions CAR_SIZE_TYPICAL       = {1.770, 4.480};   // Typical car (e.g., midsize sedan, 5.8 ft wide, 14.7 ft long)
static const Dimensions CAR_SIZE_MIDSIZE       = {1.829, 4.785};   // Midsize sedan (e.g., Toyota Camry, 6.0 ft wide, 15.7 ft long)
static const Dimensions CAR_SIZE_COMPACT_SUV   = {1.829, 4.877};   // Compact SUV (e.g., Honda CR-V, 6.0 ft wide, 16 ft long)
static const Dimensions CAR_SIZE_LARGE_SUV     = {1.981, 5.334};   // Large SUV (e.g., Chevrolet Traverse, 6.5 ft wide, 17.5 ft long)
static const Dimensions CAR_SIZE_FULLSIZE      = {1.829, 5.486};   // Full-size car (e.g., Dodge Charger, 6.0 ft wide, 18 ft long)
static const Dimensions CAR_SIZE_PICKUP        = {2.134, 5.608};   // Full-size pickup (e.g., Ford F-150, 7.0 ft wide, 18.4 ft long)
static const Dimensions CAR_SIZE_BUS           = {2.438, 12.000};  // Full-size bus (e.g., 40 ft transit bus, 8.0 ft wide, 40 ft long)

Dimensions car_dimensions_get_random_preset(void);