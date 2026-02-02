#include "car.h"
#include <stdlib.h>
#include <math.h>

CarAccelProfile car_accel_profile_get_effective(CarAccelProfile capable, CarAccelProfile preferred) {
    CarAccelProfile result;
    result.max_acceleration = fmin(capable.max_acceleration, preferred.max_acceleration);
    result.max_acceleration_reverse_gear = fmin(capable.max_acceleration_reverse_gear, preferred.max_acceleration_reverse_gear);
    result.max_deceleration = fmin(capable.max_deceleration, preferred.max_deceleration);
    return result;
}


CarPersonality preferences_sample_random(void) {
    CarPersonality p;
    p.lane = rand_int_range(0, 2);
    p.turn_speed = rand_uniform(from_mph(5), from_mph(15));
    p.scenic = rand_0_to_1();
    p.distance_to_front_vehicle = rand_uniform(from_feet(10), from_feet(100));
    p.distance_to_back_vehicle = rand_uniform(from_feet(10), from_feet(100));
    p.run_yellow_light = (bool)rand_int_range(0, 1);
    p.turn_right_on_red = (bool)rand_int_range(0, 1);
    p.aversion_to_rough_terrain = rand_0_to_1();
    p.aversion_to_traffic_jam = rand_0_to_1();
    p.aversion_to_construction_zone = rand_0_to_1();
    p.aversion_to_traffic_light = rand_0_to_1();
    p.aversion_to_lane_change = rand_0_to_1();
    p.average_speed_offset = rand_uniform(from_mph(-10), from_mph(0));
    p.min_speed_offset = rand_uniform(from_mph(-20), from_mph(-10));
    p.max_speed_offset = rand_uniform(from_mph(10), from_mph(20));
    p.acceleration_profile = car_accel_profile_get_random_preference_preset();
    p.time_headway = rand_uniform(2.0, 4.0);
    return p;
}


// acceleration profile capability of various types of vehicles:

const CarAccelProfile CAR_ACC_PROFILE_SPORTS_CAR     = {9.0, 3.0, 12.0};  // Sports car (≈ 0–60 mph in ~3.0s)
const CarAccelProfile CAR_ACC_PROFILE_ELECTRIC_SEDAN = {6.5, 5.0, 11.0};  // Electric sedan (≈ 0–60 mph in ~4.1s)
const CarAccelProfile CAR_ACC_PROFILE_GAS_SEDAN      = {4.5, 2.0, 11.0};   // Gas sedan (≈ 0–60 mph in ~6.0s)
const CarAccelProfile CAR_ACC_PROFILE_SUV            = {4.0, 2.0, 10.0};   // SUV/crossover (≈ 0–60 mph in ~6.7s)
const CarAccelProfile CAR_ACC_PROFILE_TRUCK          = {3.0, 1.5, 9.0};   // Pickup truck (≈ 0–60 mph in ~8.9s)
const CarAccelProfile CAR_ACC_PROFILE_HEAVY_TRUCK    = {1.8, 1.0, 8.0};   // Heavy-duty truck (≈ 0–60 mph in ~14.9s)

CarAccelProfile car_accel_profile_get_random_preset(void) {
    const CarAccelProfile presets[] = {
        CAR_ACC_PROFILE_SPORTS_CAR,
        CAR_ACC_PROFILE_ELECTRIC_SEDAN,
        CAR_ACC_PROFILE_GAS_SEDAN,
        CAR_ACC_PROFILE_SUV,
        CAR_ACC_PROFILE_TRUCK,
        CAR_ACC_PROFILE_HEAVY_TRUCK,
        
    };
    size_t count = sizeof(presets) / sizeof(presets[0]);
    return presets[rand_int(count)];
}


// preferred acceleration profile presets:

const CarAccelProfile CAR_ACC_PREF_COMFORT    = {2.5, 1.0, 9.0};   // Comfort: soft accel & braking (≈ 0–60 mph in ~10.7s)
const CarAccelProfile CAR_ACC_PREF_ECO        = {3.0, 1.5, 10.0};   // Eco: efficiency-focused (≈ 0–60 mph in ~8.9s)
const CarAccelProfile CAR_ACC_PREF_NORMAL     = {4.0, 2.0, 11.0};   // Normal: balanced everyday driving (≈ 0–60 mph in ~6.7s)
const CarAccelProfile CAR_ACC_PREF_SPORT      = {6.0, 3.0, 11.0};  // Sport: responsive feel (≈ 0–60 mph in ~4.5s)
const CarAccelProfile CAR_ACC_PREF_AGGRESSIVE = {9.0, 4.0, 12.0};  // Aggressive: max driver intent (≈ 0–60 mph in ~3.0s)

CarAccelProfile car_accel_profile_get_random_preference_preset(void) {
    const CarAccelProfile prefs[] = {
        CAR_ACC_PREF_COMFORT,
        CAR_ACC_PREF_NORMAL,
        CAR_ACC_PREF_ECO,
        CAR_ACC_PREF_SPORT,
        CAR_ACC_PREF_AGGRESSIVE
    };
    size_t count = sizeof(prefs) / sizeof(prefs[0]);
    return prefs[rand_int(count)];
}


// Dimensions of various types of cars:

const Dimensions3D CAR_SIZE_COMPACT       = {1.707, 4.267, 1.450};   // Compact car (e.g., Honda Civic, 5.6 ft wide, 14 ft long)
const Dimensions3D CAR_SIZE_TYPICAL       = {1.770, 4.480, 1.450};   // Typical car (e.g., midsize sedan, 5.8 ft wide, 14.7 ft long)
const Dimensions3D CAR_SIZE_MIDSIZE       = {1.829, 4.785, 1.450};   // Midsize sedan (e.g., Toyota Camry, 6.0 ft wide, 15.7 ft long)
const Dimensions3D CAR_SIZE_COMPACT_SUV   = {1.829, 4.877, 1.650};   // Compact SUV (e.g., Honda CR-V, 6.0 ft wide, 16 ft long)
const Dimensions3D CAR_SIZE_LARGE_SUV     = {1.981, 5.334, 1.800};   // Large SUV (e.g., Chevrolet Traverse, 6.5 ft wide, 17.5 ft long)
const Dimensions3D CAR_SIZE_FULLSIZE      = {1.829, 5.486, 1.500};   // Full-size car (e.g., Dodge Charger, 6.0 ft wide, 18 ft long)
const Dimensions3D CAR_SIZE_PICKUP        = {2.134, 5.608, 1.900};   // Full-size pickup (e.g., Ford F-150, 7.0 ft wide, 18.4 ft long)
const Dimensions3D CAR_SIZE_BUS           = {2.438, 9.000, 3.200};  // Mid-size bus (8.0 ft wide, 30 ft long)

Dimensions3D car_dimensions_get_random_preset(void) {
    const Dimensions3D dims[] = { // TODO: once you find a way to make the roads longer, consider allowing BUS sized cars.
        CAR_SIZE_TYPICAL,
        CAR_SIZE_COMPACT,
        CAR_SIZE_MIDSIZE,
        CAR_SIZE_FULLSIZE,
        CAR_SIZE_COMPACT_SUV,
        CAR_SIZE_LARGE_SUV,
        CAR_SIZE_PICKUP,
        CAR_SIZE_BUS,
    };
    size_t count = sizeof(dims) / sizeof(dims[0]);
    return dims[rand_int(count)];
}


void car_dimensions_and_acc_profile_get_random_realistic_preset(Dimensions3D *dims, CarAccelProfile *profile) {
    const struct {
        Dimensions3D dims;
        CarAccelProfile profile;
    } realistic_presets[] = {
        // Compacts & Economy
        {CAR_SIZE_COMPACT, CAR_ACC_PROFILE_GAS_SEDAN},      // Economy compact
        {CAR_SIZE_COMPACT, CAR_ACC_PROFILE_ELECTRIC_SEDAN}, // Electric compact
        {CAR_SIZE_COMPACT, CAR_ACC_PROFILE_SPORTS_CAR},     // Sports coupe / Hot hatch

        // Sedans (Typical to Fullsize)
        {CAR_SIZE_TYPICAL, CAR_ACC_PROFILE_GAS_SEDAN},
        {CAR_SIZE_TYPICAL, CAR_ACC_PROFILE_ELECTRIC_SEDAN},
        {CAR_SIZE_MIDSIZE, CAR_ACC_PROFILE_GAS_SEDAN},
        {CAR_SIZE_MIDSIZE, CAR_ACC_PROFILE_ELECTRIC_SEDAN},
        {CAR_SIZE_FULLSIZE, CAR_ACC_PROFILE_GAS_SEDAN},
        {CAR_SIZE_FULLSIZE, CAR_ACC_PROFILE_SPORTS_CAR},    // Muscle car / Performance sedan

        // SUVs & Crossovers
        {CAR_SIZE_COMPACT_SUV, CAR_ACC_PROFILE_SUV},
        {CAR_SIZE_COMPACT_SUV, CAR_ACC_PROFILE_ELECTRIC_SEDAN}, // Electric Crossover
        {CAR_SIZE_LARGE_SUV, CAR_ACC_PROFILE_SUV},
        {CAR_SIZE_LARGE_SUV, CAR_ACC_PROFILE_TRUCK},        // Large Truck-based SUV

        // Trucks
        {CAR_SIZE_PICKUP, CAR_ACC_PROFILE_TRUCK},
        {CAR_SIZE_PICKUP, CAR_ACC_PROFILE_HEAVY_TRUCK},     // Heavy duty pickup

        // Buses
        {CAR_SIZE_BUS, CAR_ACC_PROFILE_HEAVY_TRUCK},
    };
    size_t count = sizeof(realistic_presets) / sizeof(realistic_presets[0]);
    int idx = rand_int(count);
    *dims = realistic_presets[idx].dims;
    *profile = realistic_presets[idx].profile;
}