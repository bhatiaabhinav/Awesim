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
    CarPersonality p = {
        .lane = rand_int_range(0, 2), // Random lane preference (0 to 2)
        .turn_speed = rand_uniform(from_mph(5), from_mph(15)),
        .scenic = rand_0_to_1(),
        .distance_to_front_vehicle = rand_uniform(from_feet(10), from_feet(100)),
        .distance_to_back_vehicle = rand_uniform(from_feet(10), from_feet(100)),
        .run_yellow_light = (bool)rand_int_range(0, 1),
        .turn_right_on_red = (bool)rand_int_range(0, 1),
        .aversion_to_rough_terrain = rand_0_to_1(),
        .aversion_to_traffic_jam = rand_0_to_1(),
        .aversion_to_construction_zone = rand_0_to_1(),
        .aversion_to_traffic_light = rand_0_to_1(),
        .aversion_to_lane_change = rand_0_to_1(),
        .average_speed_offset = rand_uniform(from_mph(-10), from_mph(0)),
        .min_speed_offset = rand_uniform(from_mph(-20), from_mph(-10)),
        .max_speed_offset = rand_uniform(from_mph(10), from_mph(20)),
        .acceleration_profile = car_accel_profile_get_random_preference_preset(),
    };
    return p;
}


CarAccelProfile car_accel_profile_get_random_preset(void) {
    const CarAccelProfile presets[] = { // TODO: re-enable slower cars. Currently, slower profiles make the cars brake too slow for short roads (which don't have enough braking distance available) and this forces the cars to jump traffic
        CAR_ACC_PROFILE_SPORTS_CAR,
        // CAR_ACC_PROFILE_ELECTRIC_SEDAN,
        // CAR_ACC_PROFILE_GAS_SEDAN,
        // CAR_ACC_PROFILE_SUV,
        // CAR_ACC_PROFILE_TRUCK,
        // CAR_ACC_PROFILE_HEAVY_TRUCK,
        
    };
    size_t count = sizeof(presets) / sizeof(presets[0]);
    return presets[rand() % count];
}

CarAccelProfile car_accel_profile_get_random_preference_preset(void) {
    const CarAccelProfile prefs[] = {   // TODO: re-enable slower profiles. Currently, slower profiles make the cars brake too slow for short roads (which don't have enough braking distance available) and this forces the cars to jump traffic lights.
        // CAR_ACC_PREF_COMFORT,
        // CAR_ACC_PREF_NORMAL,
        // CAR_ACC_PREF_ECO,
        // CAR_ACC_PREF_SPORT,
        CAR_ACC_PREF_AGGRESSIVE
    };
    size_t count = sizeof(prefs) / sizeof(prefs[0]);
    return prefs[rand() % count];
}

Dimensions car_dimensions_get_random_preset(void) {
    const Dimensions dims[] = { // TODO: once you find a way to make the roads longer, consider allowing BUS sized cars.
        CAR_SIZE_TYPICAL,
        CAR_SIZE_COMPACT,
        CAR_SIZE_MIDSIZE,
        CAR_SIZE_FULLSIZE,
        CAR_SIZE_COMPACT_SUV,
        CAR_SIZE_LARGE_SUV,
        CAR_SIZE_PICKUP,
        // CAR_SIZE_BUS
    };
    size_t count = sizeof(dims) / sizeof(dims[0]);
    return dims[rand() % count];
}


