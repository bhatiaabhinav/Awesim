#include "car.h"

Dimensions car_dimensions_typical(void) { // Typical car (e.g., midsize sedan, 5.8 ft wide, 14.7 ft long)
    return (Dimensions){1.770, 4.480};
}

Dimensions car_dimensions_compact(void) { // Compact car (e.g., Honda Civic, 5.6 ft wide, 14 ft long)
    return (Dimensions){1.707, 4.267};
}

Dimensions car_dimensions_midsize(void) { // Midsize sedan (e.g., Toyota Camry, 6.0 ft wide, 15.7 ft long)
    return (Dimensions){1.829, 4.785};
}

Dimensions car_dimensions_fullsize(void) { // Full-size car (e.g., Dodge Charger, 6.0 ft wide, 18 ft long)
    return (Dimensions){1.829, 5.486};
}

Dimensions car_dimensions_compact_suv(void) { // Compact SUV (e.g., Honda CR-V, 6.0 ft wide, 16 ft long)
    return (Dimensions){1.829, 4.877};
}

Dimensions car_dimensions_large_suv(void) { // Large SUV (e.g., Chevrolet Traverse, 6.5 ft wide, 17.5 ft long)
    return (Dimensions){1.981, 5.334};
}

Dimensions car_dimensions_pickup(void) { // Full-size pickup (e.g., Ford F-150, 7.0 ft wide, 18.4 ft long)
    return (Dimensions){2.134, 5.608};
}


Dimensions car_dimensions_random_preset(void) {
    int random_choice = rand_int_range(0, 6);
    switch (random_choice) {
        case 0: return car_dimensions_typical();
        case 1: return car_dimensions_compact();
        case 2: return car_dimensions_midsize();
        case 3: return car_dimensions_fullsize();
        case 4: return car_dimensions_compact_suv();
        case 5: return car_dimensions_large_suv();
        case 6: return car_dimensions_pickup();
        default: return car_dimensions_typical(); // Fallback to typical
    };
}


CarCapabilities car_capable_max(void) { // A car with infinite capabilities
    return (CarCapabilities){1e9, 1e9, 1e9, 0.0};
}


PreferenceProfile preferences_random(void) {
    PreferenceProfile p = {
        .lane = rand_int_range(0, 2), // Random lane preference (0 to 2)
        .turn_speed = rand_uniform(from_mph(10), from_mph(30)),
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
        .max_acceleration = car_capable_max().max_acceleration,
        .max_deceleration = car_capable_max().max_deceleration,
        .average_speed_offset = rand_uniform(from_mph(-5), from_mph(5)),
        .min_speed_offset = rand_uniform(from_mph(-20), from_mph(-10)),
        .max_speed_offset = rand_uniform(from_mph(10), from_mph(20)),
        .acceleration_mode = (AccelerationMode)rand_int_range(0, 2), // Random acceleration mode (0 to 2)
    };
    return p;
}

PreferenceProfile preferences_random_with_average_speed_offset(const MetersPerSecond average_speed_offset) {
    PreferenceProfile p = preferences_random();
    p.average_speed_offset = average_speed_offset;
    return p;
}
