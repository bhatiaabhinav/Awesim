#include "awesim.h"
#include "map.h"
#include "utils.h"
#include "logging.h"
#include <stdio.h>
#include <math.h>

static bool placement_would_cause_collision(Car* car, Lane* lane, Simulation* sim, Meters progress) {
    Meters progress_meters = progress * lane->length;
    Meters car_length = car_get_length(car);
    bool would_collide = false;
    for (int i = 0; i < lane->num_cars; i++) {
        Car* other_car = lane_get_car(lane, sim, i);
        Meters other_progress_meters = car_get_lane_progress_meters(other_car);
        Meters distance_between_centers = fabs(progress_meters - other_progress_meters);
        Meters other_length = car_get_length(other_car);
        if (distance_between_centers < meters(1) + (car_length + other_length) / 2.0) {
            LOG_TRACE("Placement of car id %d on lane %d at %.2f meters would cause collision with car %d at progress %.2f meters.", car->id, lane->id, progress_meters, other_car->id, other_progress_meters);
            would_collide = true;
            break;
        }
    }
    return would_collide;
}

void awesim_setup(Simulation* sim, Meters city_width, int num_cars, Seconds dt, ClockReading initial_clock_reading, Weather weather) {
    sim_init(sim);
    sim_set_dt(sim, dt);
    sim_set_initial_clock_reading(sim, initial_clock_reading);
    sim_set_weather(sim, weather);
    Map* map = sim_get_map(sim);
    LOG_TRACE("Map allocated at %p, with memory size %llu bytes", (void*)map, sizeof(*map));
    awesim_map_setup(map, city_width);
    int num_intersection_lanes = 0;
    for (int i = 0; i < map->num_lanes; i++) {
        Lane* lane = map_get_lane(map, i);
        if (lane->is_at_intersection) {
            num_intersection_lanes++;
        }
    }
    int num_road_lanes = map->num_lanes - num_intersection_lanes;
    LOG_DEBUG("Map setup complete with %d roads, %d intersections, %d lanes (%d on roads, %d on intersections)", map->num_roads, map->num_intersections, map->num_lanes, num_road_lanes, num_intersection_lanes);
    if (num_cars > MAX_CARS_IN_SIMULATION) {
        LOG_ERROR("Cannot setup simulation with %d cars. Maximum allowed is %d.", num_cars, MAX_CARS_IN_SIMULATION);
        num_cars = MAX_CARS_IN_SIMULATION;
    }
    for(int i = 0; i < num_cars; i++) {
        Car* car = sim_get_new_car(sim);
        car_init(car, car_dimensions_get_random_preset(), (CarCapabilities){CAR_ACC_PROFILE_SPORTS_CAR, from_mph(120), 0.0}, preferences_sample_random());
        Road* random_road;
        Lane* random_lane;
        double random_progress;
        int attempts = 0;
        int max_attempts = 1000;
        bool would_collide = true;
        while (would_collide && attempts < max_attempts) {
            random_road = map_get_road(map, rand_int_range(0, map->num_roads - 1));
            random_lane = road_get_lane(random_road, map, rand_int_range(0, road_get_num_lanes(random_road) - 1));
            random_progress = rand_uniform(0.1, 0.9);
            attempts++;
            would_collide = placement_would_cause_collision(car, random_lane, sim, random_progress);
            if (would_collide) LOG_TRACE("Car id %d placement attempt %d on lane %d at position %.2f meters failed due to a potential collision. Retrying with a new lane.", car->id, attempts, random_lane->id, random_progress * random_lane->length);
        }
        if (would_collide) {
            LOG_ERROR("Failed to place car id %d after %d attempts. There may be too many cars and not enough space on the road. Not adding any more cars", car->id, attempts);
            sim->num_cars = i; // Set the number of cars to the successfully placed ones
            break;
        } else {
            lane_add_car(random_lane, car->id);
            car_set_lane(car, random_lane);
            car_set_lane_progress(car, random_progress, random_progress * random_lane->length);
            // car_set_speed(car, random_lane->speed_limit + car->preferences.average_speed_offset);
            car_set_speed(car, 0);
            LOG_TRACE("Placed car id %d on lane %d (%s) at position %.2f meters (speed = %.2f mph) after %d attempts", car->id, random_lane->id, road_get_name(random_road), car_get_lane_progress_meters(car), to_mph(car_get_speed(car)), attempts);
        }
    }
    LOG_INFO("Awesome simulator setup complete with city width: %.2f meters, %d cars, Clock: %02d:%02d:%02d, Weather: %s, dt: %.2f seconds",
             city_width, sim->num_cars,
             sim->initial_clock_reading.hours,
             sim->initial_clock_reading.minutes,
             (int)sim->initial_clock_reading.seconds,
             weather_strings[sim->weather],
             sim->dt);
}


void awesim_map_setup(Map* map, Meters city_width) {
    map_init(map);

    // Define constants
    const int interstate_num_lanes = 3;
    const int state_num_lanes = 2;
    const Meters lane_width = from_feet(12);
    const Meters city_height = city_width;
    const Meters city_width_half = city_width / 2;
    const Meters city_height_half = city_height / 2;
    const MetersPerSecond interstate_speed_limit = from_mph(55);
    const MetersPerSecond interstate_turn_speed_limit = from_mph(50);
    const MetersPerSecond state_speed_limit = from_mph(45);
    const MetersPerSecond state_turn_speed_limit = from_mph(40);
    const MetersPerSecond exit_ramp_speed_limit = from_mph(35);
    const MetersPerSecond local_speed_limit = from_mph(35);
    const MetersPerSecond local_turn_speed_limit = from_mph(25);
    const Meters state_turn_radius = 5;
    const Meters local_turn_radius = 3;
    const Meters ramp_length = meters(25);
    const bool local_four_way_stop = false;
    LOG_DEBUG("Creating map with parameters: city_width=%.2f, interstate_num_lanes=%d, state_num_lanes=%d, lane_width=%.2f, interstate_speed_limit=%.2f, state_speed_limit=%.2f, interstate_turn_speed_limit=%.2f, state_turn_speed_limit=%.2f, exit_ramp_speed_limit=%.2f, local_speed_limit=%.2f, local_turn_speed_limit=%.2f, state_turn_radius=%.2f, local_turn_radius=%.2f",
              city_width, interstate_num_lanes, state_num_lanes, lane_width,
              interstate_speed_limit, state_speed_limit,
              interstate_turn_speed_limit, state_turn_speed_limit,
              exit_ramp_speed_limit, local_speed_limit,
              local_turn_speed_limit, state_turn_radius, local_turn_radius);

    // Interstate roads
    const Coordinates I2E_center = coordinates_create(0, city_height_half);
    const Meters I2E_length = city_width * 0.93;
    Road* I2E = straight_road_create_from_center_dir_len(map, I2E_center, DIRECTION_EAST, I2E_length, interstate_num_lanes, lane_width, interstate_speed_limit, 1);
    LOG_TRACE("Created interstate road I-2 E at center: (%.2f, %.2f) with length: %.2f", I2E_center.x, I2E_center.y, I2E_length);

    const Coordinates I2W_center = coordinates_create(0, -city_height_half);
    const Meters I2W_length = city_width * 0.93;
    Road* I2W = straight_road_create_from_center_dir_len(map, I2W_center, DIRECTION_WEST, I2W_length, interstate_num_lanes, lane_width, interstate_speed_limit, 1);
    LOG_TRACE("Created interstate road I-2 W at center: (%.2f, %.2f) with length: %.2f", I2W_center.x, I2W_center.y, I2W_length);

    const Coordinates I1N_center = coordinates_create(-city_width_half, 0);
    const Meters I1N_length = city_height * 0.93;
    Road* I1N = straight_road_create_from_center_dir_len(map, I1N_center, DIRECTION_NORTH, I1N_length, interstate_num_lanes, lane_width, interstate_speed_limit, 1);
    LOG_TRACE("Created interstate road I-1 N at center: (%.2f, %.2f) with length: %.2f", I1N_center.x, I1N_center.y, I1N_length);

    const Coordinates I1S_center = coordinates_create(city_width_half, 0);
    const Meters I1S_length = city_height * 0.93;
    Road* I1S = straight_road_create_from_center_dir_len(map, I1S_center, DIRECTION_SOUTH, I1S_length, interstate_num_lanes, lane_width, interstate_speed_limit, 1);
    LOG_TRACE("Created interstate road I-1 S at center: (%.2f, %.2f) with length: %.2f", I1S_center.x, I1S_center.y, I1S_length);

    road_set_name(I2E, "I-2 E");
    road_set_name(I2W, "I-2 W");
    road_set_name(I1N, "I-1 N");
    road_set_name(I1S, "I-1 S");
    LOG_DEBUG("Interstate roads created: I-2 E, I-2 W, I-1 N, I-1 S");

    // Interstate connectors
    Road* interstate_connectors[4];
    interstate_connectors[0] = turn_create_and_set_connections_and_adjacents(map, I2E, I1S, DIRECTION_CW, interstate_turn_speed_limit, 1.0, NULL); // NE
    interstate_connectors[1] = turn_create_and_set_connections_and_adjacents(map, I1S, I2W, DIRECTION_CW, interstate_turn_speed_limit, 1.0, NULL); // SE
    interstate_connectors[2] = turn_create_and_set_connections_and_adjacents(map, I2W, I1N, DIRECTION_CW, interstate_turn_speed_limit, 1.0, NULL); // SW
    interstate_connectors[3] = turn_create_and_set_connections_and_adjacents(map, I1N, I2E, DIRECTION_CW, interstate_turn_speed_limit, 1.0, NULL); // NW
    LOG_TRACE("Created interstate connectors: NE, SE, SW, NW");

    road_set_name(interstate_connectors[0], "I-2 E -> I-1 S");
    road_set_name(interstate_connectors[1], "I-1 S -> I-2 W");
    road_set_name(interstate_connectors[2], "I-2 W -> I-1 N");
    road_set_name(interstate_connectors[3], "I-1 N -> I-2 E");
    LOG_DEBUG("Interstate connectors named: I-2 E -> I-1 S, I-1 S -> I-2 W, I-2 W -> I-1 N, I-1 N -> I-2 E");

    // State highways
    const Meters state_hgway_width = state_num_lanes * lane_width;
    const Meters state_edge_offset = I2E->width / 2 + 2 * state_hgway_width;
    double S1_x_offset = 0.5 * lane_width * state_num_lanes;
    double S2_y_offset = S1_x_offset;

    const Coordinates S1N_start = coordinates_create(S1_x_offset, -city_height_half + state_edge_offset);
    const Coordinates S1N_end = coordinates_create(S1_x_offset, city_height_half - state_edge_offset);
    Road* _S1N = straight_road_create_from_start_end(map, S1N_start, S1N_end, state_num_lanes, lane_width, state_speed_limit, 1.0);
    LOG_TRACE("Created state road S-1 N from (%.2f, %.2f) to (%.2f, %.2f)", S1N_start.x, S1N_start.y, S1N_end.x, S1N_end.y);
    Road* _S1S = straight_road_make_opposite_and_update_adjacents(map, _S1N);
    LOG_TRACE("Created state road S-1 S from (%.2f, %.2f) to (%.2f, %.2f)", S1N_end.x, S1N_end.y, S1N_start.x, S1N_start.y);

    const Coordinates S2E_start = coordinates_create(-city_width_half + state_edge_offset, -S2_y_offset);
    const Coordinates S2E_end = coordinates_create(city_width_half - state_edge_offset, -S2_y_offset);
    Road* _S2E = straight_road_create_from_start_end(map, S2E_start, S2E_end, state_num_lanes, lane_width, state_speed_limit, 1.0);
    LOG_TRACE("Created state road S-2 E from (%.2f, %.2f) to (%.2f, %.2f)", S2E_start.x, S2E_start.y, S2E_end.x, S2E_end.y);
    Road* _S2W = straight_road_make_opposite_and_update_adjacents(map, _S2E);
    LOG_TRACE("Created state road S-2 W from (%.2f, %.2f) to (%.2f, %.2f)", S2E_end.x, S2E_end.y, S2E_start.x, S2E_start.y);

    road_set_name(_S1N, "S-1 N");
    road_set_name(_S1S, "S-1 S");
    road_set_name(_S2E, "S-2 E");
    road_set_name(_S2W, "S-2 W");
    LOG_DEBUG("State roads created: S-1 N, S-1 S, S-2 E, S-2 W");

    // Central intersection
    Intersection* intersection = intersection_create_from_crossing_roads_and_update_connections(map, _S2E, _S2W, _S1N, _S1S, false, state_turn_radius, false, false, state_speed_limit, 1.0);
    intersection_set_name(intersection, "S-2 & S-1 Intersection");
    LOG_DEBUG("Created central intersection S-2 & S-1 at center: (%.2f, %.2f), with %d lanes", intersection->center.x, intersection->center.y, intersection->num_lanes);

    Road* _S2E_left = _S2E;
    Road* _S2E_right = intersection_get_road_eastbound_to(intersection, map);
    Road* _S2W_right = _S2W;
    Road* _S2W_left = intersection_get_road_westbound_to(intersection, map);
    Road* _S1N_below = _S1N;
    Road* _S1N_above = intersection_get_road_northbound_to(intersection, map);
    Road* _S1S_above = _S1S;
    Road* _S1S_below = intersection_get_road_southbound_to(intersection, map);

    road_set_name(_S2E_right, "S-2 E");
    road_set_name(_S2W_left, "S-2 W");
    road_set_name(_S1N_above, "S-1 N");
    road_set_name(_S1S_below, "S-1 S");

    // 1st Avenue
    const Coordinates _Av1N_center = vec_midpoint(_S2E_left->center, _S2W_left->center);
    const Meters av1_length = city_height - 4 * I2E->width;
    Road* _Av1N = straight_road_create_from_center_dir_len(map, _Av1N_center, DIRECTION_NORTH, av1_length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created 1st Av N at center: (%.2f, %.2f) with length: %.2f", _Av1N_center.x, _Av1N_center.y, av1_length);
    Road* _Av1S = straight_road_make_opposite_and_update_adjacents(map, _Av1N);
    LOG_TRACE("Created 1st Av S at center: (%.2f, %.2f) with length: %.2f", _Av1S->center.x, _Av1S->center.y, av1_length);

    road_set_name(_Av1N, "1st Av N");
    road_set_name(_Av1S, "1st Av S");
    LOG_DEBUG("1st Avenue roads created: 1st Av N, 1st Av S");

    Intersection* Av1S2_intersection = intersection_create_from_crossing_roads_and_update_connections(map, _S2E_left, _S2W_left, _Av1N, _Av1S, false, local_turn_radius, false, false, local_speed_limit, 1.0);
    intersection_set_name(Av1S2_intersection, "S-2 & 1st Av Intersection");
    LOG_DEBUG("Created intersection S-2 & 1st Av at center: (%.2f, %.2f)", Av1S2_intersection->center.x, Av1S2_intersection->center.y);

    Road* _S2E_left_left = _S2E_left;
    Road* _S2E_left_new = intersection_get_road_eastbound_to(Av1S2_intersection, map);
    Road* _S2W_left_left = intersection_get_road_westbound_to(Av1S2_intersection, map);
    Road* _Av1N_below = _Av1N;
    Road* _Av1N_above = intersection_get_road_northbound_to(Av1S2_intersection, map);
    Road* _Av1S_above = _Av1S;
    Road* _Av1S_below = intersection_get_road_southbound_to(Av1S2_intersection, map);

    road_set_name(_S2E_left_new, "S-2 E");
    road_set_name(_S2W_left_left, "S-2 W");
    road_set_name(_Av1N_above, "1st Av N");
    road_set_name(_Av1S_below, "1st Av S");

    // 2nd Avenue
    const Coordinates _Av2N_center = vec_midpoint(_S2E_right->center, _S2W_right->center);
    Road* _Av2N = straight_road_create_from_center_dir_len(map, _Av2N_center, DIRECTION_NORTH, av1_length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created 2nd Av N at center: (%.2f, %.2f) with length: %.2f", _Av2N_center.x, _Av2N_center.y, av1_length);
    Road* _Av2S = straight_road_make_opposite_and_update_adjacents(map, _Av2N);
    LOG_TRACE("Created 2nd Av S at center: (%.2f, %.2f) with length: %.2f", _Av2S->center.x, _Av2S->center.y, av1_length);

    road_set_name(_Av2N, "2nd Av N");
    road_set_name(_Av2S, "2nd Av S");
    LOG_DEBUG("2nd Avenue roads created: 2nd Av N, 2nd Av S");

    Intersection* Av2S2_intersection = intersection_create_from_crossing_roads_and_update_connections(map, _S2E_right, _S2W_right, _Av2N, _Av2S, false, local_turn_radius, false, false, local_speed_limit, 1.0);
    intersection_set_name(Av2S2_intersection, "S-2 & 2nd Av Intersection");
    LOG_DEBUG("Created intersection S-2 & 2nd Av at center: (%.2f, %.2f)", Av2S2_intersection->center.x, Av2S2_intersection->center.y);

    Road* _S2E_right_right = intersection_get_road_eastbound_to(Av2S2_intersection, map);
    Road* _S2W_right_right = _S2W_right;
    Road* _S2W_right_new = intersection_get_road_westbound_to(Av2S2_intersection, map);
    Road* _Av2N_below = _Av2N;
    Road* _Av2N_above = intersection_get_road_northbound_to(Av2S2_intersection, map);
    Road* _Av2S_above = _Av2S;
    Road* _Av2S_below = intersection_get_road_southbound_to(Av2S2_intersection, map);

    road_set_name(_S2E_right_right, "S-2 E");
    road_set_name(_S2W_right_new, "S-2 W");
    road_set_name(_Av2N_above, "2nd Av N");
    road_set_name(_Av2S_below, "2nd Av S");

    // 1st Street
    const Coordinates _St1E_center = vec_midpoint(_S1N_above->center, _S1S_above->center);
    const Meters st1_length = city_width - 4 * I2E->width;
    Road* _St1E = straight_road_create_from_center_dir_len(map, _St1E_center, DIRECTION_EAST, st1_length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created 1st St E at center: (%.2f, %.2f) with length: %.2f", _St1E_center.x, _St1E_center.y, st1_length);
    Road* _St1W = straight_road_make_opposite_and_update_adjacents(map, _St1E);
    LOG_TRACE("Created 1st St W at center: (%.2f, %.2f) with length: %.2f", _St1W->center.x, _St1W->center.y, st1_length);

    road_set_name(_St1E, "1st St E");
    road_set_name(_St1W, "1st St W");
    LOG_DEBUG("1st Street roads created: 1st St E, 1st St W");

    Intersection* St1S1_intersection = intersection_create_from_crossing_roads_and_update_connections(map, _St1E, _St1W, _S1N_above, _S1S_above, false, local_turn_radius, false, false, local_speed_limit, 1.0);
    intersection_set_name(St1S1_intersection, "1st St & S-1 Intersection");
    LOG_DEBUG("Created intersection 1st St & S-1 at center: (%.2f, %.2f)", St1S1_intersection->center.x, St1S1_intersection->center.y);

    Road* _St1E_left = _St1E;
    Road* _St1E_right = intersection_get_road_eastbound_to(St1S1_intersection, map);
    Road* _St1W_right = _St1W;
    Road* _St1W_left = intersection_get_road_westbound_to(St1S1_intersection, map);
    Road* _S1N_above_above = intersection_get_road_northbound_to(St1S1_intersection, map);
    Road* _S1S_above_above = _S1S_above;
    Road* _S1S_above_new = intersection_get_road_southbound_to(St1S1_intersection, map);

    road_set_name(_St1E_right, "1st St E");
    road_set_name(_St1W_left, "1st St W");
    road_set_name(_S1N_above_above, "S-1 N");
    road_set_name(_S1S_above_new, "S-1 S");

    // 2nd Street
    const Coordinates _St2E_center = vec_midpoint(_S1N_below->center, _S1S_below->center);
    Road* _St2E = straight_road_create_from_center_dir_len(map, _St2E_center, DIRECTION_EAST, st1_length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created 2nd St E at center: (%.2f, %.2f) with length: %.2f", _St2E_center.x, _St2E_center.y, st1_length);
    Road* _St2W = straight_road_make_opposite_and_update_adjacents(map, _St2E);
    LOG_TRACE("Created 2nd St W at center: (%.2f, %.2f) with length: %.2f", _St2W->center.x, _St2W->center.y, st1_length);

    road_set_name(_St2E, "2nd St E");
    road_set_name(_St2W, "2nd St W");
    LOG_DEBUG("2nd Street roads created: 2nd St E, 2nd St W");

    Intersection* St2S1_intersection = intersection_create_from_crossing_roads_and_update_connections(map, _St2E, _St2W, _S1N_below, _S1S_below, false, local_turn_radius, false, false, local_speed_limit, 1.0);
    intersection_set_name(St2S1_intersection, "2nd St & S-1 Intersection");
    LOG_DEBUG("Created intersection 2nd St & S-1 at center: (%.2f, %.2f)", St2S1_intersection->center.x, St2S1_intersection->center.y);

    Road* _St2E_left = _St2E;
    Road* _St2E_right = intersection_get_road_eastbound_to(St2S1_intersection, map);
    Road* _St2W_right = _St2W;
    Road* _St2W_left = intersection_get_road_westbound_to(St2S1_intersection, map);
    Road* _S1N_below_below = _S1N_below;
    Road* _S1N_below_new = intersection_get_road_northbound_to(St2S1_intersection, map);
    Road* _S1S_below_below = intersection_get_road_southbound_to(St2S1_intersection, map);

    road_set_name(_St2E_right, "2nd St E");
    road_set_name(_St2W_left, "2nd St W");
    road_set_name(_S1N_below_new, "S-1 N");
    road_set_name(_S1S_below_below, "S-1 S");

    // 1st St and 1st Av intersection
    Intersection* St1Av1_intersection = intersection_create_from_crossing_roads_and_update_connections(map, _St1E_left, _St1W_left, _Av1N_above, _Av1S_above, local_four_way_stop, local_turn_radius, false, false, local_speed_limit, 1.0);
    intersection_set_name(St1Av1_intersection, "1st St & 1st Av Intersection");
    LOG_DEBUG("Created intersection 1st St & 1st Av at center: (%.2f, %.2f)", St1Av1_intersection->center.x, St1Av1_intersection->center.y);

    Road* _St1E_left_left = _St1E_left;
    Road* _St1E_left_new = intersection_get_road_eastbound_to(St1Av1_intersection, map);
    Road* _St1W_left_left = intersection_get_road_westbound_to(St1Av1_intersection, map);
    Road* _Av1N_above_above = intersection_get_road_northbound_to(St1Av1_intersection, map);
    Road* _Av1S_above_above = _Av1S_above;
    Road* _Av1S_above_new = intersection_get_road_southbound_to(St1Av1_intersection, map);

    road_set_name(_St1E_left_new, "1st St E");
    road_set_name(_St1W_left_left, "1st St W");
    road_set_name(_Av1N_above_above, "1st Av N");
    road_set_name(_Av1S_above_new, "1st Av S");

    // 1st St and 2nd Av intersection
    Intersection* St1Av2_intersection = intersection_create_from_crossing_roads_and_update_connections(map, _St1E_right, _St1W_right, _Av2N_above, _Av2S_above, local_four_way_stop, local_turn_radius, false, false, local_speed_limit, 1.0);
    intersection_set_name(St1Av2_intersection, "1st St & 2nd Av Intersection");
    LOG_DEBUG("Created intersection 1st St & 2nd Av at center: (%.2f, %.2f)", St1Av2_intersection->center.x, St1Av2_intersection->center.y);

    Road* _St1E_right_right = intersection_get_road_eastbound_to(St1Av2_intersection, map);
    Road* _St1W_right_right = _St1W_right;
    Road* _St1W_right_new = intersection_get_road_westbound_to(St1Av2_intersection, map);
    Road* _Av2N_above_above = intersection_get_road_northbound_to(St1Av2_intersection, map);
    Road* _Av2S_above_above = _Av2S_above;
    Road* _Av2S_above_new = intersection_get_road_southbound_to(St1Av2_intersection, map);

    road_set_name(_St1E_right_right, "1st St E");
    road_set_name(_St1W_right_new, "1st St W");
    road_set_name(_Av2N_above_above, "2nd Av N");
    road_set_name(_Av2S_above_new, "2nd Av S");

    // 2nd St and 1st Av intersection
    Intersection* St2Av1_intersection = intersection_create_from_crossing_roads_and_update_connections(map, _St2E_left, _St2W_left, _Av1N_below, _Av1S_below, local_four_way_stop, local_turn_radius, false, false, local_speed_limit, 1.0);
    intersection_set_name(St2Av1_intersection, "2nd St & 1st Av Intersection");
    LOG_DEBUG("Created intersection 2nd St & 1st Av at center: (%.2f, %.2f)", St2Av1_intersection->center.x, St2Av1_intersection->center.y);

    Road* _St2E_left_left = _St2E_left;
    Road* _St2E_left_new = intersection_get_road_eastbound_to(St2Av1_intersection, map);
    Road* _St2W_left_left = intersection_get_road_westbound_to(St2Av1_intersection, map);
    Road* _Av1N_below_below = _Av1N_below;
    Road* _Av1N_below_new = intersection_get_road_northbound_to(St2Av1_intersection, map);
    Road* _Av1S_below_below = intersection_get_road_southbound_to(St2Av1_intersection, map);

    road_set_name(_St2E_left_new, "2nd St E");
    road_set_name(_St2W_left_left, "2nd St W");
    road_set_name(_Av1N_below_new, "1st Av N");
    road_set_name(_Av1S_below_below, "1st Av S");

    // 2nd St and 2nd Av intersection
    Intersection* St2Av2_intersection = intersection_create_from_crossing_roads_and_update_connections(map, _St2E_right, _St2W_right, _Av2N_below, _Av2S_below, local_four_way_stop, local_turn_radius, false, false, local_speed_limit, 1.0);
    intersection_set_name(St2Av2_intersection, "2nd St & 2nd Av Intersection");
    LOG_DEBUG("Created intersection 2nd St & 2nd Av at center: (%.2f, %.2f)", St2Av2_intersection->center.x, St2Av2_intersection->center.y);

    Road* _St2E_right_right = intersection_get_road_eastbound_to(St2Av2_intersection, map);
    Road* _St2W_right_right = _St2W_right;
    Road* _St2W_right_new = intersection_get_road_westbound_to(St2Av2_intersection, map);
    Road* _Av2N_below_below = _Av2N_below;
    Road* _Av2N_below_new = intersection_get_road_northbound_to(St2Av2_intersection, map);
    Road* _Av2S_below_below = intersection_get_road_southbound_to(St2Av2_intersection, map);

    road_set_name(_St2E_right_right, "2nd St E");
    road_set_name(_St2W_right_new, "2nd St W");
    road_set_name(_Av2N_below_new, "2nd Av N");
    road_set_name(_Av2S_below_below, "2nd Av S");

    // Top Right Extension (TRE, TRW, RTN, RTS)
    const Coordinates TRE_center = {_St1E_right_right->center.x, _Av2N_above_above->end_point.y + local_turn_radius + 0.5 * lane_width};
    Road* TRE = straight_road_create_from_center_dir_len(map, TRE_center, DIRECTION_EAST, _St1E_right_right->length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created top right extension road TRE at center: (%.2f, %.2f) with length: %.2f", TRE_center.x, TRE_center.y, _St1E_right_right->length);
    Road* TRW = straight_road_make_opposite_and_update_adjacents(map, TRE);
    LOG_TRACE("Created top right extension road TRW at center: (%.2f, %.2f) with length: %.2f", TRW->center.x, TRW->center.y, _St1E_right_right->length);
    const Coordinates RTN_center = {_St1E_right_right->end_point.x + local_turn_radius + 1.5 * lane_width, _Av2N_above_above->center.y};
    Road* RTN = straight_road_create_from_center_dir_len(map, RTN_center, DIRECTION_NORTH, _Av2N_above_above->length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created top right extension road RTN at center: (%.2f, %.2f) with length: %.2f", RTN_center.x, RTN_center.y, _Av2N_above_above->length);
    Road* RTS = straight_road_make_opposite_and_update_adjacents(map, RTN);
    LOG_TRACE("Created top right extension road RTS at center: (%.2f, %.2f) with length: %.2f", RTS->center.x, RTS->center.y, _Av2N_above_above->length);

    Road* _Av2N_above_above_TRE_connector = turn_create_and_set_connections_and_adjacents(map, _Av2N_above_above, TRE, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from 2nd Av N to TRE at center: (%.2f, %.2f)", _Av2N_above_above_TRE_connector->center.x, _Av2N_above_above_TRE_connector->center.y);
    Road* TRW_Av2S_above_above_connector = turn_create_and_set_connections_and_adjacents(map, TRW, _Av2S_above_above, DIRECTION_CCW, local_turn_speed_limit, 1.0, _Av2N_above_above_TRE_connector);
    LOG_TRACE("Created connector from TRW to 2nd Av S at center: (%.2f, %.2f)", TRW_Av2S_above_above_connector->center.x, TRW_Av2S_above_above_connector->center.y);
    Road* RTN_TRW_connector = turn_create_and_set_connections_and_adjacents(map, RTN, TRW, DIRECTION_CCW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from RTN to TRW at center: (%.2f, %.2f)", RTN_TRW_connector->center.x, RTN_TRW_connector->center.y);
    Road* TRE_RTS_connector = turn_create_and_set_connections_and_adjacents(map, TRE, RTS, DIRECTION_CW, local_turn_speed_limit, 1.0, RTN_TRW_connector);
    LOG_TRACE("Created connector from TRE to RTS at center: (%.2f, %.2f)", TRE_RTS_connector->center.x, TRE_RTS_connector->center.y);
    Road* St1ERR_RTN_connector = turn_create_and_set_connections_and_adjacents(map, _St1E_right_right, RTN, DIRECTION_CCW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from St1E to RTN at center: (%.2f, %.2f)", St1ERR_RTN_connector->center.x, St1ERR_RTN_connector->center.y);
    Road* RTS_St1WRR_connector = turn_create_and_set_connections_and_adjacents(map, RTS, _St1W_right_right, DIRECTION_CW, local_turn_speed_limit, 1.0, St1ERR_RTN_connector);
    LOG_TRACE("Created connector from RTS to St1W at center: (%.2f, %.2f)", RTS_St1WRR_connector->center.x, RTS_St1WRR_connector->center.y);

    road_set_name(TRE, "Euler St E");
    road_set_name(TRW, "Euler St W");
    road_set_name(RTN, "Gauss Av N");
    road_set_name(RTS, "Gauss Av S");
    road_set_name(_Av2N_above_above_TRE_connector, "2nd Av N -> Euler St E");
    road_set_name(TRW_Av2S_above_above_connector, "Euler St W -> 2nd Av S");
    road_set_name(RTN_TRW_connector, "Gauss Av N -> Euler St W");
    road_set_name(TRE_RTS_connector, "Euler St E -> Gauss Av S");
    road_set_name(RTS_St1WRR_connector, "Gauss Av S -> 1st St W");
    road_set_name(St1ERR_RTN_connector, "1st St E -> Gauss Av N");
    LOG_DEBUG("Top right extension roads created: Euler St E, Euler St W, Gauss Av N, Gauss Av S");

    // Bottom Right Extension (BRE, BRW, RBN, RBS)
    const Coordinates BRE_center = {_St2E_right_right->center.x, _Av2N_below_below->start_point.y - local_turn_radius - 1.5 * lane_width};
    Road* BRE = straight_road_create_from_center_dir_len(map, BRE_center, DIRECTION_EAST, _St2E_right_right->length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created bottom right extension road BRE at center: (%.2f, %.2f) with length: %.2f", BRE_center.x, BRE_center.y, _St2E_right_right->length);
    Road* BRW = straight_road_make_opposite_and_update_adjacents(map, BRE);
    LOG_TRACE("Created bottom right extension road BRW at center: (%.2f, %.2f) with length: %.2f", BRW->center.x, BRW->center.y, _St2E_right_right->length);
    const Coordinates RBN_center = {_St2E_right_right->end_point.x + local_turn_radius + 1.5 * lane_width, _Av2N_below_below->center.y};
    Road* RBN = straight_road_create_from_center_dir_len(map, RBN_center, DIRECTION_NORTH, _Av2N_below_below->length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created bottom right extension road RBN at center: (%.2f, %.2f) with length: %.2f", RBN_center.x, RBN_center.y, _Av2N_below_below->length);
    Road* RBS = straight_road_make_opposite_and_update_adjacents(map, RBN);
    LOG_TRACE("Created bottom right extension road RBS at center: (%.2f, %.2f) with length: %.2f", RBS->center.x, RBS->center.y, _Av2N_below_below->length);

    Road* RBN_St2WRR_connector = turn_create_and_set_connections_and_adjacents(map, RBN, _St2W_right_right, DIRECTION_CCW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from RBN to 2nd St W at center: (%.2f, %.2f)", RBN_St2WRR_connector->center.x, RBN_St2WRR_connector->center.y);
    Road* St2ERR_RBS_connector = turn_create_and_set_connections_and_adjacents(map, _St2E_right_right, RBS, DIRECTION_CW, local_turn_speed_limit, 1.0, RBN_St2WRR_connector);
    LOG_TRACE("Created connector from 2nd St E to RBS at center: (%.2f, %.2f)", St2ERR_RBS_connector->center.x, St2ERR_RBS_connector->center.y);
    Road* BRW_Av2N_below_below_connector = turn_create_and_set_connections_and_adjacents(map, BRW, _Av2N_below_below, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from BRW to 2nd Av N at center: (%.2f, %.2f)", BRW_Av2N_below_below_connector->center.x, BRW_Av2N_below_below_connector->center.y);
    Road* Av2S_below_below_BRE_connector = turn_create_and_set_connections_and_adjacents(map, _Av2S_below_below, BRE, DIRECTION_CCW, local_turn_speed_limit, 1.0, BRW_Av2N_below_below_connector);
    LOG_TRACE("Created connector from 2nd Av S to BRE at center: (%.2f, %.2f)", Av2S_below_below_BRE_connector->center.x, Av2S_below_below_BRE_connector->center.y);
    Road* RBS_BRW_connector = turn_create_and_set_connections_and_adjacents(map, RBS, BRW, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from RBS to BRW at center: (%.2f, %.2f)", RBS_BRW_connector->center.x, RBS_BRW_connector->center.y);
    Road* BRE_RBN_connector = turn_create_and_set_connections_and_adjacents(map, BRE, RBN, DIRECTION_CCW, local_turn_speed_limit, 1.0, RBS_BRW_connector);
    LOG_TRACE("Created connector from BRE to RBN at center: (%.2f, %.2f)", BRE_RBN_connector->center.x, BRE_RBN_connector->center.y);

    road_set_name(BRE, "Newton St E");
    road_set_name(BRW, "Newton St W");
    road_set_name(RBN, "Leibniz Av N");
    road_set_name(RBS, "Leibniz Av S");
    road_set_name(RBN_St2WRR_connector, "Leibniz Av N -> 2nd St W");
    road_set_name(BRW_Av2N_below_below_connector, "Newton St W -> 2nd Av N");
    road_set_name(Av2S_below_below_BRE_connector, "2nd Av S -> Newton St E");
    road_set_name(BRE_RBN_connector, "Newton St E -> Leibniz Av N");
    road_set_name(RBS_BRW_connector, "Leibniz Av S -> Newton St W");
    road_set_name(St2ERR_RBS_connector, "2nd St E -> Leibniz Av S");
    LOG_DEBUG("Bottom right extension roads created: Newton St E, Newton St W, Leibniz Av N, Leibniz Av S");

    // Bottom Left Extension (BLE, BLW, LBN, LBS)
    const Coordinates BLE_center = {_St2E_left_left->center.x, _Av1N_below_below->start_point.y - local_turn_radius - 1.5 * lane_width};
    Road* BLE = straight_road_create_from_center_dir_len(map, BLE_center, DIRECTION_EAST, _St2E_left_left->length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created bottom left extension road BLE at center: (%.2f, %.2f) with length: %.2f", BLE_center.x, BLE_center.y, _St2E_left_left->length);
    Road* BLW = straight_road_make_opposite_and_update_adjacents(map, BLE);
    LOG_TRACE("Created bottom left extension road BLW at center: (%.2f, %.2f) with length: %.2f", BLW->center.x, BLW->center.y, _St2E_left_left->length);
    const Coordinates LBN_center = {_St2E_left_left->start_point.x - local_turn_radius - lane_width / 2, _Av1N_below_below->center.y};
    Road* LBN = straight_road_create_from_center_dir_len(map, LBN_center, DIRECTION_NORTH, _Av1N_below_below->length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created bottom left extension road LBN at center: (%.2f, %.2f) with length: %.2f", LBN_center.x, LBN_center.y, _Av1N_below_below->length);
    Road* LBS = straight_road_make_opposite_and_update_adjacents(map, LBN);
    LOG_TRACE("Created bottom left extension road LBS at center: (%.2f, %.2f) with length: %.2f", LBS->center.x, LBS->center.y, _Av1N_below_below->length);

    Road* LBN_St2ELL_connector = turn_create_and_set_connections_and_adjacents(map, LBN, _St2E_left_left, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from LBN to 2nd St E at center: (%.2f, %.2f)", LBN_St2ELL_connector->center.x, LBN_St2ELL_connector->center.y);
    Road* St2WLL_LBS_connector = turn_create_and_set_connections_and_adjacents(map, _St2W_left_left, LBS, DIRECTION_CCW, local_turn_speed_limit, 1.0, LBN_St2ELL_connector);
    LOG_TRACE("Created connector from 2nd St W to LBS at center: (%.2f, %.2f)", St2WLL_LBS_connector->center.x, St2WLL_LBS_connector->center.y);
    Road* BLW_LBN_connector = turn_create_and_set_connections_and_adjacents(map, BLW, LBN, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from BLW to LBN at center: (%.2f, %.2f)", BLW_LBN_connector->center.x, BLW_LBN_connector->center.y);
    Road* Av1S_below_below_BLW_connector = turn_create_and_set_connections_and_adjacents(map, _Av1S_below_below, BLW, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from 1st Av S to BLW at center: (%.2f, %.2f)", Av1S_below_below_BLW_connector->center.x, Av1S_below_below_BLW_connector->center.y);
    Road* LBS_BLE_connector = turn_create_and_set_connections_and_adjacents(map, LBS, BLE, DIRECTION_CCW, local_turn_speed_limit, 1.0, BLW_LBN_connector);
    LOG_TRACE("Created connector from LBS to BLE at center: (%.2f, %.2f)", LBS_BLE_connector->center.x, LBS_BLE_connector->center.y);
    Road* BLE_Av1N_connector = turn_create_and_set_connections_and_adjacents(map, BLE, _Av1N_below_below, DIRECTION_CCW, local_turn_speed_limit, 1.0, Av1S_below_below_BLW_connector);
    LOG_TRACE("Created connector from BLE to 1st Av N at center: (%.2f, %.2f)", BLE_Av1N_connector->center.x, BLE_Av1N_connector->center.y);

    road_set_name(BLE, "Turing St E");
    road_set_name(BLW, "Turing St W");
    road_set_name(LBN, "Von Neumann Av N");
    road_set_name(LBS, "Von Neumann Av S");
    road_set_name(LBN_St2ELL_connector, "Von Neumann Av N -> 2nd St E");
    road_set_name(BLW_LBN_connector, "Turing St W -> Von Neumann Av N");
    road_set_name(Av1S_below_below_BLW_connector, "1st Av S -> Turing St W");
    road_set_name(BLE_Av1N_connector, "Turing St E -> 1st Av N");
    road_set_name(LBS_BLE_connector, "Von Neumann Av S -> Turing St E");
    road_set_name(St2WLL_LBS_connector, "2nd St W -> Von Neumann Av S");
    LOG_DEBUG("Bottom left extension roads created: Turing St E, Turing St W, Von Neumann Av N, Von Neumann Av S");

    // Top Left Extension (TLE, TLW, LTN, LTS)
    const Coordinates TLE_center = {_St1E_left_left->center.x, _Av1N_above_above->end_point.y + local_turn_radius + lane_width / 2};
    Road* TLE = straight_road_create_from_center_dir_len(map, TLE_center, DIRECTION_EAST, _St1E_left_left->length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created top left extension road TLE at center: (%.2f, %.2f) with length: %.2f", TLE_center.x, TLE_center.y, _St1E_left_left->length);
    Road* TLW = straight_road_make_opposite_and_update_adjacents(map, TLE);
    LOG_TRACE("Created top left extension road TLW at center: (%.2f, %.2f) with length: %.2f", TLW->center.x, TLW->center.y, _St1E_left_left->length);
    const Coordinates LTN_center = {_St1E_left_left->start_point.x - local_turn_radius - lane_width / 2, _Av1N_above_above->center.y};
    Road* LTN = straight_road_create_from_center_dir_len(map, LTN_center, DIRECTION_NORTH, _Av1N_above_above->length, 1, lane_width, local_speed_limit, 1.0);
    LOG_TRACE("Created top left extension road LTN at center: (%.2f, %.2f) with length: %.2f", LTN_center.x, LTN_center.y, _Av1N_above_above->length);
    Road* LTS = straight_road_make_opposite_and_update_adjacents(map, LTN);
    LOG_TRACE("Created top left extension road LTS at center: (%.2f, %.2f) with length: %.2f", LTS->center.x, LTS->center.y, _Av1N_above_above->length);

    Road* LTN_TLE_connector = turn_create_and_set_connections_and_adjacents(map, LTN, TLE, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from LTN to TLE at center: (%.2f, %.2f)", LTN_TLE_connector->center.x, LTN_TLE_connector->center.y);
    Road* St1WLL_LTN_connector = turn_create_and_set_connections_and_adjacents(map, _St1W_left_left, LTN, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from 1st St W to LTN at center: (%.2f, %.2f)", St1WLL_LTN_connector->center.x, St1WLL_LTN_connector->center.y);
    Road* TLW_LTS_connector = turn_create_and_set_connections_and_adjacents(map, TLW, LTS, DIRECTION_CCW, local_turn_speed_limit, 1.0, LTN_TLE_connector);
    LOG_TRACE("Created connector from TLW to LTS at center: (%.2f, %.2f)", TLW_LTS_connector->center.x, TLW_LTS_connector->center.y);
    Road* Av1N_above_above_TLW_connector = turn_create_and_set_connections_and_adjacents(map, _Av1N_above_above, TLW, DIRECTION_CCW, local_turn_speed_limit, 1.0, NULL);
    LOG_TRACE("Created connector from 1st Av N to TLW at center: (%.2f, %.2f)", Av1N_above_above_TLW_connector->center.x, Av1N_above_above_TLW_connector->center.y);
    Road* LTS_St1E_connector = turn_create_and_set_connections_and_adjacents(map, LTS, _St1E_left_left, DIRECTION_CCW, local_turn_speed_limit, 1.0, St1WLL_LTN_connector);
    LOG_TRACE("Created connector from LTS to 1st St E at center: (%.2f, %.2f)", LTS_St1E_connector->center.x, LTS_St1E_connector->center.y);
    Road* TLE_Av1S_connector = turn_create_and_set_connections_and_adjacents(map, TLE, _Av1S_above_above, DIRECTION_CW, local_turn_speed_limit, 1.0, Av1N_above_above_TLW_connector);
    LOG_TRACE("Created connector from TLE to 1st Av S at center: (%.2f, %.2f)", TLE_Av1S_connector->center.x, TLE_Av1S_connector->center.y);

    road_set_name(TLE, "Riemann St E");
    road_set_name(TLW, "Riemann St W");
    road_set_name(LTN, "Hilbert Av N");
    road_set_name(LTS, "Hilbert Av S");
    road_set_name(LTN_TLE_connector, "Hilbert Av N -> Riemann St E");
    road_set_name(St1WLL_LTN_connector, "1st St W -> Hilbert Av N");
    road_set_name(TLW_LTS_connector, "Riemann St W -> Hilbert Av S");
    road_set_name(Av1N_above_above_TLW_connector, "1st Av N -> Riemann St W");
    road_set_name(LTS_St1E_connector, "Hilbert Av S -> 1st St E");
    road_set_name(TLE_Av1S_connector, "Riemann St E -> 1st Av S");
    LOG_DEBUG("Top left extension roads created: Riemann St E, Riemann St W, Hilbert Av N, Hilbert Av S");

    // Exit and entry ramps
    Road* exit_shoulders[4];
    Road* exit_ramps[4];
    Road* entry_shoulders[4];
    Road* entry_ramps[4];

    struct { Road* hwy; Road* perp_rd; } exit_pairs[4] = {
        {I2E, _S1S_above_above}, {I1S, _S2W_right_right}, {I2W, _S1N_below_below}, {I1N, _S2E_left_left}
    };
    struct { Road* hwy; Road* perp_rd; } entry_pairs[4] = {
        {I2E, _S1N_above_above}, {I1S, _S2E_right_right}, {I2W, _S1S_below_below}, {I1N, _S2W_left_left}
    };

    for (int i = 0; i < 4; i++) {
        // Exit ramps
        Vec2D hwy_dir = direction_to_vector(exit_pairs[i].hwy->direction);
        Vec2D perp_dir = direction_to_vector(exit_pairs[i].perp_rd->direction);
        Coordinates s = vec_add(exit_pairs[i].hwy->center,
                               vec_add(vec_scale(hwy_dir, -(ramp_length + 2 * exit_pairs[i].perp_rd->width)),
                                       vec_scale(perp_dir, (exit_pairs[i].hwy->width / 2 + exit_pairs[i].perp_rd->width / 2))));
        Coordinates e = vec_add(exit_pairs[i].hwy->center,
                               vec_add(vec_scale(hwy_dir, -(2 * exit_pairs[i].perp_rd->width)),
                                       vec_scale(perp_dir, (exit_pairs[i].hwy->width / 2 + exit_pairs[i].perp_rd->width / 2))));
        exit_shoulders[i] = straight_road_create_from_start_end(map, s, e, exit_pairs[i].perp_rd->num_lanes, lane_width, state_speed_limit, 1.0);
        lane_set_exit_lane(road_get_rightmost_lane(exit_pairs[i].hwy, map), road_get_leftmost_lane(exit_shoulders[i], map),
                           (exit_pairs[i].hwy->length / 2 - (ramp_length + 2 * exit_pairs[i].perp_rd->width)) / exit_pairs[i].hwy->length,
                           (exit_pairs[i].hwy->length / 2 - (2 * exit_pairs[i].perp_rd->width)) / exit_pairs[i].hwy->length);
        exit_ramps[i] = turn_create_and_set_connections_and_adjacents(map, exit_shoulders[i], exit_pairs[i].perp_rd, DIRECTION_CW, exit_ramp_speed_limit, 1.0, NULL);
        LOG_TRACE("Created exit ramp from %s to %s at center: (%.2f, %.2f)",
                 road_get_name(exit_pairs[i].hwy), road_get_name(exit_pairs[i].perp_rd), exit_ramps[i]->center.x, exit_ramps[i]->center.y);

        // Entry ramps
        hwy_dir = direction_to_vector(entry_pairs[i].hwy->direction);
        perp_dir = direction_to_vector(entry_pairs[i].perp_rd->direction);
        s = vec_add(entry_pairs[i].hwy->center,
                    vec_add(vec_scale(hwy_dir, 2 * entry_pairs[i].perp_rd->width),
                            vec_scale(perp_dir, -(entry_pairs[i].hwy->width / 2 + entry_pairs[i].perp_rd->width / 2))));
        e = vec_add(entry_pairs[i].hwy->center,
                    vec_add(vec_scale(hwy_dir, ramp_length + 2 * entry_pairs[i].perp_rd->width),
                            vec_scale(perp_dir, -(entry_pairs[i].hwy->width / 2 + entry_pairs[i].perp_rd->width / 2))));
        entry_shoulders[i] = straight_road_create_from_start_end(map, s, e, entry_pairs[i].perp_rd->num_lanes, lane_width, interstate_speed_limit, 1.0);
        lane_set_merges_into(road_get_leftmost_lane(entry_shoulders[i], map), road_get_rightmost_lane(entry_pairs[i].hwy, map),
                             (entry_pairs[i].hwy->length / 2 + (2 * entry_pairs[i].perp_rd->width)) / entry_pairs[i].hwy->length,
                             (entry_pairs[i].hwy->length / 2 + (2 * entry_pairs[i].perp_rd->width + ramp_length)) / entry_pairs[i].hwy->length);
        entry_ramps[i] = turn_create_and_set_connections_and_adjacents(map, entry_pairs[i].perp_rd, entry_shoulders[i], DIRECTION_CW, state_turn_speed_limit, 1.0, NULL);
        LOG_TRACE("Created entry ramp from %s to %s at center: (%.2f, %.2f)",
                 road_get_name(entry_pairs[i].perp_rd), road_get_name(entry_pairs[i].hwy), entry_ramps[i]->center.x, entry_ramps[i]->center.y);
    }

    road_set_name(exit_shoulders[0], "I-2 E Exit Ramp");
    road_set_name(exit_ramps[0], "I-2 E Exit Ramp -> S-1 S");
    road_set_name(exit_shoulders[1], "I-1 S Exit Ramp");
    road_set_name(exit_ramps[1], "I-1 S Exit Ramp -> S-2 W");
    road_set_name(exit_shoulders[2], "I-2 W Exit Ramp");
    road_set_name(exit_ramps[2], "I-2 W Exit Ramp -> S-1 N");
    road_set_name(exit_shoulders[3], "I-1 N Exit Ramp");
    road_set_name(exit_ramps[3], "I-1 N Exit Ramp -> S-2 E");

    road_set_name(entry_shoulders[0], "I-2 E Entry Ramp");
    road_set_name(entry_ramps[0], "S-1 N -> I-2 E Entry Ramp");
    road_set_name(entry_shoulders[1], "I-1 S Entry Ramp");
    road_set_name(entry_ramps[1], "S-2 E -> I-1 S Entry Ramp");
    road_set_name(entry_shoulders[2], "I-2 W Entry Ramp");
    road_set_name(entry_ramps[2], "S-1 S -> I-2 W Entry Ramp");
    road_set_name(entry_shoulders[3], "I-1 N Entry Ramp");
    road_set_name(entry_ramps[3], "S-2 W -> I-1 N Entry Ramp");
    LOG_DEBUG("Exit and entry ramps created for all highways");
}