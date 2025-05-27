#include "awesim.h"
#include "road.h"
#include "utils.h"
#include <stdio.h>
#include "logging.h"


Simulation* awesim(Meters city_width, int num_cars, Seconds dt) {
    Map* map = awesim_map(city_width);
    Simulation* sim = sim_create(map, dt);
    for(int i = 0; i < num_cars; i++) {
        Car* car = car_create(car_dimensions_get_random_preset(), (CarCapabilities){CAR_ACC_PROFILE_SPORTS_CAR, from_mph(120), 0.0}, preferences_sample_random());
        const Road* random_road = map->roads[rand_int_range(0, map->num_roads - 1)];
        const Lane* random_lane = random_road->lanes[rand_int_range(0, random_road->num_lanes - 1)];
        lane_add_car((Lane*)random_lane, car);
        car_set_lane(car, random_lane);
        car_set_lane_progress(car, rand_uniform(0.2, 0.8));
        // car_set_speed(car, random_lane->speed_limit + car->preferences.average_speed_offset);
        car_set_speed(car, 0);
        sim_add_car(sim, car);
        // TODO: ensure that new cars are not spawned already crashed into other cars
    }
    LOG_INFO("Created an awesome simulator with city width: %.2f meters, %d cars, and dt: %.2f seconds", city_width, num_cars, dt);
    return sim;
}


Map* awesim_map(Meters city_width) {
    Map* map = map_create();

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

    // Interstate roads
    const Coordinates I2E_center = coordinates_create(0, city_height_half);
    const Meters I2E_length = city_width * 0.93;
    StraightRoad* I2E = straight_road_create_from_center_dir_len(I2E_center, DIRECTION_EAST, I2E_length, interstate_num_lanes, lane_width, interstate_speed_limit, 1);

    const Coordinates I2W_center = coordinates_create(0, -city_height_half);
    const Meters I2W_length = city_width * 0.93;
    StraightRoad* I2W = straight_road_create_from_center_dir_len(I2W_center, DIRECTION_WEST, I2W_length, interstate_num_lanes, lane_width, interstate_speed_limit, 1);

    const Coordinates I1N_center = coordinates_create(-city_width_half, 0);
    const Meters I1N_length = city_height * 0.93;
    StraightRoad* I1N = straight_road_create_from_center_dir_len(I1N_center, DIRECTION_NORTH, I1N_length, interstate_num_lanes, lane_width, interstate_speed_limit, 1);

    const Coordinates I1S_center = coordinates_create(city_width_half, 0);
    const Meters I1S_length = city_height * 0.93;
    StraightRoad* I1S = straight_road_create_from_center_dir_len(I1S_center, DIRECTION_SOUTH, I1S_length, interstate_num_lanes, lane_width, interstate_speed_limit, 1);

    map_add_straight_road(map, I2E);
    road_set_name((Road*)I2E, "I-2 E");
    map_add_straight_road(map, I2W);
    road_set_name((Road*)I2W, "I-2 W");
    map_add_straight_road(map, I1N);
    road_set_name((Road*)I1N, "I-1 N");
    map_add_straight_road(map, I1S);
    road_set_name((Road*)I1S, "I-1 S");

    // Interstate connectors
    Turn* interstate_connectors[4];
    interstate_connectors[0] = turn_create_and_set_connections_and_adjacents(I2E, I1S, DIRECTION_CW, interstate_turn_speed_limit, 1.0, NULL); // NE
    interstate_connectors[1] = turn_create_and_set_connections_and_adjacents(I1S, I2W, DIRECTION_CW, interstate_turn_speed_limit, 1.0, NULL); // SE
    interstate_connectors[2] = turn_create_and_set_connections_and_adjacents(I2W, I1N, DIRECTION_CW, interstate_turn_speed_limit, 1.0, NULL); // SW
    interstate_connectors[3] = turn_create_and_set_connections_and_adjacents(I1N, I2E, DIRECTION_CW, interstate_turn_speed_limit, 1.0, NULL); // NW

    for (int i = 0; i < 4; i++) {
        map_add_turn(map, interstate_connectors[i]);
    }
    road_set_name((Road*)interstate_connectors[0], "I-2 E -> I-1 S");
    road_set_name((Road*)interstate_connectors[1], "I-1 S -> I-2 W");
    road_set_name((Road*)interstate_connectors[2], "I-2 W -> I-1 N");
    road_set_name((Road*)interstate_connectors[3], "I-1 N -> I-2 E");

    // State highways
    const Meters state_hgway_width = state_num_lanes * lane_width;
    const Meters state_edge_offset = I2E->width / 2 + 2 * state_hgway_width;
    double S1_x_offset = 0.5 * lane_width * state_num_lanes;
    double S2_y_offset = S1_x_offset;

    const Coordinates S1N_start = coordinates_create(S1_x_offset, -city_height_half + state_edge_offset);
    const Coordinates S1N_end = coordinates_create(S1_x_offset, city_height_half - state_edge_offset);
    StraightRoad* _S1N = straight_road_create_from_start_end(S1N_start, S1N_end, state_num_lanes, lane_width, state_speed_limit, 1.0);
    StraightRoad* _S1S = straight_road_make_opposite_and_set_adjacent_left(_S1N);

    const Coordinates S2E_start = coordinates_create(-city_width_half + state_edge_offset, -S2_y_offset);
    const Coordinates S2E_end = coordinates_create(city_width_half - state_edge_offset, -S2_y_offset);
    StraightRoad* _S2E = straight_road_create_from_start_end(S2E_start, S2E_end, state_num_lanes, lane_width, state_speed_limit, 1.0);
    StraightRoad* _S2W = straight_road_make_opposite_and_set_adjacent_left(_S2E);

    map_add_straight_road(map, _S1N);
    road_set_name((Road*)_S1N, "S-1 N");
    map_add_straight_road(map, _S1S);
    road_set_name((Road*)_S1S, "S-1 S");
    map_add_straight_road(map, _S2E);
    road_set_name((Road*)_S2E, "S-2 E");
    map_add_straight_road(map, _S2W);
    road_set_name((Road*)_S2W, "S-2 W");

    // Central intersection
    Intersection* intersection = intersection_create_from_crossing_roads_and_update_connections(_S2E, _S2W, _S1N, _S1S, false, state_turn_radius, false, false, state_speed_limit, 1.0);
    map_add_intersection(map, intersection);
    road_set_name((Road*)intersection, "S-2 & S-1 Intersection");

    StraightRoad* _S2E_left = _S2E;
    StraightRoad* _S2E_right = (StraightRoad*)intersection->road_eastbound_to;
    StraightRoad* _S2W_right = _S2W;
    StraightRoad* _S2W_left = (StraightRoad*)intersection->road_westbound_to;
    StraightRoad* _S1N_below = _S1N;
    StraightRoad* _S1N_above = (StraightRoad*)intersection->road_northbound_to;
    StraightRoad* _S1S_above = _S1S;
    StraightRoad* _S1S_below = (StraightRoad*)intersection->road_southbound_to;

    map_add_straight_road(map, _S2E_right);
    road_set_name((Road*)_S2E_right, "S-2 E");
    map_add_straight_road(map, _S2W_left);
    road_set_name((Road*)_S2W_left, "S-2 W");
    map_add_straight_road(map, _S1N_above);
    road_set_name((Road*)_S1N_above, "S-1 N");
    map_add_straight_road(map, _S1S_below);
    road_set_name((Road*)_S1S_below, "S-1 S");

    // 1st Avenue
    const Coordinates _Av1N_center = vec_midpoint(_S2E_left->base.center, _S2W_left->base.center);
    const Meters av1_length = city_height - 4 * I2E->width;
    StraightRoad* _Av1N = straight_road_create_from_center_dir_len(_Av1N_center, DIRECTION_NORTH, av1_length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* _Av1S = straight_road_make_opposite_and_set_adjacent_left(_Av1N);

    map_add_straight_road(map, _Av1N);
    road_set_name((Road*)_Av1N, "1st Av N");
    map_add_straight_road(map, _Av1S);
    road_set_name((Road*)_Av1S, "1st Av S");

    Intersection* Av1S2_intersection = intersection_create_from_crossing_roads_and_update_connections(_S2E_left, _S2W_left, _Av1N, _Av1S, false, local_turn_radius, false, false, local_speed_limit, 1.0);
    map_add_intersection(map, Av1S2_intersection);
    road_set_name((Road*)Av1S2_intersection, "S-2 & 1st Av Intersection");

    StraightRoad* _S2E_left_left = _S2E_left;
    StraightRoad* _S2E_left_new = (StraightRoad*)Av1S2_intersection->road_eastbound_to;
    StraightRoad* _S2W_left_left = (StraightRoad*)Av1S2_intersection->road_westbound_to;
    StraightRoad* _Av1N_below = _Av1N;
    StraightRoad* _Av1N_above = (StraightRoad*)Av1S2_intersection->road_northbound_to;
    StraightRoad* _Av1S_above = _Av1S;
    StraightRoad* _Av1S_below = (StraightRoad*)Av1S2_intersection->road_southbound_to;

    map_add_straight_road(map, _S2E_left_new);
    road_set_name((Road*)_S2E_left_new, "S-2 E");
    map_add_straight_road(map, _S2W_left_left);
    road_set_name((Road*)_S2W_left_left, "S-2 W");
    map_add_straight_road(map, _Av1N_above);
    road_set_name((Road*)_Av1N_above, "1st Av N");
    map_add_straight_road(map, _Av1S_below);
    road_set_name((Road*)_Av1S_below, "1st Av S");

    // 2nd Avenue
    const Coordinates _Av2N_center = vec_midpoint(_S2E_right->base.center, _S2W_right->base.center);
    StraightRoad* _Av2N = straight_road_create_from_center_dir_len(_Av2N_center, DIRECTION_NORTH, av1_length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* _Av2S = straight_road_make_opposite_and_set_adjacent_left(_Av2N);

    map_add_straight_road(map, _Av2N);
    road_set_name((Road*)_Av2N, "2nd Av N");
    map_add_straight_road(map, _Av2S);
    road_set_name((Road*)_Av2S, "2nd Av S");

    Intersection* Av2S2_intersection = intersection_create_from_crossing_roads_and_update_connections(_S2E_right, _S2W_right, _Av2N, _Av2S, false, local_turn_radius, false, false, local_speed_limit, 1.0);
    map_add_intersection(map, Av2S2_intersection);
    road_set_name((Road*)Av2S2_intersection, "S-2 & 2nd Av Intersection");

    StraightRoad* _S2E_right_right = (StraightRoad*)Av2S2_intersection->road_eastbound_to;
    StraightRoad* _S2W_right_right = _S2W_right;
    StraightRoad* _S2W_right_new = (StraightRoad*)Av2S2_intersection->road_westbound_to;
    StraightRoad* _Av2N_below = _Av2N;
    StraightRoad* _Av2N_above = (StraightRoad*)Av2S2_intersection->road_northbound_to;
    StraightRoad* _Av2S_above = _Av2S;
    StraightRoad* _Av2S_below = (StraightRoad*)Av2S2_intersection->road_southbound_to;

    map_add_straight_road(map, _S2E_right_right);
    road_set_name((Road*)_S2E_right_right, "S-2 E");
    map_add_straight_road(map, _S2W_right_new);
    road_set_name((Road*)_S2W_right_new, "S-2 W");
    map_add_straight_road(map, _Av2N_above);
    road_set_name((Road*)_Av2N_above, "2nd Av N");
    map_add_straight_road(map, _Av2S_below);
    road_set_name((Road*)_Av2S_below, "2nd Av S");

    // 1st Street
    const Coordinates _St1E_center = vec_midpoint(_S1N_above->base.center, _S1S_above->base.center);
    const Meters st1_length = city_width - 4 * I2E->width;
    StraightRoad* _St1E = straight_road_create_from_center_dir_len(_St1E_center, DIRECTION_EAST, st1_length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* _St1W = straight_road_make_opposite_and_set_adjacent_left(_St1E);

    map_add_straight_road(map, _St1E);
    road_set_name((Road*)_St1E, "1st St E");
    map_add_straight_road(map, _St1W);
    road_set_name((Road*)_St1W, "1st St W");

    Intersection* St1S1_intersection = intersection_create_from_crossing_roads_and_update_connections(_St1E, _St1W, _S1N_above, _S1S_above, false, local_turn_radius, false, false, local_speed_limit, 1.0);
    map_add_intersection(map, St1S1_intersection);
    road_set_name((Road*)St1S1_intersection, "1st St & S-1 Intersection");

    StraightRoad* _St1E_left = _St1E;
    StraightRoad* _St1E_right = (StraightRoad*)St1S1_intersection->road_eastbound_to;
    StraightRoad* _St1W_right = _St1W;
    StraightRoad* _St1W_left = (StraightRoad*)St1S1_intersection->road_westbound_to;
    StraightRoad* _S1N_above_above = (StraightRoad*)St1S1_intersection->road_northbound_to;
    StraightRoad* _S1S_above_above = _S1S_above;
    StraightRoad* _S1S_above_new = (StraightRoad*)St1S1_intersection->road_southbound_to;

    map_add_straight_road(map, _St1E_right);
    road_set_name((Road*)_St1E_right, "1st St E");
    map_add_straight_road(map, _St1W_left);
    road_set_name((Road*)_St1W_left, "1st St W");
    map_add_straight_road(map, _S1N_above_above);
    road_set_name((Road*)_S1N_above_above, "S-1 N");
    map_add_straight_road(map, _S1S_above_new);
    road_set_name((Road*)_S1S_above_new, "S-1 S");

    // 2nd Street
    const Coordinates _St2E_center = vec_midpoint(_S1N_below->base.center, _S1S_below->base.center);
    StraightRoad* _St2E = straight_road_create_from_center_dir_len(_St2E_center, DIRECTION_EAST, st1_length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* _St2W = straight_road_make_opposite_and_set_adjacent_left(_St2E);

    map_add_straight_road(map, _St2E);
    road_set_name((Road*)_St2E, "2nd St E");
    map_add_straight_road(map, _St2W);
    road_set_name((Road*)_St2W, "2nd St W");

    Intersection* St2S1_intersection = intersection_create_from_crossing_roads_and_update_connections(_St2E, _St2W, _S1N_below, _S1S_below, false, local_turn_radius, false, false, local_speed_limit, 1.0);
    map_add_intersection(map, St2S1_intersection);
    road_set_name((Road*)St2S1_intersection, "2nd St & S-1 Intersection");

    StraightRoad* _St2E_left = _St2E;
    StraightRoad* _St2E_right = (StraightRoad*)St2S1_intersection->road_eastbound_to;
    StraightRoad* _St2W_right = _St2W;
    StraightRoad* _St2W_left = (StraightRoad*)St2S1_intersection->road_westbound_to;
    StraightRoad* _S1N_below_below = _S1N_below;
    StraightRoad* _S1N_below_new = (StraightRoad*)St2S1_intersection->road_northbound_to;
    StraightRoad* _S1S_below_below = (StraightRoad*)St2S1_intersection->road_southbound_to;

    map_add_straight_road(map, _St2E_right);
    road_set_name((Road*)_St2E_right, "2nd St E");
    map_add_straight_road(map, _St2W_left);
    road_set_name((Road*)_St2W_left, "2nd St W");
    map_add_straight_road(map, _S1N_below_new);
    road_set_name((Road*)_S1N_below_new, "S-1 N");
    map_add_straight_road(map, _S1S_below_below);
    road_set_name((Road*)_S1S_below_below, "S-1 S");

    // 1st St and 1st Av intersection
    Intersection* St1Av1_intersection = intersection_create_from_crossing_roads_and_update_connections(_St1E_left, _St1W_left, _Av1N_above, _Av1S_above, local_four_way_stop, local_turn_radius, false, false, local_speed_limit, 1.0);
    map_add_intersection(map, St1Av1_intersection);
    road_set_name((Road*)St1Av1_intersection, "1st St & 1st Av Intersection");

    StraightRoad* _St1E_left_left = _St1E_left;
    StraightRoad* _St1E_left_new = (StraightRoad*)St1Av1_intersection->road_eastbound_to;
    StraightRoad* _St1W_left_left = (StraightRoad*)St1Av1_intersection->road_westbound_to;
    StraightRoad* _Av1N_above_above = (StraightRoad*)St1Av1_intersection->road_northbound_to;
    StraightRoad* _Av1S_above_above = _Av1S_above;
    StraightRoad* _Av1S_above_new = (StraightRoad*)St1Av1_intersection->road_southbound_to;

    map_add_straight_road(map, _St1E_left_new);
    road_set_name((Road*)_St1E_left_new, "1st St E");
    map_add_straight_road(map, _St1W_left_left);
    road_set_name((Road*)_St1W_left_left, "1st St W");
    map_add_straight_road(map, _Av1N_above_above);
    road_set_name((Road*)_Av1N_above_above, "1st Av N");
    map_add_straight_road(map, _Av1S_above_new);
    road_set_name((Road*)_Av1S_above_new, "1st Av S");

    // 1st St and 2nd Av intersection
    Intersection* St1Av2_intersection = intersection_create_from_crossing_roads_and_update_connections(_St1E_right, _St1W_right, _Av2N_above, _Av2S_above, local_four_way_stop, local_turn_radius, false, false, local_speed_limit, 1.0);
    map_add_intersection(map, St1Av2_intersection);
    road_set_name((Road*)St1Av2_intersection, "1st St & 2nd Av Intersection");

    StraightRoad* _St1E_right_right = (StraightRoad*)St1Av2_intersection->road_eastbound_to;
    StraightRoad* _St1W_right_right = _St1W_right;
    StraightRoad* _St1W_right_new = (StraightRoad*)St1Av2_intersection->road_westbound_to;
    StraightRoad* _Av2N_above_above = (StraightRoad*)St1Av2_intersection->road_northbound_to;
    StraightRoad* _Av2S_above_above = _Av2S_above;
    StraightRoad* _Av2S_above_new = (StraightRoad*)St1Av2_intersection->road_southbound_to;

    map_add_straight_road(map, _St1E_right_right);
    road_set_name((Road*)_St1E_right_right, "1st St E");
    map_add_straight_road(map, _St1W_right_new);
    road_set_name((Road*)_St1W_right_new, "1st St W");
    map_add_straight_road(map, _Av2N_above_above);
    road_set_name((Road*)_Av2N_above_above, "2nd Av N");
    map_add_straight_road(map, _Av2S_above_new);
    road_set_name((Road*)_Av2S_above_new, "2nd Av S");

    // 2nd St and 1st Av intersection
    Intersection* St2Av1_intersection = intersection_create_from_crossing_roads_and_update_connections(_St2E_left, _St2W_left, _Av1N_below, _Av1S_below, local_four_way_stop, local_turn_radius, false, false, local_speed_limit, 1.0);
    map->intersections[map->num_intersections++] = St2Av1_intersection;
    road_set_name((Road*)St2Av1_intersection, "2nd St & 1st Av Intersection");

    StraightRoad* _St2E_left_left = _St2E_left;
    StraightRoad* _St2E_left_new = (StraightRoad*)St2Av1_intersection->road_eastbound_to;
    StraightRoad* _St2W_left_left = (StraightRoad*)St2Av1_intersection->road_westbound_to;
    StraightRoad* _Av1N_below_below = _Av1N_below;
    StraightRoad* _Av1N_below_new = (StraightRoad*)St2Av1_intersection->road_northbound_to;
    StraightRoad* _Av1S_below_below = (StraightRoad*)St2Av1_intersection->road_southbound_to;

    map_add_straight_road(map, _St2E_left_new);
    road_set_name((Road*)_St2E_left_new, "2nd St E");
    map_add_straight_road(map, _St2W_left_left);
    road_set_name((Road*)_St2W_left_left, "2nd St W");
    map_add_straight_road(map, _Av1N_below_new);
    road_set_name((Road*)_Av1N_below_new, "1st Av N");
    map_add_straight_road(map, _Av1S_below_below);
    road_set_name((Road*)_Av1S_below_below, "1st Av S");

    // 2nd St and 2nd Av intersection
    Intersection* St2Av2_intersection = intersection_create_from_crossing_roads_and_update_connections(_St2E_right, _St2W_right, _Av2N_below, _Av2S_below, local_four_way_stop, local_turn_radius, false, false, local_speed_limit, 1.0);
    map_add_intersection(map, St2Av2_intersection);
    road_set_name((Road*)St2Av2_intersection, "2nd St & 2nd Av Intersection");

    StraightRoad* _St2E_right_right = (StraightRoad*)St2Av2_intersection->road_eastbound_to;
    StraightRoad* _St2W_right_right = _St2W_right;
    StraightRoad* _St2W_right_new = (StraightRoad*)St2Av2_intersection->road_westbound_to;
    StraightRoad* _Av2N_below_below = _Av2N_below;
    StraightRoad* _Av2N_below_new = (StraightRoad*)St2Av2_intersection->road_northbound_to;
    StraightRoad* _Av2S_below_below = (StraightRoad*)St2Av2_intersection->road_southbound_to;

    map_add_straight_road(map, _St2E_right_right);
    road_set_name((Road*)_St2E_right_right, "2nd St E");
    map_add_straight_road(map, _St2W_right_new);
    road_set_name((Road*)_St2W_right_new, "2nd St W");
    map_add_straight_road(map, _Av2N_below_new);
    road_set_name((Road*)_Av2N_below_new, "2nd Av N");
    map_add_straight_road(map, _Av2S_below_below);
    road_set_name((Road*)_Av2S_below_below, "2nd Av S");

    // Top Right Extension (TRE, TRW, RTN, RTS)
    const Coordinates TRE_center = {_St1E_right_right->base.center.x, _Av2N_above_above->end_point.y + local_turn_radius + 0.5 * lane_width};
    StraightRoad* TRE = straight_road_create_from_center_dir_len(TRE_center, DIRECTION_EAST, _St1E_right_right->length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* TRW = straight_road_make_opposite_and_set_adjacent_left(TRE);
    const Coordinates RTN_center = {_St1E_right_right->end_point.x + local_turn_radius + 1.5 * lane_width, _Av2N_above_above->base.center.y};
    StraightRoad* RTN = straight_road_create_from_center_dir_len(RTN_center, DIRECTION_NORTH, _Av2N_above_above->length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* RTS = straight_road_make_opposite_and_set_adjacent_left(RTN);

    Turn* _Av2N_above_above_TRE_connector = turn_create_and_set_connections_and_adjacents(_Av2N_above_above, TRE, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    Turn* TRW_Av2S_above_above_connector = turn_create_and_set_connections_and_adjacents(TRW, _Av2S_above_above, DIRECTION_CCW, local_turn_speed_limit, 1.0, _Av2N_above_above_TRE_connector);
    Turn* RTN_TRW_connector = turn_create_and_set_connections_and_adjacents(RTN, TRW, DIRECTION_CCW, local_turn_speed_limit, 1.0, NULL);
    Turn* TRE_RTS_connector = turn_create_and_set_connections_and_adjacents(TRE, RTS, DIRECTION_CW, local_turn_speed_limit, 1.0, RTN_TRW_connector);
    Turn* St1ERR_RTN_connector = turn_create_and_set_connections_and_adjacents(_St1E_right_right, RTN, DIRECTION_CCW, local_turn_speed_limit, 1.0, NULL);
    Turn* RTS_St1WRR_connector = turn_create_and_set_connections_and_adjacents(RTS, _St1W_right_right, DIRECTION_CW, local_turn_speed_limit, 1.0, St1ERR_RTN_connector);

    map_add_straight_road(map, TRE);
    road_set_name((Road*)TRE, "Euler St E");
    map_add_straight_road(map, TRW);
    road_set_name((Road*)TRW, "Euler St W");
    map_add_straight_road(map, RTN);
    road_set_name((Road*)RTN, "Gauss Av N");
    map_add_straight_road(map, RTS);
    road_set_name((Road*)RTS, "Gauss Av S");
    map_add_turn(map, _Av2N_above_above_TRE_connector);
    road_set_name((Road*)_Av2N_above_above_TRE_connector, "2nd Av N -> Euler St E");
    map_add_turn(map, TRW_Av2S_above_above_connector);
    road_set_name((Road*)TRW_Av2S_above_above_connector, "Euler St W -> 2nd Av S");
    map_add_turn(map, RTN_TRW_connector);
    road_set_name((Road*)RTN_TRW_connector, "Gauss Av N -> Euler St W");
    map_add_turn(map, TRE_RTS_connector);
    road_set_name((Road*)TRE_RTS_connector, "Euler St E -> Gauss Av S");
    map_add_turn(map, RTS_St1WRR_connector);
    road_set_name((Road*)RTS_St1WRR_connector, "Gauss Av S -> 1st St W");
    map_add_turn(map, St1ERR_RTN_connector);
    road_set_name((Road*)St1ERR_RTN_connector, "1st St E -> Gauss Av N");

    // Bottom Right Extension (BRE, BRW, RBN, RBS)
    const Coordinates BRE_center = {_St2E_right_right->base.center.x, _Av2N_below_below->start_point.y - local_turn_radius - 1.5 * lane_width};
    StraightRoad* BRE = straight_road_create_from_center_dir_len(BRE_center, DIRECTION_EAST, _St2E_right_right->length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* BRW = straight_road_make_opposite_and_set_adjacent_left(BRE);
    const Coordinates RBN_center = {_St2E_right_right->end_point.x + local_turn_radius + 1.5 * lane_width, _Av2N_below_below->base.center.y};
    StraightRoad* RBN = straight_road_create_from_center_dir_len(RBN_center, DIRECTION_NORTH, _Av2N_below_below->length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* RBS = straight_road_make_opposite_and_set_adjacent_left(RBN);

    Turn* RBN_St2WRR_connector = turn_create_and_set_connections_and_adjacents(RBN, _St2W_right_right, DIRECTION_CCW, local_turn_speed_limit, 1.0, NULL);
    Turn* St2ERR_RBS_connector = turn_create_and_set_connections_and_adjacents(_St2E_right_right, RBS, DIRECTION_CW, local_turn_speed_limit, 1.0, RBN_St2WRR_connector);
    Turn* BRW_Av2N_below_below_connector = turn_create_and_set_connections_and_adjacents(BRW, _Av2N_below_below, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    Turn* Av2S_below_below_BRE_connector = turn_create_and_set_connections_and_adjacents(_Av2S_below_below, BRE, DIRECTION_CCW, local_turn_speed_limit, 1.0, BRW_Av2N_below_below_connector);
    Turn* RBS_BRW_connector = turn_create_and_set_connections_and_adjacents(RBS, BRW, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    Turn* BRE_RBN_connector = turn_create_and_set_connections_and_adjacents(BRE, RBN, DIRECTION_CCW, local_turn_speed_limit, 1.0, RBS_BRW_connector);

    map_add_straight_road(map, BRE);
    road_set_name((Road*)BRE, "Newton St E");
    map_add_straight_road(map, BRW);
    road_set_name((Road*)BRW, "Newton St W");
    map_add_straight_road(map, RBN);
    road_set_name((Road*)RBN, "Leibniz Av N");
    map_add_straight_road(map, RBS);
    road_set_name((Road*)RBS, "Leibniz Av S");
    map_add_turn(map, RBN_St2WRR_connector);
    road_set_name((Road*)RBN_St2WRR_connector, "Leibniz Av N -> 2nd St W");
    map_add_turn(map, BRW_Av2N_below_below_connector);
    road_set_name((Road*)BRW_Av2N_below_below_connector, "Newton St W -> 2nd Av N");
    map_add_turn(map, Av2S_below_below_BRE_connector);
    road_set_name((Road*)Av2S_below_below_BRE_connector, "2nd Av S -> Newton St E");
    map_add_turn(map, BRE_RBN_connector);
    road_set_name((Road*)BRE_RBN_connector, "Newton St E -> Leibniz Av N");
    map_add_turn(map, RBS_BRW_connector);
    road_set_name((Road*)RBS_BRW_connector, "Leibniz Av S -> Newton St W");
    map_add_turn(map, St2ERR_RBS_connector);
    road_set_name((Road*)St2ERR_RBS_connector, "2nd St E -> Leibniz Av S");

    // Bottom Left Extension (BLE, BLW, LBN, LBS)
    const Coordinates BLE_center = {_St2E_left_left->base.center.x, _Av1N_below_below->start_point.y - local_turn_radius - 1.5 * lane_width};
    StraightRoad* BLE = straight_road_create_from_center_dir_len(BLE_center, DIRECTION_EAST, _St2E_left_left->length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* BLW = straight_road_make_opposite_and_set_adjacent_left(BLE);
    const Coordinates LBN_center = {_St2E_left_left->start_point.x - local_turn_radius - lane_width / 2, _Av1N_below_below->base.center.y};
    StraightRoad* LBN = straight_road_create_from_center_dir_len(LBN_center, DIRECTION_NORTH, _Av1N_below_below->length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* LBS = straight_road_make_opposite_and_set_adjacent_left(LBN);

    Turn* LBN_St2ELL_connector = turn_create_and_set_connections_and_adjacents(LBN, _St2E_left_left, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    Turn* St2WLL_LBS_connector = turn_create_and_set_connections_and_adjacents(_St2W_left_left, LBS, DIRECTION_CCW, local_turn_speed_limit, 1.0, LBN_St2ELL_connector);
    Turn* BLW_LBN_connector = turn_create_and_set_connections_and_adjacents(BLW, LBN, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    Turn* Av1S_below_below_BLW_connector = turn_create_and_set_connections_and_adjacents(_Av1S_below_below, BLW, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    Turn* LBS_BLE_connector = turn_create_and_set_connections_and_adjacents(LBS, BLE, DIRECTION_CCW, local_turn_speed_limit, 1.0, BLW_LBN_connector);
    Turn* BLE_Av1N_connector = turn_create_and_set_connections_and_adjacents(BLE, _Av1N_below_below, DIRECTION_CCW, local_turn_speed_limit, 1.0, Av1S_below_below_BLW_connector);

    map_add_straight_road(map, BLE);
    road_set_name((Road*)BLE, "Turing St E");
    map_add_straight_road(map, BLW);
    road_set_name((Road*)BLW, "Turing St W");
    map_add_straight_road(map, LBN);
    road_set_name((Road*)LBN, "Von Neumann Av N");
    map_add_straight_road(map, LBS);
    road_set_name((Road*)LBS, "Von Neumann Av S");
    map_add_turn(map, LBN_St2ELL_connector);
    road_set_name((Road*)LBN_St2ELL_connector, "Von Neumann Av N -> 2nd St E");
    map_add_turn(map, BLW_LBN_connector);
    road_set_name((Road*)BLW_LBN_connector, "Turing St W -> Von Neumann Av N");
    map_add_turn(map, Av1S_below_below_BLW_connector);
    road_set_name((Road*)Av1S_below_below_BLW_connector, "1st Av S -> Turing St W");
    map_add_turn(map, BLE_Av1N_connector);
    road_set_name((Road*)BLE_Av1N_connector, "Turing St E -> 1st Av N");
    map_add_turn(map, LBS_BLE_connector);
    road_set_name((Road*)LBS_BLE_connector, "Von Neumann Av S -> Turing St E");
    map_add_turn(map, St2WLL_LBS_connector);
    road_set_name((Road*)St2WLL_LBS_connector, "2nd St W -> Von Neumann Av S");

    // Top Left Extension (TLE, TLW, LTN, LTS)
    const Coordinates TLE_center = {_St1E_left_left->base.center.x, _Av1N_above_above->end_point.y + local_turn_radius + lane_width / 2};
    StraightRoad* TLE = straight_road_create_from_center_dir_len(TLE_center, DIRECTION_EAST, _St1E_left_left->length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* TLW = straight_road_make_opposite_and_set_adjacent_left(TLE);
    const Coordinates LTN_center = {_St1E_left_left->start_point.x - local_turn_radius - lane_width / 2, _Av1N_above_above->base.center.y};
    StraightRoad* LTN = straight_road_create_from_center_dir_len(LTN_center, DIRECTION_NORTH, _Av1N_above_above->length, 1, lane_width, local_speed_limit, 1.0);
    StraightRoad* LTS = straight_road_make_opposite_and_set_adjacent_left(LTN);

    Turn* LTN_TLE_connector = turn_create_and_set_connections_and_adjacents(LTN, TLE, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    Turn* St1WLL_LTN_connector = turn_create_and_set_connections_and_adjacents(_St1W_left_left, LTN, DIRECTION_CW, local_turn_speed_limit, 1.0, NULL);
    Turn* TLW_LTS_connector = turn_create_and_set_connections_and_adjacents(TLW, LTS, DIRECTION_CCW, local_turn_speed_limit, 1.0, LTN_TLE_connector);
    Turn* Av1N_above_above_TLW_connector = turn_create_and_set_connections_and_adjacents(_Av1N_above_above, TLW, DIRECTION_CCW, local_turn_speed_limit, 1.0, NULL);
    Turn* LTS_St1E_connector = turn_create_and_set_connections_and_adjacents(LTS, _St1E_left_left, DIRECTION_CCW, local_turn_speed_limit, 1.0, St1WLL_LTN_connector);
    Turn* TLE_Av1S_connector = turn_create_and_set_connections_and_adjacents(TLE, _Av1S_above_above, DIRECTION_CW, local_turn_speed_limit, 1.0, Av1N_above_above_TLW_connector);

    map_add_straight_road(map, TLE);
    road_set_name((Road*)TLE, "Riemann St E");
    map_add_straight_road(map, TLW);
    road_set_name((Road*)TLW, "Riemann St W");
    map_add_straight_road(map, LTN);
    road_set_name((Road*)LTN, "Hilbert Av N");
    map_add_straight_road(map, LTS);
    road_set_name((Road*)LTS, "Hilbert Av S");
    map_add_turn(map, LTN_TLE_connector);
    road_set_name((Road*)LTN_TLE_connector, "Hilbert Av N -> Riemann St E");
    map_add_turn(map, St1WLL_LTN_connector);
    road_set_name((Road*)St1WLL_LTN_connector, "1st St W -> Hilbert Av N");
    map_add_turn(map, TLW_LTS_connector);
    road_set_name((Road*)TLW_LTS_connector, "Riemann St W -> Hilbert Av S");
    map_add_turn(map, Av1N_above_above_TLW_connector);
    road_set_name((Road*)Av1N_above_above_TLW_connector, "1st Av N -> Riemann St W");
    map_add_turn(map, LTS_St1E_connector);
    road_set_name((Road*)LTS_St1E_connector, "Hilbert Av S -> 1st St E");
    map_add_turn(map, TLE_Av1S_connector);
    road_set_name((Road*)TLE_Av1S_connector, "Riemann St E -> 1st Av S");

    // Exit and entry ramps
    StraightRoad* exit_shoulders[4];
    Turn* exit_ramps[4];
    StraightRoad* entry_shoulders[4];
    Turn* entry_ramps[4];

    struct { StraightRoad* hwy; StraightRoad* perp_rd; } exit_pairs[4] = {
        {I2E, _S1S_above_above}, {I1S, _S2W_right_right}, {I2W, _S1N_below_below}, {I1N, _S2E_left_left}
    };
    struct { StraightRoad* hwy; StraightRoad* perp_rd; } entry_pairs[4] = {
        {I2E, _S1N_above_above}, {I1S, _S2E_right_right}, {I2W, _S1S_below_below}, {I1N, _S2W_left_left}
    };

    for (int i = 0; i < 4; i++) {
        // Exit ramps
        Vec2D hwy_dir = direction_to_vector(exit_pairs[i].hwy->direction);
        Vec2D perp_dir = direction_to_vector(exit_pairs[i].perp_rd->direction);
        Coordinates s = vec_add(exit_pairs[i].hwy->base.center,
                               vec_add(vec_scale(hwy_dir, -(ramp_length + 2 * exit_pairs[i].perp_rd->width)),
                                       vec_scale(perp_dir, (exit_pairs[i].hwy->width / 2 + exit_pairs[i].perp_rd->width / 2))));
        Coordinates e = vec_add(exit_pairs[i].hwy->base.center,
                               vec_add(vec_scale(hwy_dir, -(2 * exit_pairs[i].perp_rd->width)),
                                       vec_scale(perp_dir, (exit_pairs[i].hwy->width / 2 + exit_pairs[i].perp_rd->width / 2))));
        exit_shoulders[i] = straight_road_create_from_start_end(s, e, exit_pairs[i].perp_rd->base.num_lanes, lane_width, state_speed_limit, 1.0);
        lane_set_exit_lane((Lane*)exit_pairs[i].hwy->base.lanes[exit_pairs[i].hwy->base.num_lanes - 1], exit_shoulders[i]->base.lanes[0],
                           (exit_pairs[i].hwy->length / 2 - (ramp_length + 2 * exit_pairs[i].perp_rd->width)) / exit_pairs[i].hwy->length,
                           (exit_pairs[i].hwy->length / 2 - (2 * exit_pairs[i].perp_rd->width)) / exit_pairs[i].hwy->length);
        exit_ramps[i] = turn_create_and_set_connections_and_adjacents(exit_shoulders[i], exit_pairs[i].perp_rd, DIRECTION_CW, exit_ramp_speed_limit, 1.0, NULL);
        map_add_straight_road(map, exit_shoulders[i]);
        map_add_turn(map, exit_ramps[i]);

        // Entry ramps
        hwy_dir = direction_to_vector(entry_pairs[i].hwy->direction);
        perp_dir = direction_to_vector(entry_pairs[i].perp_rd->direction);
        s = vec_add(entry_pairs[i].hwy->base.center,
                    vec_add(vec_scale(hwy_dir, 2 * entry_pairs[i].perp_rd->width),
                            vec_scale(perp_dir, -(entry_pairs[i].hwy->width / 2 + entry_pairs[i].perp_rd->width / 2))));
        e = vec_add(entry_pairs[i].hwy->base.center,
                    vec_add(vec_scale(hwy_dir, ramp_length + 2 * entry_pairs[i].perp_rd->width),
                            vec_scale(perp_dir, -(entry_pairs[i].hwy->width / 2 + entry_pairs[i].perp_rd->width / 2))));
        entry_shoulders[i] = straight_road_create_from_start_end(s, e, entry_pairs[i].perp_rd->base.num_lanes, lane_width, interstate_speed_limit, 1.0);
        lane_set_merges_into((Lane*)entry_shoulders[i]->base.lanes[0], entry_pairs[i].hwy->base.lanes[entry_pairs[i].hwy->base.num_lanes - 1],
                             (entry_pairs[i].hwy->length / 2 + (2 * entry_pairs[i].perp_rd->width)) / entry_pairs[i].hwy->length,
                             (entry_pairs[i].hwy->length / 2 + (2 * entry_pairs[i].perp_rd->width + ramp_length)) / entry_pairs[i].hwy->length);
        entry_ramps[i] = turn_create_and_set_connections_and_adjacents(entry_pairs[i].perp_rd, entry_shoulders[i], DIRECTION_CW, state_turn_speed_limit, 1.0, NULL);
        map_add_straight_road(map, entry_shoulders[i]);
        map_add_turn(map, entry_ramps[i]);
    }

    road_set_name((Road*)exit_shoulders[0], "I-2 E Exit Ramp");
    road_set_name((Road*)exit_ramps[0], "I-2 E Exit Ramp -> S-1 S");
    road_set_name((Road*)exit_shoulders[1], "I-1 S Exit Ramp");
    road_set_name((Road*)exit_ramps[1], "I-1 S Exit Ramp -> S-2 W");
    road_set_name((Road*)exit_shoulders[2], "I-2 W Exit Ramp");
    road_set_name((Road*)exit_ramps[2], "I-2 W Exit Ramp -> S-1 N");
    road_set_name((Road*)exit_shoulders[3], "I-1 N Exit Ramp");
    road_set_name((Road*)exit_ramps[3], "I-1 N Exit Ramp -> S-2 E");

    road_set_name((Road*)entry_shoulders[0], "I-2 E Entry Ramp");
    road_set_name((Road*)entry_ramps[0], "S-1 N -> I-2 E Entry Ramp");
    road_set_name((Road*)entry_shoulders[1], "I-1 S Entry Ramp");
    road_set_name((Road*)entry_ramps[1], "S-2 E -> I-1 S Entry Ramp");
    road_set_name((Road*)entry_shoulders[2], "I-2 W Entry Ramp");
    road_set_name((Road*)entry_ramps[2], "S-1 S -> I-2 W Entry Ramp");
    road_set_name((Road*)entry_shoulders[3], "I-1 N Entry Ramp");
    road_set_name((Road*)entry_ramps[3], "S-2 W -> I-1 N Entry Ramp");

    return map;
}

