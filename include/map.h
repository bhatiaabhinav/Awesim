#pragma once

#include <stdio.h>
#include <stdbool.h>
#include "utils.h"

// Constants
#define MAX_CARS_PER_LANE 64
#define MAX_NUM_ROADS 128
#define MAX_NUM_LANES_PER_ROAD 4
#define MAX_NUM_INTERSECTIONS 10
#define MAX_NUM_LANES_PER_INTERSECTION 32
#define MAX_NUM_LANES 350
#define LANE_DEGRADATIONS_BUCKETS 5
#define GREEN_DURATION 13
#define YELLOW_DURATION 5
#define MINOR_RED_EXTENSION 2
#define ID_NULL -1

// Typedef Definitions

typedef int LaneId;
typedef int RoadId;
typedef int IntersectionId;
typedef int CarId;

// Forward Declarations

typedef struct Car Car;
typedef struct Road Road;
typedef struct Lane Lane;
typedef struct Intersection Intersection;
typedef struct Map Map;
typedef struct Simulation Simulation;
Car* sim_get_car(Simulation* sim, CarId id);
double car_get_lane_progress(const Car* self);
CarId car_get_id(const Car* self);
int car_get_lane_rank(const Car* self);
void car_set_lane_rank(Car* self, int rank);

// Degradations: Represents grip degradation across segments of a lane.
typedef struct {
    double values[LANE_DEGRADATIONS_BUCKETS];
} Degradations;

extern const Degradations DEGRADATIONS_ZERO;

// Enum for lane geometry types.
typedef enum {
    LINEAR_LANE,
    QUARTER_ARC_LANE
} LaneType;

// Lane structure representing a single traffic lane.
struct Lane {
    LaneId id;                 // Unique identifier for the lane
    LaneType type;
    char name[64];          // Name of the lane (optional)
    Direction direction;
    Meters width;
    MetersPerSecond speed_limit;
    double grip;
    Degradations degradations;

    LaneId connections_incoming[3];    // Indices of incoming lanes (left , straight, right). On non-intersections, there is no incoming left or incomg right, there is just incoming straight (which could be coming in from a curved road, too, which would be the case when this lane is output of a simple turn connector between two perpendicular straight roads). On intersections, left means that a vehicle took a left turn to enter this lane, straight means that a vehicle entered this lane from the straight lane, and right means that a vehicle took a right turn to enter this lane.
    LaneId connections[3];             // Indices of connected lanes (left, stright, right). For simple turn connectors between two perpendicular straight roads, use "straight", since vehicles don't use an indicator in such cases. Left and right have meaning only for intersections.
    LaneId adjacents[2];               // Indices of adjacent lanes (left, right)
    LaneId merges_into_id;
    double merges_into_start;
    double merges_into_end;
    LaneId exit_lane_id;
    double exit_lane_start;
    double exit_lane_end;

    Coordinates start_point;
    Coordinates end_point;
    Coordinates mid_point; // Midpoint of the lane i.e., 50% progress point. For straight lanes, same as center; for quarter arc lanes, this is the midpoint *on* the arc.
    Coordinates center;     // for quarter Arc Lanes, this is the center of the circle. Else this is the midpoint of the start and end points.
    Meters length;

    
    Meters radius;          // Radius for quarter arc lanes
    Radians start_angle;    // Start angle for quarter arc lanes
    Radians end_angle;      // End angle for quarter arc lanes
    Quadrant quadrant;      // Quadrant for quarter arc lanes

    RoadId road_id;                    // The road this lane belongs to, if any
    IntersectionId intersection_id;    // The intersection this lane belongs to, if any
    bool is_at_intersection;        // True if this lane is at an intersection, false otherwise

    CarId cars_ids[MAX_CARS_PER_LANE]; // Array of cars currently in this lane, ordered/ranked by their position on the lane in descending order (i.e., the first car (index = 0) is the one closest to the end of the lane).
    int num_cars;                  // Number of cars currently in this lane
};


// Lane Setters
void lane_set_connection_incoming_left(Lane* self, const Lane* lane);
void lane_set_connection_incoming_straight(Lane* self, const Lane* lane);
void lane_set_connection_incoming_right(Lane* self, const Lane* lane);
void lane_set_connection_left(Lane* self, const Lane* lane);
void lane_set_connection_straight(Lane* self, const Lane* lane);
void lane_set_connection_right(Lane* self, const Lane* lane);
void lane_set_adjacents(Lane* self, const Lane* left, const Lane* right);
void lane_set_adjacent_left(Lane* self, const Lane* left);
void lane_set_adjacent_right(Lane* self, const Lane* right);
void lane_set_merges_into(Lane* self, const Lane* lane, double start, double end);
void lane_set_exit_lane(Lane* self, const Lane* lane, double start, double end);
void lane_set_speed_limit(Lane* self, MetersPerSecond speed_limit);
void lane_set_grip(Lane* self, double grip);
void lane_set_degradations(Lane* self, Degradations degradations);
void lane_set_road(Lane* self, const Road* road);
void lane_set_intersection(Lane* self, const Intersection* intersection);
void lane_set_name(Lane* self, const char* name);

// Lane Vehicle Management
void lane_add_car(Lane* self, Car* car, Simulation* sim);
void lane_remove_car(Lane* self, Car* car, Simulation* sim);
void lane_move_car(Lane* self, Car* car, Simulation* sim);

// Lane Getters
int lane_get_id(const Lane* self);
LaneType lane_get_type(const Lane* self);
Direction lane_get_direction(const Lane* self);
Meters lane_get_width(const Lane* self);
MetersPerSecond lane_get_speed_limit(const Lane* self);
double lane_get_grip(const Lane* self);
Coordinates lane_get_start_point(const Lane* self);
Coordinates lane_get_mid_point(const Lane* self);
Coordinates lane_get_end_point(const Lane* self);
Coordinates lane_get_center(const Lane* self);
Meters lane_get_length(const Lane* self);
Degradations lane_get_degradations(const Lane* self);
LaneId lane_get_connection_incoming_left_id(const Lane* self);
Lane* lane_get_connection_incoming_left(const Lane* self, Map* map);
LaneId lane_get_connection_incoming_straight_id(const Lane* self);
Lane* lane_get_connection_incoming_straight(const Lane* self, Map* map);
LaneId lane_get_connection_incoming_right_id(const Lane* self);
Lane* lane_get_connection_incoming_right(const Lane* self, Map* map);
LaneId lane_get_connection_left_id(const Lane* self);
Lane* lane_get_connection_left(const Lane* self, Map* map);
LaneId lane_get_connection_straight_id(const Lane* self);
Lane* lane_get_connection_straight(const Lane* self, Map* map);
LaneId lane_get_connection_right_id(const Lane* self);
Lane* lane_get_connection_right(const Lane* self, Map* map);
LaneId lane_get_merge_into_id(const Lane* self);
Lane* lane_get_merge_into(const Lane* self, Map* map);
LaneId lane_get_exit_lane_id(const Lane* self);
Lane* lane_get_exit_lane(const Lane* self, Map* map);
LaneId lane_get_adjacent_left_id(const Lane* self);
Lane* lane_get_adjacent_left(const Lane* self, Map* map);
LaneId lane_get_adjacent_right_id(const Lane* self);
Lane* lane_get_adjacent_right(const Lane* self, Map* map);
RoadId lane_get_road_id(const Lane* self);
Road* lane_get_road(const Lane* self, Map* map);
IntersectionId lane_get_intersection_id(const Lane* self);
bool lane_is_at_intersection(const Lane* self);
Intersection* lane_get_intersection(const Lane* self, Map* map);
int lane_get_num_cars(const Lane* self);
// Returns the car ID at the given index in the lane. The index is 0-based, where 0 is the first car in the lane (the one closest to the end of the lane), and num_cars-1 is the last car in the lane (the one closest to the start of the lane). Interpret the index as a rank.
CarId lane_get_car_id(const Lane* self, int index);
// Returns the car at the given index in the lane. The index is 0-based, where 0 is the first car in the lane (the one closest to the end of the lane), and num_cars-1 is the last car in the lane (the one closest to the start of the lane). Interpret the index as a rank.
Car* lane_get_car(const Lane* self, Simulation* sim, int index);

// Fancier functions

int lane_get_num_connections_incoming(const Lane* self);
int lane_get_num_connections(const Lane* self);
bool lane_is_merge_available(const Lane* self);
bool lane_is_exit_lane_available(const Lane* self, double progress);
bool lane_is_exit_lane_eventually_available(const Lane* self, double progress);

// Lane Creaters

Lane* lane_create(Map* map, LaneType type, Direction direction, Meters width, MetersPerSecond speed_limit, double grip, Degradations degradations);
Lane* linear_lane_create_from_center_dir_len(Map* map, Coordinates center, Direction direction, Meters length, Meters width, MetersPerSecond speed_limit, double grip, Degradations degradations);
Lane* linear_lane_create_from_start_end(Map* map, Coordinates start, Coordinates end, Meters width, MetersPerSecond speed_limit, double grip, Degradations degradations);
Lane* quarter_arc_lane_create_from_start_end(Map* map, Coordinates start_point, Coordinates end_point, Direction direction, Meters width, MetersPerSecond speed_limit, double grip, Degradations degradations);






// Road Types
typedef enum {
    STRAIGHT,
    TURN,
} RoadType;

// Road Structure
struct Road {
    int id;         // Unique identifier for the road
    RoadType type;
    char name[64];  // Name of the road
    LaneId lane_ids[MAX_NUM_LANES_PER_ROAD];
    int num_lanes; // 0 = leftmost, num_lanes-1 = rightmost
    MetersPerSecond speed_limit;
    double grip;
    Meters lane_width;
    Direction direction;
    Meters length;
    Meters width;
    Coordinates start_point;
    Coordinates end_point;
    Coordinates mid_point;  // 50% along the road.
    Coordinates center;     // For straight roads, this is same as the midpoint of the road. For turns, this is the center of the circle of the turn.
    Meters radius; // For turns, this is the radius of the turn. For straight roads, this is 0.
    Radians start_angle; // For turns, this is the start angle of the turn. For straight roads, this is 0.
    Radians end_angle;   // For turns, this is the end angle of the turn. For straight roads, this is 0.
    Quadrant quadrant; // For turns, this is the quadrant of the turn. For straight roads, this is QUADRANT_TOP_RIGHT.

    // Only for turns:
    RoadId road_from_id;
    RoadId road_to_id;
};

// Road Setters

void road_set_name(Road* self, const char* name);
void road_set_speed_limit(Road* self, MetersPerSecond speed_limit);
void road_set_grip(Road* self, double grip);

// Simple Getters

int road_get_id(const Road* self);
RoadType road_get_type(const Road* self);
const char* road_get_name(const Road* self);
int road_get_num_lanes(const Road* self);
MetersPerSecond road_get_speed_limit(const Road* self);
double road_get_grip(const Road* self);
Meters road_get_lane_width(const Road* self);
Direction road_get_direction(const Road* self);
Meters road_get_length(const Road* self);
Meters road_get_width(const Road* self);
Coordinates road_get_start_point(const Road* self);
Coordinates road_get_end_point(const Road* self);
Coordinates road_get_mid_point(const Road* self);
Coordinates road_get_center(const Road* self);

// Fancier functions

Lane* road_get_lane(const Road* self, Map* map, int index);
Lane* road_get_leftmost_lane(const Road* self, Map* map);
Lane* road_get_rightmost_lane(const Road* self, Map* map);
bool road_is_merge_available(const Road* self, Map* map);
bool road_is_exit_road_available(const Road* self, Map* map, double progress);
bool road_is_exit_road_eventually_available(const Road* self, Map* map, double progress);
Road* turn_road_get_from(const Road* self, Map* map);
Road* turn_road_get_to(const Road* self, Map* map);


// Road Creaters

Road* straight_road_create_from_center_dir_len(Map* map, Coordinates center, Direction direction, Meters length, int num_lanes, Meters lane_width, MetersPerSecond speed_limit, double grip);
Road* straight_road_create_from_start_end(Map* map, Coordinates start, Coordinates end, int num_lanes, Meters lane_width, MetersPerSecond speed_limit, double grip);
Road* straight_road_make_opposite(Map* map, const Road* road);
Road* straight_road_make_opposite_and_update_adjacents(Map* map, Road* road);

// Returns second half of the road split at the given position, and updates connections. The first half is modified in-place.
Road* straight_road_split_at_and_update_connections(Road* road, Map* map, Meters position, Meters gap);


Road* turn_create_and_set_connections_and_adjacents(Map* map, Road* road_from, Road* road_to, Direction direction, MetersPerSecond speed_limit, double grip, Road* opposite_turn);


// --- Intersection ---


// Intersection State
typedef enum {
    NS_GREEN_EW_RED,
    NS_YELLOW_EW_RED,
    ALL_RED_BEFORE_EW_GREEN,
    NS_RED_EW_GREEN,
    EW_YELLOW_NS_RED,
    ALL_RED_BEFORE_NS_GREEN,
    FOUR_WAY_STOP,
} IntersectionState;

#define NUM_TRAFFIC_CONTROL_STATES 6

extern const Seconds TRAFFIC_STATE_DURATIONS[NUM_TRAFFIC_CONTROL_STATES];

typedef struct Intersection {
    // Road base;
    IntersectionId id; // Unique identifier for the intersection
    char name[64]; // Name of the intersection
    LaneId lane_ids[MAX_NUM_LANES_PER_INTERSECTION]; // Lanes at the intersection
    int num_lanes; // Number of lanes at the intersection
    Meters lane_width; // Width of the lanes at the intersection
    MetersPerSecond speed_limit; // Speed limit at the intersection
    double grip; // Grip factor at the intersection
    Coordinates center; // Center point of the intersection
    Dimensions dimensions;
    RoadId road_eastbound_from_id;
    RoadId road_eastbound_to_id;
    RoadId road_westbound_from_id;
    RoadId road_westbound_to_id;
    RoadId road_northbound_from_id;
    RoadId road_northbound_to_id;
    RoadId road_southbound_from_id;
    RoadId road_southbound_to_id;
    Meters turn_radius;
    bool left_lane_turns_left_only;
    bool right_lane_turns_right_only;
    IntersectionState state;
    Seconds countdown;
} Intersection;

// Intersection Setters
void intersection_set_name(Intersection* self, const char* name);
void intersection_set_speed_limit(Intersection* self, MetersPerSecond speed_limit);
void intersection_set_grip(Intersection* self, double grip);

// Intersection Getters
int intersection_get_id(const Intersection* self);
const char* intersection_get_name(const Intersection* self);
int intersection_get_num_lanes(const Intersection* self);
Coordinates intersection_get_center(const Intersection* self);
IntersectionState intersection_get_state(const Intersection* self);
Seconds intersection_get_countdown(const Intersection* self);
bool intersection_get_left_lane_turns_left_only(const Intersection* self);
bool intersection_get_right_lane_turns_right_only(const Intersection* self);
Meters intersection_get_turn_radius(const Intersection* self);
Dimensions intersection_get_dimensions(const Intersection* self);
Lane* intersection_get_lane(const Intersection* self, Map* map, int index);
Road* intersection_get_road_eastbound_from(const Intersection* self, Map* map);
Road* intersection_get_road_eastbound_to(const Intersection* self, Map* map);
Road* intersection_get_road_westbound_from(const Intersection* self, Map* map);
Road* intersection_get_road_westbound_to(const Intersection* self, Map* map);
Road* intersection_get_road_northbound_from(const Intersection* self, Map* map);
Road* intersection_get_road_northbound_to(const Intersection* self, Map* map);
Road* intersection_get_road_southbound_from(const Intersection* self, Map* map);
Road* intersection_get_road_southbound_to(const Intersection* self, Map* map);

// Intersection Creaters

Intersection* intersection_create_and_form_connections(
    Map* map,
    Road* road_eastbound_from,
    Road* road_eastbound_to,
    Road* road_westbound_from,
    Road* road_westbound_to,
    Road* road_northbound_from,
    Road* road_northbound_to,
    Road* road_southbound_from,
    Road* road_southbound_to,
    bool is_four_way_stop,
    bool left_lane_turns_left_only,
    bool right_lane_turns_right_only,
    MetersPerSecond speed_limit,
    double grip
);
Intersection* intersection_create_from_crossing_roads_and_update_connections(
    Map* map,
    Road* eastbound,
    Road* westbound,
    Road* northbound,
    Road* southbound,
    bool is_four_way_stop,
    Meters turn_radius,
    bool left_lane_turns_left_only,
    bool right_lane_turns_right_only,
    MetersPerSecond speed_limit,
    double grip
);
void intersection_update(Intersection* self, Seconds dt);
void print_traffic_state(const Intersection* intersection);

Intersection* road_leads_to_intersection(const Road* road, Map* map);
Intersection* road_comes_from_intersection(const Road* road, Map* map);





// Structure representing a simulation map containing roads and intersections
typedef struct Map {
    Lane lanes[MAX_NUM_LANES];
    Road roads[MAX_NUM_ROADS];
    Intersection intersections[MAX_NUM_INTERSECTIONS];
    int num_lanes;  // Total number of lanes in the map
    int num_roads;  // Total number of roads in the map
    int num_intersections;  // Total number of intersections in the map
} Map;

// Clears the map and initializes it to an empty state
void map_init(Map* self);
Lane* map_get_new_lane(Map* self);
Road* map_get_new_road(Map* self);
Intersection* map_get_new_intersection(Map* self);

// --- Getters ---

Lane* map_get_lane(Map* self, LaneId id);
Road* map_get_road(Map* self, RoadId id);
Road* map_get_road_by_lane_id(Map* self, LaneId id);
Intersection* map_get_intersection(Map* self, IntersectionId id);
Intersection* map_get_intersection_by_lane_id(Map* self, LaneId id);
Lane* map_get_lane_by_name(Map* self, const char* name);
Road* map_get_road_by_name(Map* self, const char* name);
Intersection* map_get_intersection_by_name(Map* self, const char* name);

// Returns pointer to array of all lanes
Lane* map_get_lanes(Map* self);

// Returns number of lanes currently in the map
int map_get_num_lanes(const Map* self);

// Returns pointer to array of all roads
Road* map_get_roads(Map* self);

// Returns number of roads currently in the map
int map_get_num_roads(const Map* self);

// Returns pointer to array of all intersections
Intersection* map_get_intersections(Map* self);

// Returns number of intersections currently in the map
int map_get_num_intersections(const Map* self);


// Prints the full map information to a file or stdout
void map_print(Map* self, FILE* file);
