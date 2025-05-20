#pragma once

#include <stdbool.h>
#include "utils.h"

// Constants
#define MAX_CARS_PER_LANE 32
#define MAX_NUM_LANES 32
#define LANE_DEGRADATIONS_BUCKETS 5
#define GREEN_DURATION 13
#define YELLOW_DURATION 5
#define MINOR_RED_EXTENSION 2

// Forward Declarations
typedef struct Car Car;
typedef struct Road Road;
typedef struct Lane Lane;

// Degradations: Represents grip degradation across segments of a lane.
typedef struct {
    double values[LANE_DEGRADATIONS_BUCKETS];
} Degradations;

static const Degradations DEGRADATIONS_ZERO = {0};

// Enum for lane geometry types.
typedef enum {
    LINEAR_LANE,
    QUARTER_ARC_LANE
} LaneType;

// Lane structure representing a single traffic lane.
struct Lane {
    LaneType type;
    Direction direction;
    Meters width;
    MetersPerSecond speed_limit;
    double grip;
    Degradations degradations;

    const Lane* for_straight_connects_to;
    const Lane* for_left_connects_to;
    const Lane* for_right_connects_to;
    const Lane* adjacent_left;
    const Lane* adjacent_right;
    const Lane* merges_into;
    double merges_into_start;
    double merges_into_end;
    const Lane* exit_lane;
    double exit_lane_start;
    double exit_lane_end;

    const Car* cars[MAX_CARS_PER_LANE];
    int num_cars;

    const Road* road;

    Coordinates start_point;
    Coordinates end_point;
    Coordinates center;
    Meters length;
};

// Initialization
void lane_init(Lane* self, LaneType type, Direction direction, Meters width, MetersPerSecond speed_limit, double grip, Degradations degradations);
void lane_free(Lane* self);

// Lane Connections
void lane_connect_to_straight(Lane* self, const Lane* lane);
void lane_connect_to_left(Lane* self, const Lane* lane);
void lane_connect_to_right(Lane* self, const Lane* lane);
void lane_set_adjacents(Lane* self, const Lane* left, const Lane* right);
void lane_set_adjacent_left(Lane* self, const Lane* left);
void lane_set_adjacent_right(Lane* self, const Lane* right);
void lane_set_merges_into(Lane* self, const Lane* lane, double start, double end);
void lane_set_exit_lane(Lane* self, const Lane* lane, double start, double end);
void lane_set_speed_limit(Lane* self, MetersPerSecond speed_limit);
void lane_set_grip(Lane* self, double grip);
void lane_set_degradations(Lane* self, Degradations degradations);
void lane_set_road(Lane* self, const Road* road);

// Lane Vehicle Management
void lane_add_car(Lane* self, const Car* car);
void lane_remove_car(Lane* self, const Car* car);

// Lane Getters
LaneType lane_get_type(const Lane* self);
Direction lane_get_direction(const Lane* self);
Meters lane_get_width(const Lane* self);
MetersPerSecond lane_get_speed_limit(const Lane* self);
double lane_get_grip(const Lane* self);
Coordinates lane_get_start_point(const Lane* self);
Coordinates lane_get_end_point(const Lane* self);
Coordinates lane_get_center(const Lane* self);
Meters lane_get_length(const Lane* self);
int lane_get_num_cars(const Lane* self);
const Car* lane_get_car(const Lane* self, int index);
const Road* lane_get_road(const Lane* self);
Degradations lane_get_degradations(const Lane* self);
int lane_get_num_connections(const Lane* self);
const Lane* lane_get_first_connection(const Lane* self);
const Lane* lane_get_last_connection(const Lane* self);
bool lane_is_merge_available(const Lane* self);
bool lane_is_exit_lane_available(const Lane* self, double progress);
bool lane_is_exit_lane_eventually_available(const Lane* self, double progress);


// Linear Lane
typedef struct {
    Lane base;
} LinearLane;
LinearLane* linear_lane_create_from_center_dir_len(Coordinates center, Direction direction, Meters length, Meters width, MetersPerSecond speed_limit, double grip, Degradations degradations);
LinearLane* linear_lane_create_from_start_end(Coordinates start, Coordinates end, Meters width, MetersPerSecond speed_limit, double grip, Degradations degradations);
void linear_lane_free(LinearLane* self);

// Quarter Arc Lane
typedef struct {
    Lane base;
    Meters radius;
    Radians start_angle;
    Radians end_angle;
    Quadrant quadrant;
} QuarterArcLane;
QuarterArcLane* quarter_arc_lane_create_from_start_end(Coordinates start_point, Coordinates end_point, Direction direction, Meters width, MetersPerSecond speed_limit, double grip, Degradations degradations);
void quarter_arc_lane_free(QuarterArcLane* self);

// Road Types
typedef enum {
    STRAIGHT,
    TURN,
    INTERSECTION
} RoadType;

// Road Structure
struct Road {
    RoadType type;
    const Lane* lanes[MAX_NUM_LANES];
    int num_lanes; // 0 = leftmost, num_lanes-1 = rightmost
    MetersPerSecond speed_limit;
    double grip;
    Coordinates center;
};

// Frees the road and all its lanes.
void road_free(Road* self);

// Road Getters
RoadType road_get_type(const Road* self);
int road_get_num_lanes(const Road* self);
const Lane* road_get_lane(const Road* self, int index);
MetersPerSecond road_get_speed_limit(const Road* self);
double road_get_grip(const Road* self);
Coordinates road_get_center(const Road* self);
const Lane* road_get_leftmost_lane(const Road* self);
const Lane* road_get_rightmost_lane(const Road* self);
bool road_is_merge_available(const Road* self);
bool road_is_exit_road_available(const Road* self, double progress);
bool road_is_exit_road_eventually_available(const Road* self, double progress);

// Straight Road
typedef struct {
    Road base;
    Meters lane_width;
    Direction direction;
    Meters length;
    Meters width;
    Coordinates start_point;
    Coordinates end_point;
} StraightRoad;

StraightRoad* straight_road_create_from_center_dir_len(Coordinates center, Direction direction, Meters length, int num_lanes, Meters lane_width, MetersPerSecond speed_limit, double grip);
StraightRoad* straight_road_create_from_start_end(Coordinates start, Coordinates end, int num_lanes, Meters lane_width, MetersPerSecond speed_limit, double grip);
StraightRoad* straight_road_make_opposite(const StraightRoad* self);
StraightRoad* straight_road_make_opposite_and_set_adjacent_left(StraightRoad* self);
StraightRoad* straight_road_split_at_and_update_connections(StraightRoad* road, Meters position, Meters gap);
// Free the road and all its lanes.
void straight_road_free(StraightRoad* self);

// Turn
typedef struct {
    Road base;
    const StraightRoad* road_from;
    const StraightRoad* road_to;
    Direction direction;
    Coordinates start_point;
    Coordinates end_point;
} Turn;

Turn* turn_create_and_set_connections_and_adjacents(StraightRoad* road_from, StraightRoad* road_to, Direction direction, MetersPerSecond speed_limit, double grip, Turn* opposite_turn);
// Free the road and all its lanes.
void turn_free(Turn* self);

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

static const Seconds TRAFFIC_STATE_DURATIONS[NUM_TRAFFIC_CONTROL_STATES] = {
    GREEN_DURATION,
    YELLOW_DURATION,
    MINOR_RED_EXTENSION,
    GREEN_DURATION,
    YELLOW_DURATION,
    MINOR_RED_EXTENSION,
};

typedef struct {
    Road base;
    Dimensions dimensions;
    const StraightRoad* road_eastbound_from;
    const StraightRoad* road_eastbound_to;
    const StraightRoad* road_westbound_from;
    const StraightRoad* road_westbound_to;
    const StraightRoad* road_northbound_from;
    const StraightRoad* road_northbound_to;
    const StraightRoad* road_southbound_from;
    const StraightRoad* road_southbound_to;
    Meters turn_radius;
    bool left_lane_turns_left_only;
    bool right_lane_turns_right_only;
    IntersectionState state;
    Seconds countdown;
} Intersection;

Intersection* intersection_create_and_form_connections(
    StraightRoad* road_eastbound_from,
    StraightRoad* road_eastbound_to,
    StraightRoad* road_westbound_from,
    StraightRoad* road_westbound_to,
    StraightRoad* road_northbound_from,
    StraightRoad* road_northbound_to,
    StraightRoad* road_southbound_from,
    StraightRoad* road_southbound_to,
    bool is_four_way_stop,
    bool left_lane_turns_left_only,
    bool right_lane_turns_right_only,
    MetersPerSecond speed_limit,
    double grip
);
Intersection* intersection_create_from_crossing_roads_and_update_connections(
    StraightRoad* eastbound,
    StraightRoad* westbound,
    StraightRoad* northbound,
    StraightRoad* southbound,
    bool is_four_way_stop,
    Meters turn_radius,
    bool left_lane_turns_left_only,
    bool right_lane_turns_right_only,
    MetersPerSecond speed_limit,
    double grip
);
void intersection_update(Intersection* self, Seconds dt);
// Free the intersection and all its lanes.
void intersection_free(Intersection* self);
void print_traffic_state(const Intersection* intersection);
const Intersection* road_leads_to_intersection(const Road* road);
IntersectionState intersection_get_state(const Intersection* self);
Seconds intersection_get_countdown(const Intersection* self);
bool intersection_get_left_lane_turns_left_only(const Intersection* self);
bool intersection_get_right_lane_turns_right_only(const Intersection* self);
Meters intersection_get_turn_radius(const Intersection* self);
Dimensions intersection_get_dimensions(const Intersection* self);