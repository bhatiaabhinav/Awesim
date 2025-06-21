// Contents from include/utils.h



//
// Math Utilities
//



// Returns true if the difference between a and b is within epsilon.
bool approxeq(double a, double b, double epsilon);

void seed_rng(uint32_t seed);

// Returns a random 32-bit unsigned integer using a linear congruential generator (LCG).
uint32_t lcg_rand();

// Returns the current state of the LCG as a 32-bit unsigned integer.
uint32_t lcg_get_state();

// Returns a random double in the range [0.0, 1.0).
double rand_0_to_1(void);

// Returns a random double in the range [min, max).
double rand_uniform(double min, double max);

// Returns a random integer in the inclusive range [min, max].
int rand_int_range(int min, int max);

// Returns a random integer in the range [0, max). Assumes max > 0.
int rand_int(int max);

// Returns the value clamped between min and max.
double fclamp(double value, double min, double max);

//
// System Utilities
//

// Returns system time in seconds since Unix epoch with microsecond precision.
double get_sys_time_seconds(void);

// Sleeps for the specified number of milliseconds.
void sleep_ms(int milliseconds);

//
// 2D Vector structure and operations
//

// Represents a 2D vector or point in space.
struct Vec2D {
    double x;
    double y;
};
typedef struct Vec2D Vec2D;

// Creates a 2D vector with given x and y.
Vec2D vec_create(double x, double y);

// Adds two vectors.
Vec2D vec_add(Vec2D v1, Vec2D v2);

// Subtracts second vector from first.
Vec2D vec_sub(Vec2D v1, Vec2D v2);

// Multiplies a vector by a scalar.
Vec2D vec_scale(Vec2D v, double scalar);

// Divides a vector by a scalar.
Vec2D vec_div(Vec2D v, double scalar);

// Returns the dot product of two vectors.
double vec_dot(Vec2D v1, Vec2D v2);

// Returns the magnitude (length) of a vector.
double vec_magnitude(Vec2D v);

// Returns the distance between two vectors (points).
double vec_distance(Vec2D v1, Vec2D v2);

// Returns the unit (normalized) vector.
Vec2D vec_normalize(Vec2D v);

// Rotates a vector by angle (in radians).
Vec2D vec_rotate(Vec2D v, double angle);

// Returns the midpoint between two vectors (points).
Vec2D vec_midpoint(Vec2D v1, Vec2D v2);

// Returns true if vectors are approximately equal (within epsilon).
bool vec_approx_equal(Vec2D a, Vec2D b, double epsilon);


//
// Coordinate and dimension aliases
//

// Represents a 2D coordinate (alias for Vec2D).
typedef Vec2D Coordinates;

// Creates coordinates from x and y values.
Coordinates coordinates_create(double x, double y);

// Represents 2D dimensions (alias for Vec2D).
typedef Vec2D Dimensions;

// Creates dimensions from width and height.
Dimensions dimensions_create(double width, double height);


//
// Line segments
//

// Represents a line segment between two points.
typedef struct LineSegment {
    Coordinates start;
    Coordinates end;
} LineSegment;

// Creates a line segment from start and end points.
LineSegment line_segment_create(Coordinates start, Coordinates end);

// Creates a line segment centered at a point, pointing in a direction, with specified length.
LineSegment line_segment_from_center(Vec2D center, Vec2D direction, double length);

// Returns the unit direction vector of a line segment.
Vec2D line_segment_unit_vector(LineSegment line);


//
// Typed units
//

typedef double Meters;
typedef double Seconds;
typedef double MetersPerSecond;
typedef double MetersPerSecondSquared;
typedef double Radians;

// Initializes a Meters value.
Meters meters(double value);

// Initializes a Seconds value.
Seconds seconds(double value);

// Initializes a MetersPerSecond value.
MetersPerSecond mps(double value);

// Initializes a MetersPerSecondSquared value.
MetersPerSecondSquared mpss(double value);


//
// Distance conversion functions
//

// Converts feet to meters.
Meters from_feet(double value);

// Converts meters to feet.
double to_feet(Meters m);

// Converts miles to meters.
Meters from_miles(double value);

// Converts meters to miles.
double to_miles(Meters m);

// Converts kilometers to meters.
Meters from_kilometers(double value);

// Converts meters to kilometers.
double to_kilometers(Meters m);

// Converts inches to meters.
Meters from_inches(double value);

// Converts meters to inches.
double to_inches(Meters m);

// Converts centimeters to meters.
Meters from_centimeters(double value);

// Converts meters to centimeters.
double to_centimeters(Meters m);

// Converts millimeters to meters.
Meters from_millimeters(double value);

// Converts meters to millimeters.
double to_millimeters(Meters m);


//
// Speed conversion functions
//

// Converts kilometers per hour to meters per second.
MetersPerSecond from_kmph(double value);

// Converts meters per second to kilometers per hour.
double to_kmph(MetersPerSecond mps);

// Converts miles per hour to meters per second.
MetersPerSecond from_mph(double value);

// Converts meters per second to miles per hour.
double to_mph(MetersPerSecond mps);

// Converts feet per second to meters per second.
MetersPerSecond from_fps(double value);

// Converts meters per second to feet per second.
double to_fps(MetersPerSecond mps);


//
// Directions and direction-based utilities
//

// Enum representing compass directions and rotation.
typedef enum {
    DIRECTION_CCW,    // Counter-clockwise
    DIRECTION_CW,     // Clockwise
    DIRECTION_EAST,
    DIRECTION_WEST,
    DIRECTION_NORTH,
    DIRECTION_SOUTH,
} Direction;

// Returns the unit vector for a direction (N, S, E, W only).
Vec2D direction_to_vector(Direction direction);

// Returns a unit vector perpendicular to a direction (CCW rotation).
Vec2D direction_perpendicular_vector(Direction direction);

// Returns the opposite of a given direction.
Direction direction_opposite(Direction direction);


//
// Quadrants
//

// Enum representing 2D Cartesian quadrants.
typedef enum {
    QUADRANT_TOP_RIGHT,
    QUADRANT_TOP_LEFT,
    QUADRANT_BOTTOM_LEFT,
    QUADRANT_BOTTOM_RIGHT,
} Quadrant;

// Contents from include/map.h


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

// Contents from include/car.h



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
    CarId id;                         // unique identifier for the car
    Dimensions dimensions;          // width and length of the car
    CarCapabilities capabilities;   // capabilities of the car
    CarPersonality preferences;     // Preference profile of the car.

    // state variables:
    LaneId lane_id;                   // current lane
    double lane_progress;          // progress in the lane as a fraction of the lane length.
    Meters lane_progress_meters;   // progress in the lane in meters. Should be set by the simulation engine. Is a product of lane_progress and lane length.
    int lane_rank;              // rank of the car in the lane, where 0 is the first car in the lane (the one closest to the end of the lane), and num_cars-1 is the last car in the lane (the one closest to the start of the lane).
    MetersPerSecond speed;              // current speed. Should be set by the simulation engine.
    double damage;                      // damage level of the car. 0.0 for no damage and 1.0 for total damage.

    // control variables:
    MetersPerSecondSquared acceleration;    // current acceleration. Should be determined by the car's decision-making logic when car_make_decision() is called.
    CarIndictor indicator_lane;                  // Lane change indicator. Should be set by the car's decision-making logic when car_make_decision() is called.
    CarIndictor indicator_turn;                  // Turn indicator. Should be set by the car's decision-making logic when car_make_decision() is called.
};
typedef struct Car Car;

typedef struct Simulation Simulation;


void car_init(Car* car, Dimensions dimensions, CarCapabilities capabilities, CarPersonality preferences);


// getters:
CarId car_get_id(const Car* self);
Dimensions car_get_dimensions(const Car* self);
Meters car_get_length(const Car* self); 
Meters car_get_width(const Car* self);
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

MetersPerSecondSquared car_get_acceleration(const Car* self);
CarIndictor car_get_indicator_turn(const Car* self);
CarIndictor car_get_indicator_lane(const Car* self);


// setters:

// Sets the lane of the car.
void car_set_lane(Car* self, const Lane* lane);
// Sets the lane progress of the car. The progress should be between 0.0 and 1.0. Will be clipped automatically.
void car_set_lane_progress(Car* self, double progress, Meters progress_meters);
void car_set_lane_rank(Car* self, int rank);
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

// Contents from include/ai.h


// forward declarations
typedef struct Simulation Simulation;
Seconds sim_get_dt(const Simulation* self);
Map* sim_get_map(Simulation* self);

// Traffic light in context of an intented turn
typedef enum {
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_GREEN_YIELD,
    TRAFFIC_LIGHT_YELLOW,
    TRAFFIC_LIGHT_YELLOW_YIELD,
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_RED_YIELD,
    TRAFFIC_LIGHT_STOP,
    TRAFFIC_LIGHT_YIELD,
    TRAFFIC_LIGHT_NONE,
} TrafficLight;


struct BrakingDistance {
    Meters capable;            // Braking distance with full capable braking. = (v^2)/ (2 * dec_max)
    Meters preferred;          // Braking distance with preferred max braking = (v^2) / (2 * dec_preffered)
    Meters preferred_smooth;   // Braking distance with preferred max braking using PD controller
};
typedef struct BrakingDistance BrakingDistance;

struct SituationalAwareness {
    // Path
    bool is_approaching_dead_end;           // Does this lane not connect to any other lane?
    const Intersection* intersection;       // Relevant intersection, if any.
    bool is_on_intersection;                // Are we already on the intersection?
    bool is_approaching_intersection;       // Are we approaching an intersection?
    bool is_stopped_at_intersection;        // Have we stopped at intersection?
    const Road* road;                       // Current road
    const Lane* lane;                       // Current lane
    const Lane* lane_left;                  // if exits (in the same travel direction)
    const Lane* lane_right;                 // if exists
    Meters distance_to_intersection;        // Distance to the next intersection
    Meters distance_to_end_of_lane;         // Distance to the lane’s end
    bool is_close_to_end_of_lane;           // Is the distance to the lane’s end less than 10 m?
    bool is_on_leftmost_lane;             // Is this the leftmost lane?
    bool is_on_rightmost_lane;            // Is this the rightmost lane?
    double lane_progress;                  // Progress in the lane (0.0 to 1.0)
    Meters lane_progress_m;                // Progress in the lane in meters

    // Intent and Feasibility
    bool is_turn_possible[3];               // does this lane have a left, straight, right connection?  
    TrafficLight light_for_turn[3];         // Traffic light for the turn indicator (left, none, right)
    const Lane* lane_next_after_turn[3];    // Next lane for the turn indicator (left, none, right)
    bool is_lane_change_left_possible;      // Is there a lane to left that goes in the same direction?
    bool is_lane_change_right_possible;     // Is there a lane to right that goes in the same direction?
    bool is_merge_possible;                 // Is there a merge possible?
    bool is_exit_possible;                  // Is there an exit possible?
    bool is_on_entry_ramp;                  // Is the road an entry ramp? i.e., does the leftmost lane merge into something?
    bool is_exit_available;                 // Is there an exit ramp available? i.e., does the rightmost lane exit into something right now?
    bool is_exit_available_eventually;      // Is there an exit ramp available soon? i.e., does the rightmost lane exit into something eventually?
    const Lane* merges_into_lane;           // Lane that this lane merges into
    const Lane* exit_lane;                  // Lane that this lane exits into
    const Lane* lane_target_for_indicator[3];   // Potential target lane for the lane change indicator (left, none, right)
    

    // Surrounding Vehicles
    bool is_vehicle_ahead;                 // Is there a vehicle ahead of us (on the same lane)
    bool is_vehicle_behind;                // Is there a vehicle behind us (on the same lane)
    const Car* lead_vehicle;               // Vehicle ahead in the same lane
    const Car* following_vehicle;          // Vehicle behind in the same lane
    Meters distance_to_lead_vehicle;       // Distance to the lead vehicle
    Meters distance_to_following_vehicle;  // Distance to the following vehicle
    // const Car* vehicle_left;               // Closest vehicle in the left lane   // TODO
    // const Car* vehicle_right;              // Closest vehicle in the right lane  // TODO
    // Meters distance_to_vehicle_left;       // Distance to the left vehicle       // TODO
    // Meters distance_to_vehicle_right;      // Distance to the right vehicle      // TODO
    BrakingDistance braking_distance;
};
typedef struct SituationalAwareness SituationalAwareness;

// Set acceleration, indicator_turn, indicator_lane.
void npc_car_make_decisions(Car* self, Simulation* sim);
SituationalAwareness situational_awareness_build(const Car* car, Simulation* sim);

// sample a turn that is feasible from the current lane
CarIndictor turn_sample_possible(const SituationalAwareness* situation);

// sample a lane change intent that is feasible in terms of existence of such lanes
CarIndictor lane_change_sample_possible(const SituationalAwareness* situation);

bool car_is_lane_change_dangerous(const Car* car, Simulation* sim, const SituationalAwareness* situation, CarIndictor lane_change_indicator);

MetersPerSecondSquared car_compute_acceleration_chase_target(const Car* car, Meters position_target, MetersPerSecond speed_target, Meters position_target_overshoot_buffer, MetersPerSecond speed_limit);
MetersPerSecondSquared car_compute_acceleration_cruise(const Car* car, MetersPerSecond speed_target);
BrakingDistance car_compute_braking_distance(const Car* car);

bool car_should_yield_at_intersection(const Car* car, Simulation* sim, const SituationalAwareness* situation, CarIndictor turn_indicator);




// Represents the error in position and velocity relative to the equilibrium point.
// All values are expressed in the reference frame of the equilibrium.
struct ControlError {
    Meters position;           // Position error Δx (meters)
    MetersPerSecond speed;     // Velocity error Δv (meters/second)
};
typedef struct ControlError ControlError;



// Computes the acceleration required to drive both position and speed errors to zero
// using a critically damped proportional-derivative (PD) controller.
//
// Formula:
//     a = -2√k ⋅ Δv - k ⋅ Δx
//
// Characteristics:
// - Critically damped behavior avoids oscillation and overshoot.
// - Settling time (≈ within 2% of target): Δt ≈ 4 / √k
// - If (Δv/Δx)² = k, the system evolves exponentially:
//     x(t) = x₀ ⋅ exp(-√k ⋅ t)
//     v(t) = v₀ ⋅ exp(-√k ⋅ t)
//     a(t) = -k ⋅ x₀ ⋅ exp(-√k ⋅ t) = -√k ⋅ v₀ ⋅ exp(-√k ⋅ t)
//
// - Behavior varies with ratio (Δv/Δx)²:
//     > k       → Overshooting likely (braking too late)
//     < k / 4   → Initial velocity error may increase (braking too early)
//     ≈ k / 2.5 → Minimizes jerk (acceleration derivative)
//     → k as t → ∞
MetersPerSecondSquared control_compute_pd_accel(ControlError error, double k);






// Computes acceleration to reduce speed error to zero using a velocity-proportional damping controller.
// Position error is ignored.
//
// Formula:
//     a = -√k ⋅ Δv
//
// Characteristics:
// - Settling time ≈ 4 / √k
// - Evolution:
//     v(t) = v₀ ⋅ exp(-√k ⋅ t)
//     a(t) = -√k ⋅ v₀ ⋅ exp(-√k ⋅ t)
MetersPerSecondSquared control_speed_compute_pd_accel(ControlError error, double k);





// Computes the constant acceleration required to bring both speed and position errors to zero,
// assuming uniform acceleration and linear decay in speed.
//
// Formula:
//     a = Δv² / (2 ⋅ Δx)
//
// Characteristics:
// - Stopping time: Δt = -2 ⋅ Δx / Δv
//
// Preconditions:
// - Δv and Δx must have opposite signs (i.e., the object must be moving toward the equilibrium):
//     - Δv > 0 and Δx < 0 (approaching from the left)
//     - Δv < 0 and Δx > 0 (approaching from the right)
// - If this condition is not met, the function throws error.
MetersPerSecondSquared control_compute_const_accel(ControlError error);





// Computes the acceleration required to eliminate position and speed errors using
// a critically damped PD controller, with stiffness k chosen based on dynamic limits.
// This function is meant for "braking" i.e., has a precondition that the system is moving towards equilibrium i.e., Δv ⋅ Δx < 0
//
// The stiffness is selected so that the system reaches max acceleration when the
// speed error is v₀ = max_speed / 4.
//     k = (max_accel / v₀)²
//
// Usually, you should choose max acceleration and max_speed based on user comfort, not road speed limit. max_speed = 60 mph is a good choice.
//
// Behavior by scenario:
//
// CASE 1: Moving *towards* equilibrium (Δv ⋅ Δx < 0), and |Δv| <= v₀
// When (Δv/Δx)² = k:
//     - a = -2√k ⋅ Δv - k ⋅ Δx
//     - x(t) = x₀ ⋅ exp(-√k ⋅ t)
//     - v(t) = v₀ ⋅ exp(-√k ⋅ t)
//     - a(t) = -√k ⋅ v₀ ⋅ exp(-√k ⋅ t)
//     - Settling time ≈ Δt = 4 / √k = max_accel / max_speed, which is also the time to from max_speed to 0 with constant acceleration max_accel.
//     - Note that v₀ = -√k ⋅ x₀
//
// If (Δv/Δx)² > k and |Δv| <= v₀:
//     - Applies max_accel until (Δv/Δx)² = k, to prevent overshoot.
//
// If (Δv/Δx)² < k and |Δv| <= v₀:
//     - Delays braking and returns 0
//
// CASE 2: If |Δv| > v₀:
//     - Brakes with max_acc if Δx = (braking distance to v₀) + braking distance from there to 0.
//     - This combined braking distance = (Δv² - v₀²) / (2 ⋅ max_accel) + x₀.
//     - or, (Δv² - v₀²) / (4 ⋅ √k ⋅ v₀) + x₀.
//     - with |Δv| = max_speed = 4v₀, total settling time ≈ 1.75 ⋅ Δt, since time to go from 4v₀ to v₀ (= 0.75 ⋅ Δt) + time to go from v₀ to 0 (= Δt).
//
// As an example, a car that can brake from 60 to 0 mph in 4 seconds will take total 7 seconds to go from 60 to 0 mph with this controller as it will apply max_accel for 3 seconds (to go from 60mph to 15mph) and then exponentially decay for 4 seconds.
//
// CASE 3: Moving *away* from equilibrium (Δv ⋅ Δx ≥ 0):
// - Throws error
MetersPerSecondSquared control_compute_bounded_accel(ControlError error, MetersPerSecondSquared max_accel, MetersPerSecond max_speed);



// Computes the acceleration to reduce speed error to zero using damping proportional to velocity,
// with stiffness k automatically chosen to match a target dynamic response.
//
// The stiffness is computed as:
//     k = (4 ⋅ max_accel / max_speed)²
// so that when the speed error is max_speed / 4, the controller applies exactly max_accel and decays it.
// Usually, you should choose max acceleration and max_speed based on user comfort, not road speed limit. max_speed = 60 mph is a good choice.
//
// Formula:
//     a = -√k ⋅ Δv
//
// Characteristics:
// For |Δv| ≤ max_speed / 4:
//     - Acceleration is within bounds and follows exponential decay.
//     - Settling time (≈ within 2%): T ≈ max_speed / max_accel
//
// For |Δv| > max_speed / 4:
//     - Acceleration is clipped to ±max_accel until |Δv| ≤ max_speed / 4
//     - After that, exponential decay resumes
//     - With |Δv| = max_speed, total settling time ≈ 1.75 ⋅ T (0.75 ⋅ T with saturated accel, 1 ⋅ T exponential tail)
//
// As an example, when this control is used to target a particular speed (given speed error = -target speed), a car that goes from 0 to 60 mph in 4 seconds will take total 7 seconds to go from 0 to 60 mph with this controller as it will apply max_accel for 3 seconds (to go from 0mph to 45mph) and then exponentially decay the acc for 4 seconds.
// Behavior:
// - v(t) initially decreases linearly under max_accel clipping,
//   then follows exponential convergence once |Δv| ≤ max_speed / 4
MetersPerSecondSquared control_speed_compute_bounded_accel(ControlError error, MetersPerSecondSquared max_accel, MetersPerSecond max_speed);

// Contents from include/sim.h


#define MAX_CARS_IN_SIMULATION 1024 // Maximum number of cars in the simulation
#define MAX_NUM_HAZARDS_EACH_TYPE 1024 // Maximum number of hazards of each type

// Time-of-day phase start hours
#define MORNING_START 6
#define AFTERNOON_START 12
#define EVENING_START 16
#define NIGHT_START 20

typedef int Minutes;
typedef int Hours;
typedef int Days;

// Enum representing days of the week
typedef enum {
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    FRIDAY,
    SATURDAY,
    SUNDAY
} DayOfWeek;

extern const char* day_of_week_strings[];

// Structure for time representation as DD:HH:MM:SS 
typedef struct {
    Days days;
    Hours hours;
    Minutes minutes;
    Seconds seconds;
    DayOfWeek day_of_week; // 0 = Monday, 1 = Tuesday, ..., 6 = Sunday
} ClockReading;

// Constructs a ClockReading from components
ClockReading clock_reading(const Days days, const Hours hours, const Minutes minutes, const Seconds seconds);

// Converts seconds to ClockReading (throws error if < 0)
ClockReading clock_reading_from_seconds(const Seconds seconds);

// Converts a ClockReading to seconds
Seconds clock_reading_to_seconds(const ClockReading clock);

// Adds two ClockReadings
ClockReading clock_reading_add(const ClockReading clock1, const ClockReading clock2);

// Adds seconds to a ClockReading
ClockReading clock_reading_add_seconds(const ClockReading clock, const Seconds seconds);

// Weather conditions enum
typedef enum {
    WEATHER_SUNNY,
    WEATHER_CLOUDY,
    WEATHER_RAINY,
    WEATHER_FOGGY,
    WEATHER_SNOWY
} Weather;

extern const char* weather_strings[];

// Main simulation structure
typedef struct Simulation {
    Map map;       // Simulation map
    ClockReading initial_clock_reading; // Start clock
    Car cars[MAX_CARS_IN_SIMULATION];  // Car list; car[0] = agent
    int num_cars;   // Number of cars in simulation
    Seconds time;   // Time simulated so far
    Seconds dt;     // Simulation engine's time resolution for integration (in seconds)
    Weather weather;
    bool is_agent_enabled; // Whether car 0 is an agent (true) or an NPC (false). False by default. When true, car 0 is the agent car and must be controlled using car_set_acceleration, car_set_indicator_turn, and car_set_indicator_lane functions outside the sim loop.

    bool is_synchronized; // Whether the simulation is synchronized with wall time. If true, the simulation will run in real-time, simulating `simulation_speedup` seconds of simulation time per second of wall time. If false, the simulation will run as fast as possible without synchronization.
    double simulation_speedup; // How many seconds of simulation time to simulate per second of wall time. Default is 1.0, meaning real-time simulation.
    bool should_quit_when_rendering_window_closed; // Whether the simulation should quit when the rendering window is closed. If true, the `simulate` function will return when the rendering window is closed, allowing the simulation to exit gracefully. If false, the simulation will continue running even if the rendering window is closed, allowing for background processing or other tasks to continue.
    int render_socket; // Socket handle for render server connection
    bool is_connected_to_render_server; // Flag indicating connection status
} Simulation;

 // Allocate memory for a new Simulation instance. The memory is not initialized and contains garbage values.
Simulation* sim_malloc();
 // Free memory for a Simulation instance
void sim_free(Simulation* self);

// Initialize the simulation instance, setting up an empty map and default initial conditions (clock = 8:00 AM on monday, weather = sunny, dt = 0.02 seconds, no cars, no agent enabled, not synchronized).
void sim_init(Simulation* sim);

// Add a car to the simulation
Car* sim_get_new_car(Simulation* self);

// Function to connect to render server
bool sim_connect_to_render_server(Simulation* self, const char* server_ip, int port);
// Function to disconnect from render server
void sim_disconnect_from_render_server(Simulation* self);
// The render server may issue these commands to the simulation
typedef enum {
    COMMAND_NONE,
    COMMAND_QUIT,
    COMMAND_SIM_INCREASE_SPEED,
    COMMAND_SIM_DECREASE_SPEED,
} SimCommand;

// --- Getters ---
Map* sim_get_map(Simulation* self);
ClockReading sim_get_initial_clock_reading(Simulation* self);
int sim_get_num_cars(const Simulation* self);
Car* sim_get_car(Simulation* self, CarId id);
Seconds sim_get_time(Simulation* self);
Seconds sim_get_dt(const Simulation* self);
Weather sim_get_weather(Simulation* self);
ClockReading sim_get_clock_reading(Simulation* self);
DayOfWeek sim_get_day_of_week(const Simulation* self);
bool sim_is_agent_enabled(Simulation* self);
// returns NULL if the agent is not enabled
Car* sim_get_agent_car(Simulation* self);

// --- Setters ---
void sim_set_dt(Simulation* self, Seconds dt);
void sim_set_initial_clock_reading(Simulation* self, ClockReading clock);
void sim_set_weather(Simulation* self, Weather weather);
void sim_set_agent_enabled(Simulation* self, bool enabled);
void sim_set_synchronized(Simulation* self, bool is_synchronized, double simulation_speedup);
void sim_set_should_quit_when_rendering_window_closed(Simulation* self, bool should_quit);

// --- Time Queries ---
bool sim_is_weekend(Simulation* self);
bool sim_is_morning(Simulation* self);
bool sim_is_afternoon(Simulation* self);
bool sim_is_evening(Simulation* self);
bool sim_is_night(Simulation* self);

// --- Utility ---

// Returns sunlight intensity (0–1) based on time of day
double sunlight_intensity(Simulation* self);

// Integrate the simulation for a given duration in (simulated) seconds with dt resolution, causing sim_duration / dt transition updates. This will advance the simulation `time` by `sim_duration` seconds.
void sim_integrate(Simulation* self, Seconds sim_duration);

// Advance simulation by a duration in (simulated) seconds. If `self->is_synchronized` is true, the simulation will run in real-time, simulating `self->simulation_speedup` seconds of simulation time per second of wall time. If `is_synchronized` is false, the simulation will run as fast as possible without synchronization, making this function identical to `sim_integrate(self, sim_duration)`.
void simulate(Simulation* self, Seconds sim_duration);

// Advance simulation by one timestep (dt)
void sim_step(Simulation* self);


struct TrackIntersectionPoint {
    const Lane* lane1;  // Lane at the intersection point
    const Lane* lane2;  // Second lane at the intersection point
    double progress1;   // Progress along lane1
    double progress2;   // Progress along lane2
    bool exists;        // Whether the intersection point exists
};
typedef struct TrackIntersectionPoint TrackIntersectionPoint;

TrackIntersectionPoint track_intersection_check(const Lane* lane1, const Lane* lane2);
// TrackIntersectionPoint track_intersection_check_linear_linear(const LinearLane* lane1, const LinearLane* lane2);
// TrackIntersectionPoint track_intersection_check_linear_curved(const LinearLane* lane1, const QuarterArcLane* lane2);
// TrackIntersectionPoint track_intersection_check_curved_curved(const QuarterArcLane* lane1, const QuarterArcLane* lane2);


// struct CarCollision {
//     const Car* car1;    // First car involved in the collision
//     const Lane* lane1;  // Lane of the first car
//     double progress1;   // Progress along the lane of the first car
//     const Car* car2;    // Second car involved in the collision
//     const Lane* lane2;  // Lane of the second car
//     double progress2;   // Progress along the lane of the second car
//     Seconds time;       // Time of the collision
// };
// typedef struct CarCollision CarCollision;





// struct DeadEnd {
//     const Lane* lane; // Lane that is a dead end
// };
// typedef struct DeadEnd DeadEnd;

// struct Hazards {
//     const TrackIntersectionPoint* points[MAX_NUM_HAZARDS_EACH_TYPE]; // Array of intersection points
//     const DeadEnd* dead_ends[MAX_NUM_HAZARDS_EACH_TYPE]; // Array of dead ends
//     int num_intersection_points; // Number of intersection points
//     int num_dead_ends; // Number of dead ends
// };
// typedef struct Hazards Hazards;


// Hazards* hazards_create();
// Hazards* hazards_create_from_map(const Map* map);
// void hazards_free(Hazards* self);
// void hazards_add_intersection_point(Hazards* self, const TrackIntersectionPoint* point);
// void hazards_add_dead_end(Hazards* self, const DeadEnd* dead_end);

// Contents from include/awesim.h


void awesim_map_setup(Map* map, Meters city_width);

void awesim_setup(Simulation* sim, Meters city_width, int num_cars, Seconds dt, ClockReading initial_clock_reading, Weather weather);

