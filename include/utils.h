#pragma once

#include <stdbool.h>
#include <stdint.h>  // for uint32_t


//
// Math Utilities
//

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

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

// Creates dimensions from width and length.
Dimensions dimensions_create(double width, double length);


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

// Returns the intersection point of two line segments, if they intersect. Returns true if they intersect, false otherwise.
bool line_segments_intersect(LineSegment line1, LineSegment line2, Coordinates* intersection_point);


//
// Typed units
//

typedef double Meters;                      // distance
typedef double Seconds;                     // time
typedef double MetersPerSecond;             // speed
typedef double MetersPerSecondSquared;      // acceleration
typedef double Radians;                     // angle in radians
typedef double Degrees;                     // angle in degrees
typedef double Kilograms;                   // mass
typedef double Liters;                      // volume of fluid
typedef double Watts;                       // power
typedef double Newtons;                     // force
typedef double Joules;                      // energy
typedef double MetersSquared;               // area
typedef double MetersCubed;                 // volume
typedef double KgPerMetersCubed;            // density
typedef double JoulesPerLiter;              // energy density
typedef double LitersPerSecond;             // flow rate

// Initializes a Meters value.
Meters meters(double value);

// Initializes a Seconds value.
Seconds seconds(double value);

// Initializes a MetersPerSecond value.
MetersPerSecond mps(double value);

// Initializes a MetersPerSecondSquared value.
MetersPerSecondSquared mpss(double value);

// Initializes a Radians value.
Radians radians(double value);

// Initializes a Degrees value.
Degrees degrees(double value);

// Initializes a Kilograms value.
Kilograms kilograms(double value);

// Initializes a Liters value.
Liters liters(double value);

// Initializes a Watts value.
Watts watts(double value);

// Initializes a Newtons value.
Newtons newtons(double value);

// Initializes a Joules value.
Joules joules(double value);

// Initializes a MetersSquared value.
MetersSquared meters_squared(double value);

// Initializes a MetersCubed value.
MetersCubed meters_cubed(double value);

// Initializes a KgPerMetersCubed value.
KgPerMetersCubed kg_per_meters_cubed(double value);

// Initializes a JoulesPerLiter value.
JoulesPerLiter joules_per_liter(double value);

// Initializes a LitersPerSecond value.
LitersPerSecond liters_per_second(double value);


// Angles:

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


// Returns the sum of two angles in radians in [0, 2π).
Radians angles_add(Radians theta1, Radians theta2);

// Returns the difference between two angles in radians in [0, 2π).
Radians angles_sub(Radians theta1, Radians theta2);

// Normalizes an angle to the range [0, 2π).
Radians angle_normalize(Radians theta);

// Returns the quadrant of a given angle in radians.
Quadrant angle_get_quadrant(Radians theta);

// Returns the unit vector corresponding to an angle in radians.
Vec2D angle_to_unit_vector(Radians theta);

// constructs an angle in radians from degrees.
Radians from_degrees(Degrees theta);

// converts an angle in radians to degrees.
Degrees to_degrees(Radians theta);


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
// Volume conversion functions
// 

// Converts gallons (US) to liters.
Liters from_gallons(double value);

// Converts liters to gallons (US).
double to_gallons(Liters l);


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
// Vec 3D
///

// Represents a 3D vector or point in space.
struct Vec3D {
    double x;
    double y;
    double z;
};
typedef struct Vec3D Vec3D;

// Creates a 3D vector with given x, y, and z.
Vec3D vec3d_create(double x, double y, double z);

// Adds two 3D vectors.
Vec3D vec3d_add(Vec3D v1, Vec3D v2);

// Subtracts second 3D vector from first.
Vec3D vec3d_sub(Vec3D v1, Vec3D v2);

// Multiplies a 3D vector by a scalar.
Vec3D vec3d_scale(Vec3D v, double scalar);

// Divides a 3D vector by a scalar.
Vec3D vec3d_div(Vec3D v, double scalar);

// Returns the dot product of two 3D vectors.
double vec3d_dot(Vec3D v1, Vec3D v2);

// Returns the cross product of two 3D vectors.
Vec3D vec3d_cross(Vec3D v1, Vec3D v2);

// Returns the magnitude (length) of a 3D vector.
double vec3d_magnitude(Vec3D v);

// Returns the unit (normalized) 3D vector.
Vec3D vec3d_normalize(Vec3D v);

// Returns the distance between two 3D vectors (points).
double vec3d_distance(Vec3D v1, Vec3D v2);

// Returns the midpoint between two 3D vectors (points).
Vec3D vec3d_midpoint(Vec3D v1, Vec3D v2);

// Returns true if 3D vectors are approximately equal (within epsilon).
bool vec3d_approx_equal(Vec3D a, Vec3D b, double epsilon);


typedef Vec3D Coordinates3D;

// Creates 3D coordinates from x, y, and z values.
Coordinates3D coordinates3d_create(double x, double y, double z);

typedef Vec3D Dimensions3D;

// Creates 3D dimensions from width, length, and height. (as seen from a top-down view, height is the Z dimension)
Dimensions3D dimensions3d_create(double width, double length, double height); // assume map is in the X-Y plane, Z is altitude


// 3D Quad

typedef struct Quad3D {
    Coordinates3D corners[4]; // such that corners[i] to corners[(i+1)%4] forms an edge
} Quad3D;

//
// 3D Line segments
//

// Represents a 3D line segment between two points.
typedef struct LineSegment3D {
    Coordinates3D start;
    Coordinates3D end;
} LineSegment3D;

// Creates a 3D line segment from start and end points.
LineSegment3D line_segment3d_create(Coordinates3D start, Coordinates3D end);

// Creates a 3D line segment centered at a point, pointing in a direction, with specified length.
LineSegment3D line_segment3d_from_center(Coordinates3D center, Vec3D direction, double length);

// Returns the unit direction vector of a 3D line segment.
Vec3D line_segment3d_unit_vector(LineSegment3D line);


// Returns the intersection point of a 3D line segment and a Quad3D, if they intersect. Returns true if they intersect, false otherwise.
bool line_segment3d_intersect_quad(LineSegment3D line, Quad3D quad, Coordinates3D* intersection_point);


//
// Misc utilities
//

// Checks if an object is present in an array of object pointers (null-terminated).
bool object_array_contains_object(void** objects_array, const void* object);