#pragma once

#include "utils.h"
#include "car.h"
#include <math.h>

// forward declarations
typedef struct Simulation Simulation;
Seconds sim_get_dt(const Simulation* self);
Map* sim_get_map(Simulation* self);
int sim_get_num_cars(const Simulation* self);

// Traffic light in context of an intented turn
typedef enum {
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_GREEN_YIELD,
    TRAFFIC_LIGHT_YELLOW,
    TRAFFIC_LIGHT_YELLOW_YIELD,
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_RED_YIELD,
    TRAFFIC_LIGHT_STOP_FCFS,
    TRAFFIC_LIGHT_YIELD,
    TRAFFIC_LIGHT_NONE,
} TrafficLight;


struct BrakingDistance {
    Meters capable;            // Braking distance with full capable braking. = (v^2)/ (2 * dec_max)
    Meters preferred;          // Braking distance with preferred max braking = (v^2) / (2 * dec_preffered)
    Meters preferred_smooth;   // Braking distance with preferred max braking using PD controller
};
typedef struct BrakingDistance BrakingDistance;


// A vehicle near the ego vehicle, along with its distance, time to collision, and time headway.
struct NearbyVehicle {
    const Car* car;                 // Pointer to the nearby vehicle (NULL if none).
    Meters distance;                // bumper-to-bumper distance to the vehicle in meters.
                                    // Positive value generally; for absent vehicles (car == NULL), set to INFINITY. For 'colliding', vehicles, set to center-to-center distance and may be negative if the vehicle is behind the ego
    
    Seconds time_to_collision;      // Time to collision (TTC) in seconds, considering relative speed and acceleration.
                                    // Computed with bumper-to-bumper adjustment: subtract (ego_half_length + other_half_length) from distance.
                                    // Formula (constant accel assumption):
                                    //   Let rel_speed = other_speed - ego_speed (positive if closing).
                                    //   Let rel_accel = other_accel - ego_accel (positive if closing faster).
                                    //   If rel_accel != 0: Solve quadratic t = [-rel_speed + sqrt(rel_speed² + 2 * rel_accel * adj_distance)] / rel_accel
                                    //     (take smallest positive root; use -root if discriminant positive but no +root).
                                    //   If rel_accel == 0: t = adj_distance / rel_speed (if rel_speed > 0).
                                    // Special cases:
                                    //   - If no collision path (rel_speed <= 0 and rel_accel <= 0, or no positive real roots): INFINITY (safe, no threat).
                                    //   - If already colliding/overlapping (adj_distance <= 0): 0 (imminent/ongoing).
                                    //   - Negative values not used; clamp to 0 for passed/receding cases.
                                    // Informs reactive decisions: e.g., brake if TTC < 2s; feeds into RL reward (penalize low TTC).
    
    Seconds time_headway;           // Time-based safe spacing metric in seconds (proactive headway).
                                    // Direction-specific:
                                    //   - For ahead/after_next_turn (ahead): bumper_distance / ego_speed (if ego_speed > 0; else INFINITY).
                                    //     Models "3-second rule" for following; low value (<3s) means too close—slow down.
                                    //   - For behind/incoming_from_previous_turn (behind): bumper_distance / other_speed (if other_speed > 0; else INFINITY).
                                    //     Models tailgating risk; low value (<2s) means rear vehicle closing fast—accelerate or yield.
                                    // Special cases:
                                    //   - If no vehicle (car == NULL) or speed <= 0 (no risk): INFINITY.
                                    //   - If colliding/overlapping (in colliding, adj_distance <= 0): 0 (no safe gap, collision state).
                                    // Informs proactive spacing: e.g., maintain 3s headway; penalize <3s in RL reward or NPC logic.
};
typedef struct NearbyVehicle NearbyVehicle;

// Struct for vehicles near the ego, aiding lane/turn/merge decisions.
// Arrays indexed by CarIndicator: [0] left, [1] straight, [2] right.
// Fields are NULL/invalid if no vehicle (car == NULL; distances/TTC/headway = INFINITY).
// "ahead/behind" are closest non-colliding; "colliding" overlaps (TTC=0, time_headway=0).
// Populate in sim update loop; use for RL inputs (state features) and NPC behaviors (rule-based or learned).
struct NearbyVehicles {
    NearbyVehicle ahead[3];                 // Vehicle ahead in target lane for change/merge. If none, closest in next lane (straight outgoing).
    NearbyVehicle colliding[3];             // Colliding vehicle during lane change (for indicator_none, it is the currently colliding vehicle). There may be multiple colliding vehicles, so the one furthest ahead is used. // TODO: is there a better way to handle this?
    NearbyVehicle behind[3];                // Vehicle behind in target lane for change/merge. If none, closest in prev lane (straight incoming).
    NearbyVehicle after_next_turn[3];              // First vehicle encountered on outgoing path after turn. Populated only if ego vehicle is the frontmost vehicle in the lane (and therefore, about to make a turn decision).
    NearbyVehicle incoming_from_previous_turn[3];  // Leading vehicle turning onto ego's lane. Populated only if ego vehicle is the last vehicle in ego's lane (and therefore, one of these vehicles will be the first to turn into ego's lane and tail the ego vehicle).
    NearbyVehicle turning_into_same_lane[3 * 3 * 3];       // When on intersection, vehicles from other lanes turning into the same lane as ego vehicle. That lane may have 3 incoming connections (middle array), so we record the behind/colliding/ahead vehicles (outermost array) in each of the 3 connections (once we project ego vehicle to those lanes based on distance to the beginning of the outgoing lane of the intersection (where all the lanes merge)). When ego is approaching an intersection, then the "same lane" will be the one we will be turning into post leaving intersection. We can turn in three directions, so the innermost array corresponds to that.
};
typedef struct NearbyVehicles NearbyVehicles;


// Useful for deep learning, where we want to flatten the NearbyVehicles struct
struct NearbyVehiclesFlattened {
    CarId car_ids[42];  // Array to hold extracted car IDs
    Meters distances[42]; // Corresponding bumper-to-bumper distances to the cars
    Seconds time_to_collisions[42]; // Corresponding time to collisions
    Seconds time_headways[42]; // Corresponding time headways
    int count; // Number of cars extracted
};

typedef struct NearbyVehiclesFlattened NearbyVehiclesFlattened;

void nearby_vehicles_flatten(NearbyVehicles* nearby_vehicles, NearbyVehiclesFlattened* flattened, bool include_outgoing_and_incoming_lanes);
CarId nearby_vehicles_flattened_get_car_id(const NearbyVehiclesFlattened* flattened, int index);
Meters nearby_vehicles_flattened_get_distance(const NearbyVehiclesFlattened* flattened, int index);
Seconds nearby_vehicles_flattened_get_time_to_collision(const NearbyVehiclesFlattened* flattened, int index);
Seconds nearby_vehicles_flattened_get_time_headway(const NearbyVehiclesFlattened* flattened, int index);
int nearby_vehicles_flattened_get_count(const NearbyVehiclesFlattened* flattened);


struct RGB {
    uint8_t r;         // Red channel (0-255)
    uint8_t g;         // Green channel (0-255)
    uint8_t b;         // Blue channel (0-255)
};
typedef struct RGB RGB;

RGB rgb(uint8_t r, uint8_t g, uint8_t b);
extern const RGB RGB_WHITE;
extern const RGB RGB_BLACK;
extern const RGB RGB_RED;
extern const RGB RGB_GREEN;
extern const RGB RGB_BLUE;
extern const RGB RGB_YELLOW;
extern const RGB RGB_ORANGE;
extern const RGB RGB_GRAY;
extern const RGB RGB_CYAN;
extern const RGB RGB_MAGENTA;

// based on Tesla's HW3/4 car camera setup
typedef enum {
    CAR_CAMERA_NARROW_FORWARD = 0,          // FOV = 35°, Range = 250 m. Location: near rear view mirror (Dropped in HW4, but useful if images are low-res)
    CAR_CAMERA_MAIN_FORWARD = 1,            // FOV = 50°, Range = 150 m. Location: near rear view mirror
    CAR_CAMERA_WIDE_FORWARD = 2,            // FOV = 120°, Range = 60 m. Location: near rear view mirror
    CAR_CAMERA_SIDE_LEFT_FORWARDVIEW = 3,   // FOV = 90°, Range = 80 m.  Location: B-pillar left
    CAR_CAMERA_SIDE_RIGHT_FORWARDVIEW = 4,  // FOV = 90°, Range = 80 m.  Location: B-pillar right
    CAR_CAMERA_SIDE_LEFT_REARVIEW = 5,      // FOV = 75°, Range = 100 m. Location: fender left
    CAR_CAMERA_SIDE_RIGHT_REARVIEW = 6,     // FOV = 75°, Range = 100 m. Location: fender right
    CAR_CAMERA_REARVIEW = 7,                // FOV = 140°, Range = 60 m. Location: above rear license plate
    CAR_CAMERA_FRONT_BUMPER = 8,            // FOV = 150°, Range = 20 m. Location: front bumper (New in HW4)
    CAR_CAMERA_NUM_TYPES = 9                // Number of camera types
} CarCameraType;

struct RGBCamera {
    Coordinates position; // Position of the camera
    Meters z_altitude;    // Altitude of the camera (from ground level)
    Radians orientation;  // Orientation of the camera
    int width;           // Width of the frame
    int height;          // Height of the frame. Determines the vertical fov when combined with width and fov.
    Radians fov;         // Field of view in radians (horizontal direction, i.e., along the sim's XY plane (map plane))
    Meters max_distance;   // Maximum distance the camera can see (in meters)
    uint8_t* data;       // Pointer to the RGB data in HWC format, linearized
    int aa_level;        // Level of anti-aliasing (e.g., 2 for 2x SSAA)
    Car* attached_car;  // If the camera is attached to a car, pointer to that car (NULL if not attached). Once attached to a car, the camera's position and orientation are set automatically based on the car's state when capturing.
    CarCameraType attached_car_camera_type; // If the camera is attached to a car, type of the car camera. Determines the relative position and orientation of the camera w.r.t. the car and its fov and max_distance.
    
    // Z-buffer (internal use)
    float* z_buffer;
    int z_buffer_size;

    // Pre-computed background (sky + ground)
    uint8_t* background_buffer;

    // Rendering buffer (internal use, for SSAA or double buffering)
    uint8_t* render_buffer;
    int render_width;
    int render_height;
};
typedef struct RGBCamera RGBCamera;

// Set the anti-aliasing level for the camera. This will reallocate the internal rendering buffer.
void rgbcam_set_aa_level(RGBCamera* camera, int level);

// Display for showing numerical information. The info is arrange in a grid, with number of rows and columns determined by square root of number of infos. Each info entry is represented as a horizontal bar that dims from full intensity to low intensity based on the value of the info (0.0 to 1.0).
struct InfosDisplay {
    int width;
    int height;
    int num_infos;
    uint8_t* data;          // RGB data in HWC format
    double* info_data;      // Array of info values. Length = num_infos
    RGB* info_colors;       // Array of colors for each info. Length = num_infos. Defaults to white for all infos.
};
typedef struct InfosDisplay InfosDisplay;

InfosDisplay* infos_display_malloc(int width, int height, int num_infos);
void infos_display_set_info(InfosDisplay* display, int info_index, double value);
void infos_display_set_info_color(InfosDisplay* display, int info_index, RGB color);
double infos_display_get_info(const InfosDisplay* display, int info_index);
void infos_display_free(InfosDisplay* display);
void infos_display_clear(InfosDisplay* display);
void infos_display_render(InfosDisplay* display);

typedef struct PathPlanner PathPlanner; // forward declaration for MiniMap

struct MiniMap {
    int width;
    int height;
    uint8_t* data; // RGB data in HWC format
    uint8_t* static_data; // Pre-rendered layout with full intensity
    
    // Map extent
    Coordinates min_coord;
    Coordinates max_coord;
    double scale_x; // pixels per meter
    double scale_y; // pixels per meter
    
    CarId marked_car_id;
    Coordinates marked_landmarks[8];
    RGB marked_landmark_colors[8];
    PathPlanner* marked_path;
    PathPlanner* debug_planner; // For visualizing the decision graph
};
typedef struct MiniMap MiniMap;

MiniMap* minimap_malloc(int width, int height, Map* map);
void minimap_free(MiniMap* minimap);
void minimap_render(MiniMap* minimap, Simulation* sim);

struct Lidar {
    Coordinates position; // Position of the LiDAR
    Radians orientation;  // Orientation of the LiDAR
    int num_points;      // Number of points in the LiDAR frame
    Radians fov;         // Field of view in radians
    Meters max_depth;    // Maximum depth range of the LiDAR (in meters)
    double* data;        // Pointer to the depth data (in meters)
};
typedef struct Lidar Lidar;

RGBCamera* rgbcam_malloc(Coordinates position, Meters z_altitude, Radians orientation, int width, int height, Radians fov, Meters max_distance);
Lidar* lidar_malloc(Coordinates position, Radians orientation, int num_points, Radians fov, Meters max_depth);
void rgbcam_free(RGBCamera* frame);
void lidar_free(Lidar* frame);
void rgbcam_clear(RGBCamera* frame);
void lidar_clear(Lidar* frame);
uint8_t rgbcam_get_value_at(const RGBCamera* frame, int channel, int row, int col);
double lidar_get_value_at(const Lidar* frame, int index);
void rgbcam_set_value_at(RGBCamera* frame, int channel, int row, int col, uint8_t value);
void lidar_set_value_at(Lidar* frame, int index, double value);
RGB rgbcam_get_pixel_at(const RGBCamera* frame, int row, int col);
void rgbcam_set_pixel_at(RGBCamera* frame, int row, int col, const RGB pixel);
void rgbcam_copy(const RGBCamera* src, RGBCamera* dest);
void lidar_copy(const Lidar* src, Lidar* dest);
void rgbcam_capture(RGBCamera* frame, Simulation* sim);
void rgbcam_capture_exclude_objects(RGBCamera* frame, Simulation* sim, void** exclude_objects);
void rgbcam_attach_to_car(RGBCamera* camera, Car* car, CarCameraType camera_type);
void lidar_capture(Lidar* frame, Simulation* sim);
void lidar_capture_exclude_objects(Lidar* frame, Simulation* sim, void** exclude_objects);


struct SituationalAwareness {
    bool is_valid;                          // Is this situational awareness valid? If false, all other fields should be ignored.
    Seconds timestamp;                     // Timestamp of the situational awareness

    // Path
    bool is_approaching_dead_end;           // Does this lane not connect to any other lane?
    const Intersection* intersection;       // Relevant intersection, if any.
    bool is_on_intersection;                // Are we already on the intersection?
    bool is_an_intersection_upcoming;       // Are we approaching an intersection at the end of this road?
    bool is_stopped_at_intersection;        // Have we stopped at an intersection (defined as an intersection upcoming AND approaching end of lane AND speed < 0.1 mph)?
    const Road* road;                       // Current road
    const Lane* lane;                       // Current lane
    const Lane* lane_left;                  // if exits (in the same travel direction)
    const Lane* lane_right;                 // if exists
    Meters distance_to_intersection;        // Distance to the next intersection
    Meters distance_to_end_of_lane;         // Distance to the lane’s end
    Meters distance_to_end_of_lane_from_leading_edge;         // Distance to the lane’s end from the leading edge of the car
    bool is_approaching_end_of_lane;        // Is the distance from the car's leading edge to the lane’s end less than 50 m?
    bool is_at_end_of_lane;                 // Is the distance from the car's leading edge to the lane’s end less than 4 m?
    bool is_on_leftmost_lane;               // Is this the leftmost lane?
    bool is_on_rightmost_lane;              // Is this the rightmost lane?
    double lane_progress;                   // Progress in the lane (0.0 to 1.0)
    Meters lane_progress_m;                 // Progress in the lane in meters

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
    bool is_my_turn_at_stop_sign_fcfs;     // If at a four-way stop sign, is it my turn to go based on FCFS?
    

    // Surrounding Vehicles
    bool is_vehicle_ahead;                 // Is there a vehicle ahead of us (see lead_vehicle)?
    bool is_vehicle_behind;                // Is there a vehicle behind us (see following_vehicle)?
    const Car* lead_vehicle;               // Vehicle ahead in the same lane, defined as the closest vehicle with progress greater than ours. If none in the same lane, then the closest vehicle ahead in the next lane (straight outgoing).
    const Car* following_vehicle;          // Vehicle behind in the same lane, defined as the closest vehicle with progress less than ours. If none in the same lane, then the closest vehicle behind in the prev lane (straight incoming).
    Meters distance_to_lead_vehicle;       // Distance to the lead vehicle (center-to-center distance)
    Meters distance_to_following_vehicle;  // Distance to the following vehicle (center-to-center distance)
    NearbyVehicles nearby_vehicles;
    BrakingDistance braking_distance;

    // Some historical summarization
    MetersPerSecond slowest_speed_yet_on_current_lane; // Slowest speed on the current lane so far
    Meters slowest_speed_position; // Position on the current lane where the slowest speed was observed
    MetersPerSecond slowest_speed_yet_at_end_of_lane; // Slowest speed at the end of the current lane so far
    bool full_stopped_yet_at_end_of_lane; // Have we properly stopped at the end of the lane yet? This is used to detect if stop signs are being respected. We consider it a full stop if slowest speed at the end of the lane is less than 0.1 mph
    bool rolling_stopped_yet_at_end_of_lane; // Have we yet done a "rolling stop" at the end of the lane? This is used to detect if stop signs are being almost respected. We consider it a rolling stop if slowest speed at the end of the lane is less than 5 mph
};
typedef struct SituationalAwareness SituationalAwareness;

Seconds sim_get_time(const Simulation* self); // Get the current simulation time. Forward declaration.
SituationalAwareness* sim_get_situational_awareness(Simulation* self, CarId id);    // Forward declaration
void situational_awareness_build(Simulation* sim, CarId car_id);

// Set acceleration, indicator_turn, indicator_lane.
void npc_car_make_decisions(Car* self, Simulation* sim);

// sample a turn that is feasible from the current lane
CarIndicator turn_sample_possible(const SituationalAwareness* situation);

// sample a lane change intent that is feasible in terms of existence of such lanes
CarIndicator lane_change_sample_possible(const SituationalAwareness* situation);

// If you were to change lanes (or merge or exit), what would be the hypothetical position on the target lane? Assuming the target_lane is either an adjacent lane or a merge/exit lane adjacent to the current lane.
Meters calculate_hypothetical_position_on_lane_change(const Car* car, const Lane* lane, const Lane* lane_target, Map* map);

bool car_is_lane_change_dangerous(Car* car, Simulation* sim, const SituationalAwareness* situation, CarIndicator lane_change_indicator, Seconds time_headway_threshold, Meters buffer);



// Compute the acceleration required to chase an *external* target position and speed. The function automatically accounts for the car's acceleration capabilities and preferences (is use_preferred_accel_profile is true). The overshoot buffer specifies the position error that the car will try its best to avoid by applying max capable braking. Use cases of this function include following a lead vehicle and coming to a stop at a target position.
MetersPerSecondSquared car_compute_acceleration_chase_target(const Car* car, Meters position_target, MetersPerSecond speed_target, Meters position_target_overshoot_buffer, MetersPerSecond speed_limit, bool use_preferred_accel_profile);

// Compute the acceleration required to adjust the car's speed to a target speed using a proportional controller, limited by either the car's capabilities (if use_preferred_accel_profile is false) or the car's preferred acceleration profile (if use_preferred_accel_profile is true).
MetersPerSecondSquared car_compute_acceleration_adjust_speed(const Car* car, MetersPerSecond speed_target, bool use_preferred_accel_profile);

// Applies constant acceleration to adjust the car's speed linearly to a target speed, with acceleration limited by either the car's capabilities (if use_preferred_accel_profile is false) or the car's preferred acceleration profile (if use_preferred_accel_profile is true).
MetersPerSecondSquared car_compute_acceleration_adjust_speed_linear(const Car* car, MetersPerSecond speed_target, bool use_preferred_accel_profile);

// Alias for car_compute_acceleration_adjust_speed with a non-negative speed_target.
MetersPerSecondSquared car_compute_acceleration_cruise(const Car* car, MetersPerSecond speed_target, bool use_preferred_accel_profile);

// Alias for car_compute_acceleration_adjust_speed with speed_target = 0.
MetersPerSecondSquared car_compute_acceleration_stop(const Car* car, bool use_preferred_accel_profile);

// Applies max linear braking with magnitude = car's preferred max braking (if use_preferred_accel_profile is true) or car's max capable braking (if use_preferred_accel_profile is false).
MetersPerSecondSquared car_compute_acceleration_stop_max_brake(const Car* car, bool use_preferred_accel_profile);

// Computes the acceleration required to maintain a safe following distance (in seconds to collision) from a lead vehicle by matching its speed, as long as it does not exceed speed_target. If there is no vehicle ahead, it returns the acceleration required to maintain speed speed_target. The car's preferred acceleration profile is used if use_preferred_accel_profile is true, otherwise the car's capabilities are used. Regardless, full capabilities are utilized for crash avoidance.
MetersPerSecondSquared car_compute_acceleration_adaptive_cruise(const Car* car, const SituationalAwareness* situation, MetersPerSecond speed_target, MetersPerSecond lead_car_distance_target_in_seconds, Meters min_distance, bool use_preferred_accel_profile);

BrakingDistance car_compute_braking_distance(const Car* car);

bool car_has_something_to_yield_to_at_intersection(const Car* car, Simulation* sim, const SituationalAwareness* situation, CarIndicator turn_indicator);




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

#define AEB_ENGAGE_MIN_SPEED_MPS 0.89408                // AEB engages only if ego speed exceeds 0.89408 m/s (≈ 2 mph)
#define AEB_DISENGAGE_MAX_SPEED_MPS 0.0045              // AEB disengages when ego speed falls below 0.0045 m/s (≈ 0.01 mph)
#define AEB_ENGAGE_BEST_CASE_BRAKING_GAP_M 0.9144       // AEB engages if best-case gap under maximum braking is less than 0.9144 m (≈ 3 ft)
#define AEB_DISENGAGE_BEST_CASE_BRAKING_GAP_M 1.8288    // AEB disengages if best-case gap under maximum braking exceeds 1.8288 m (≈ 6 ft)
#define DRIVING_ASSISTANT_BUFFER_M STOP_LINE_BUFFER_METERS                  // Buffer distance for driving assistant to stop at intersection (in meters) and for stopping behind a lead vehicle (if it exists and is stopped)

typedef struct DrivingAssistant {
    Seconds last_configured_at;         // Timestamp of the driving assistant's last settings update

    // Acceleration based on target speed:
    MetersPerSecond speed_target;       // Determines target speed. Can be of opposite sign.

    // Requested intents to determine steering:
    CarIndicator merge_intent;           // For switching lanes within a road, or merge/exit a highway. Set to None automatically once the assistant issues merge command.
    CarIndicator turn_intent;            // For which lane to take when approaching intersections. Set to None automatically once the assistant issues turn command.

    Seconds thw;                        // Time headway distance to the next vehicle (if exists). Also, the time headway for determining merge safety when merge assistance is active.
    bool should_stop_at_intersection;          // Whether to stop at the next intersection. The car will try to brake as smoothly as possible to stop DRIVING_ASSISTANT_BUFFER_M meter before the intersection. If stop_at_intersection is invoked later than the smoothest braking distance, braking intensity will be automatically adjusted to stop DRIVING_ASSISTANT_BUFFER_M meter before the intersection. If it slightly overshoots over the buffer, it will back up. But if it is impossible to brake in time, the car will still brake hard even though it will overshoot the legal stop line. Once already overshot, this variable is ignored because there is no intersection ahead and the only way the speed still reduces is if the speed_target reduces. This variable also triggers stopping at dead ends.

    bool follow_assistance;         // when false, disables follow mode even if there is a lead vehicle, so that the car will always try to reach speed_target regardless of lead vehicle.
    // Safety assistants
    bool merge_assistance;          // Whether indicated lane change happens only when it is safe to merge, which is when post merging, the distance from the lead car and the following car is geq follow_mode_thw
    bool aeb_assistance;            // Auto-Emergency-Braking (AEB). Whether to apply max braking if our car is projected to be dangerously close, even after max braking, to the lead car (with lead's acceleration accounted). Precisely, it is triggered when this best case braking gap <= AEB_ENGAGE_BEST_CASE_BRAKING_GAP. Moreover, ego vehicle speed must be equal or above AEB_ENGAGE_MIN_SPEED.

    // Internal state variables
    bool aeb_in_progress;           // Disengages automatically once the best case braking gap > AEB_DISENGAGE_BEST_CASE_BRAKING_GAP, or the ego stops, or manually disengaged.
    bool aeb_manually_disengaged;   // Once true, will stay until auto-disengagement condition is met. See `aeb_in_progress`.
    
    // Other settings
    bool use_preferred_accel_profile; // In control modes other than AEB, whether to use the car's preferred acceleration profile (if true) to limit acceleration, or use the car's full acceleration and braking capabilities (if false). Preferred profile can be set in car->preferences.
    bool use_linear_speed_control;  // In control speed-only mode (i.e., non-PD, non-follow mode), whether to apply constant acceleration to adjust speed linearly to the target speed, with acceleration limited by either the car's capabilities (if use_preferred_accel_profile is false) or the car's preferred acceleration profile (if use_preferred_accel_profile is true).
} DrivingAssistant;

// Set the car's control variables. Returns false if there is an error, true otherwise.
bool driving_assistant_control_car(DrivingAssistant* das, Car* car, Simulation* sim);

// To update internal state of the driving assistant, this is called at the end of each simulation step.
void driving_assistant_post_sim_step(DrivingAssistant* das, Car* car, Simulation* sim);

// Clear the driving assistant's parameters to default values.
bool driving_assistant_reset_settings(DrivingAssistant* das, Car* car);


// Getters

Seconds driving_assistant_get_last_configured_at(const DrivingAssistant* das);
MetersPerSecond driving_assistant_get_speed_target(const DrivingAssistant* das);
bool driving_assistant_get_should_stop_at_intersection(const DrivingAssistant* das);
CarIndicator driving_assistant_get_merge_intent(const DrivingAssistant* das);
CarIndicator driving_assistant_get_turn_intent(const DrivingAssistant* das);
Seconds driving_assistant_get_thw(const DrivingAssistant* das);
bool driving_assistant_get_follow_assistance(const DrivingAssistant* das);
bool driving_assistant_get_merge_assistance(const DrivingAssistant* das);
bool driving_assistant_get_aeb_assistance(const DrivingAssistant* das);
bool driving_assistant_get_aeb_in_progress(const DrivingAssistant* das);
bool driving_assistant_get_aeb_manually_disengaged(const DrivingAssistant* das);
bool driving_assistant_get_use_preferred_accel_profile(const DrivingAssistant* das);
bool driving_assistant_get_use_linear_speed_control(const DrivingAssistant* das);


// Setters (will update the timestamp)

void driving_assistant_configure_speed_target(DrivingAssistant* das, const Car* car, const Simulation* sim, MetersPerSecond speed_target);
void driving_assistant_configure_should_stop_at_intersection(DrivingAssistant* das, const Car* car, const Simulation* sim, bool should_stop_at_intersection);
void driving_assistant_configure_merge_intent(DrivingAssistant* das, const Car* car, const Simulation* sim, CarIndicator merge_intent);
void driving_assistant_configure_turn_intent(DrivingAssistant* das, const Car* car, const Simulation* sim, CarIndicator turn_intent);
void driving_assistant_configure_thw(DrivingAssistant* das, const Car* car, const Simulation* sim, Seconds thw);
void driving_assistant_configure_follow_assistance(DrivingAssistant* das, const Car* car, const Simulation* sim, bool follow_assistance);
void driving_assistant_configure_merge_assistance(DrivingAssistant* das, const Car* car, const Simulation* sim, bool merge_assistance);
void driving_assistant_configure_aeb_assistance(DrivingAssistant* das, const Car* car, const Simulation* sim, bool aeb_assistance);
void driving_assistant_configure_use_preferred_accel_profile(DrivingAssistant* das, const Car* car, const Simulation* sim, bool use_preferred_accel_profile);
void driving_assistant_configure_use_linear_speed_control(DrivingAssistant* das, const Car* car, const Simulation* sim, bool use_linear_speed_control);

// Manually disengage AEB. `aeb_manually_disengaged` flag will stay true and won't reset until standard disengagement conditions are also met
void driving_assistant_aeb_manually_disengage(DrivingAssistant* das, const Car* car, const Simulation* sim);


//
// Path Planning
// 




/* One outgoing edge in adjacency list */
typedef struct DG_Edge {
    int to;                 /* destination node */
    double weight;          /* non-negative */
    struct DG_Edge* next;   /* next edge in list */
} DG_Edge;

/* Simple directed graph: adj[u] is a linked list of outgoing edges */
typedef struct DirectedGraph {
    int num_nodes;   /* how many nodes are currently "in use" */
    int max_nodes;   /* fixed capacity */
    DG_Edge** adj;   /* size=max_nodes; only [0..num_nodes-1] are valid */
} DirectedGraph;

/* Make a graph with N nodes (0..N-1). Returns NULL on failure. */
DirectedGraph* dg_create(int num_nodes);

/* Add a node, returns its ID (0..max_nodes-1), or -1 if full/NULL */
int dg_add_node(DirectedGraph* g);

/* Free the graph and all edges. Safe on NULL. */
void dg_free(DirectedGraph* g);

/* Add directed edge u->v with weight=1. Returns 1 on success, 0 on fail. */
int dg_add_edge(DirectedGraph* g, int from, int to);

/* Add directed edge u->v with custom non-negative weight. */
int dg_add_edge_w(DirectedGraph* g, int from, int to, double weight);

/* Dijkstra shortest path from start to end.
   Writes nodes [start..end] into node_array_out.
    node_array_len_out is the capacity of node_array_out.
    cost_out is set to the total cost of the path.
   Returns:
     >0 path length
      0 no path
     -1 bad args / OOM
     -2 output buffer too small

*/
int dg_dijkstra_path(const DirectedGraph* g,
                     int start,
                     int end,
                     int* node_array_out,
                     int node_array_len_out,
                     double* cost_out);



// use it in the path planner:


#define MAX_NODES_PER_LANE 100    // Estimate
#define MAX_SOLUTION_CAPACITY 256
#define MAX_GRAPH_NODES (MAX_NUM_LANES * MAX_NODES_PER_LANE)

// Navigation Actions
typedef enum {
    NAV_STRAIGHT,
    NAV_TURN_LEFT,
    NAV_TURN_RIGHT,
    NAV_LANE_CHANGE_LEFT,
    NAV_LANE_CHANGE_RIGHT,
    NAV_MERGE_LEFT,
    NAV_MERGE_RIGHT,
    NAV_EXIT_LEFT,
    NAV_EXIT_RIGHT,
    NAV_NONE,
    NAV_NUM_ACTIONS // Number of actions
} NavAction;

typedef struct MapNode {
    LaneId lane_id;          // Lane ID
    Meters progress;         // Progress along the lane (in meters)
} MapNode;

// A simple path planner that finds the shortest path between two lanes on the map, using Dijkstra's algorithm.
// The cost of each lane is determined by its length if consider_speed_limits is false, or by the time to traverse the lane at its speed limit if consider_speed_limits is true. The overall cost of the path is the sum of the costs of the lanes in the path. Only forward motion is considered (i.e., reversals).
typedef struct PathPlanner {
    Map* map;                                       // Pointer to the map
    bool consider_speed_limits;                     // Whether to consider speed limits when computing lane costs. Should not be changed after creation.

    LaneId start_lane_id;                           // Starting lane for the path planning
    Meters start_progress;                          // Progress along the starting lane (in meters)
    LaneId end_lane_id;                             // Ending lane for the path planning
    Meters end_progress;                            // Progress along the ending lane (in meters)
    
    MapNode solution_nodes[MAX_SOLUTION_CAPACITY];                        // Array of nodes in the optimal path (including start and end)
    NavAction solution_actions[MAX_SOLUTION_CAPACITY - 1];                    // Array of navigation actions corresponding to each edge in the optimal path.
    int num_solution_actions;
    double optimal_cost;                            // Cost associated with the optimal path
    bool path_exists;                               // Whether a valid path exists
    

    // precomputed structures:

    DirectedGraph* decision_graph;                  // Directed graph representing the decision graph for path planning
    MapNode map_nodes[MAX_GRAPH_NODES];             // Array of map nodes corresponding to the node ids in the decision graph

    int solution_intermediate_node_ids[MAX_SOLUTION_CAPACITY - 2];              // Array of node ids in the optimal path excluding start and end

    int lane_id_to_node_ids_count[MAX_NUM_LANES];   // Count of node ids for each lane id.
    int lane_id_to_node_ids_list[MAX_NUM_LANES][MAX_NODES_PER_LANE]; // Mapping from lane id to list of node ids in the decision graph. -1 indicates end of list.
} PathPlanner;


PathPlanner* path_planner_create(Map* map, bool consider_speed_limits);     // allocates data and precomputes useful structures
void path_planner_free(PathPlanner* planner);   // does not free the map
void path_planner_compute_shortest_path(PathPlanner* planner, const Lane* start_lane, Meters start_progress, const Lane* end_lane, Meters end_progress);    // updates the planner with the optimal path information