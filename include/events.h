#pragma once

#include <utils.h>
#include "map.h"
#include "car.h"

typedef enum TrafficViolationType {
    VIOLATION_RED_LIGHT,             // Running a red light
    VIOLATION_STOP_SIGN,             // Failing to stop at a stop sign
    VIOLATION_SPEEDING,              // Exceeding the speed limit
    VIOLATION_RECKLESS_DRIVING,      // Driving in a dangerous or aggressive manner
    VIOLATION_IMPROPER_LANE_CHANGE,  // Changing lanes without signaling or checking
    VIOLATION_TAILGATING,            // Following too closely behind another vehicle
    VIOLATION_WRONG_WAY,             // Driving against the flow of traffic
    VIOLATION_IMPROPER_TURN,         // Making an illegal or unsafe turn
    VIOLATION_YIELD_FAILURE,         // Failing to yield right-of-way
    VIOLATION_PARKING_VIOLATION,     // Illegal parking (e.g., in a handicapped zone)
    VIOLATION_PEDESTRIAN_VIOLATION,  // Failing to yield to pedestrians
    VIOLATION_SCHOOL_ZONE,           // Speeding or other violation in a school zone
    VIOLATION_NO_TURN_SIGNAL,        // Failing to use turn signals for turns or lane changes
    VIOLATION_IGNORE_SIGN,           // Ignoring road signs (e.g., no turn on red, lane usage)
    VIOLATION_EMERGENCY_YIELD_FAILURE, // Failing to yield to emergency vehicles
    VIOLATION_ROLLING_STOP,          // Failing to come to a complete stop at a red light or stop sign
    VIOLATION_NUM_TYPES              // Sentinel value for the number of types
} TrafficViolationType;

// Function to get the description for a violation type
static inline const char *get_violation_description(TrafficViolationType type) {
    switch (type) {
        case VIOLATION_RED_LIGHT: return "Running a red light";
        case VIOLATION_STOP_SIGN: return "Failing to stop at a stop sign";
        case VIOLATION_SPEEDING: return "Exceeding the speed limit";
        case VIOLATION_RECKLESS_DRIVING: return "Reckless or aggressive driving";
        case VIOLATION_IMPROPER_LANE_CHANGE: return "Improper lane change without signaling or checking";
        case VIOLATION_TAILGATING: return "Following too closely (tailgating)";
        case VIOLATION_WRONG_WAY: return "Driving against the flow of traffic";
        case VIOLATION_IMPROPER_TURN: return "Making an illegal or unsafe turn";
        case VIOLATION_YIELD_FAILURE: return "Failing to yield right-of-way";
        case VIOLATION_PARKING_VIOLATION: return "Illegal parking (e.g., handicapped zone)";
        case VIOLATION_PEDESTRIAN_VIOLATION: return "Failing to yield to pedestrians";
        case VIOLATION_SCHOOL_ZONE: return "Violation in a school zone (e.g., speeding)";
        case VIOLATION_NO_TURN_SIGNAL: return "Failing to use turn signals for turns or lane changes";
        case VIOLATION_IGNORE_SIGN: return "Ignoring road signs (e.g., no turn on red, lane usage)";
        case VIOLATION_EMERGENCY_YIELD_FAILURE: return "Failing to yield to emergency vehicles";
        case VIOLATION_ROLLING_STOP: return "Failing to come to a complete stop at a red light or stop sign";
        default: return "Description unspecified";
    }
}


// Structure for violation event metadata
typedef struct ViolationMetaData {
    bool happened;                  // Whether this violation event happened. If false, this is just a dummy entry.
    TrafficViolationType type;      // Violation type
    Seconds timestamp;              // Time when the violation occurred
    Car* car;                       // Pointer to the car involved in the violation
    Lane* lane;                     // Pointer to the lane where the violation occurred
    double lane_progress;           // Progress along the lane where the violation occurred \in [0.0, 1.0]
    MetersPerSecond speed;          // Speed of the car at the time of violation

    // More specific metadata
    Intersection* intersection;     // Pointer to the intersection if applicable
    Car* other_car;                 // Pointer to another car involved in the violation (if applicable)
    Meters distance_to_other_car;   // Distance to the other car at the time of violation
    Lane* other_lane;               // Pointer to the other lane if applicable
} ViolationMetaData;

static const ViolationMetaData VIOLATION_NONE_METADATA = (ViolationMetaData){.happened = false};


// For a given car, stores an array of (potential) violations' metadata, where the entry metadata[type] gives the metadata for the corrsponding violtion type. If the violation did not happen, the entry will have happened = false. The list to be viewed as a sparse array of violations, where only the entries with happened = true are actual violations that occurred. The rest are just placeholders.
typedef struct PotentialViolations {
    Car* car;                       // Pointer to the car for which these violations are recorded
    ViolationMetaData metadata[VIOLATION_NUM_TYPES];
} PotentialViolations;

// Count the number of violations with happened = true
int violations_get_count(const PotentialViolations* violations);

PotentialViolations violations_find(const Car* car);