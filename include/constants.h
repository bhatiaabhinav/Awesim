#pragma once

#define MAX_CARS_IN_SIMULATION 512 // Maximum number of cars in the simulation

// Map related:
#define MAX_CARS_PER_LANE 64
#define MAX_NUM_ROADS 512
#define MAX_NUM_LANES_PER_ROAD 4
#define MAX_NUM_INTERSECTIONS 64
#define MAX_NUM_LANES_PER_INTERSECTION 32
#define MAX_NUM_LANES 2048
#define LANE_DEGRADATIONS_BUCKETS 4
#define GREEN_DURATION 13
#define YELLOW_DURATION 5
#define MINOR_RED_EXTENSION 2
#define ID_NULL -1
#define LANE_LENGTH_EPSILON 0.1
#define STOP_LINE_BUFFER_METERS 1.524       // Stop line distance from end of lane in meters
#define STOP_LINE_BUFFER_TOLERANCE_METERS 0.3048       // Stop line buffer tolerance in meters. stop line +/- this value is considered the "stop line area" for determining whether a car has crossed the stop line or not, for traffic violation purposes. This is to account for small perception errors and to avoid unfairly penalizing cars for minor overshooting of the stop line due to inertia or perception noise.
#define STOP_SPEED_THRESHOLD 1e-3           // 1 mm/s considered as stopped
#define ALMOST_STOP_SPEED_THRESHOLD 0.044704 // 0.1 mph in m/s considered as almost stopped
#define CREEP_SPEED 2.2352                // 5 mph in m/s

// Time-of-day phase start hours
#define MORNING_START 6
#define AFTERNOON_START 12
#define EVENING_START 16
#define NIGHT_START 20


// use it in the path planner:
#define MAX_NODES_PER_LANE 100    // Estimate
#define MAX_SOLUTION_CAPACITY 512
#define MAX_GRAPH_NODES (MAX_NUM_LANES * MAX_NODES_PER_LANE)


// driving assistant related:

#define AEB_ENGAGE_MIN_SPEED_MPS 0.89408                // AEB engages only if ego speed exceeds 0.89408 m/s (≈ 2 mph)
#define AEB_DISENGAGE_MAX_SPEED_MPS 0.0045              // AEB disengages when ego speed falls below 0.0045 m/s (≈ 0.01 mph)
#define AEB_ENGAGE_BEST_CASE_BRAKING_GAP_M 0.3048       // AEB engages if best-case gap under maximum braking is less than 0.3048 m (≈ 1 ft)
#define AEB_DISENGAGE_BEST_CASE_BRAKING_GAP_M 0.9144    // AEB disengages if best-case gap under maximum braking exceeds 0.9144 m (≈ 3 ft)
#define DRIVING_ASSISTANT_DEFAULT_TIME_HEADWAY 3.0                     // Default time headway to maintain from lead car in follow mode
#define DRIVING_ASSISTANT_DEFAULT_CAR_DISTANCE_BUFFER 3.0                // Default minimum distance to maintain from lead car in follow mode (in meters)

// #define NO_REACTION_DEPLAYS

#ifdef NO_REACTION_DEPLAYS
    #define HUMAN_TIME_TO_PRESS_FULL_PEDAL 0.0          // Time to go from 0 to full brake/gas pedal in seconds once the foot is on the pedal. Going from setting p0 to p1 takes time = |p1 - p0| * HUMAN_TIME_TO_PRESS_FULL_PEDAL
    #define HUMAN_TIME_TO_RELEASE_FULL_PEDAL 0.0        // Time to go from full brake/gas pedal to 0 in seconds once the foot is off the pedal. Going from setting p0 to p1 takes time = |p1 - p0| * HUMAN_TIME_TO_RELEASE_FULL_PEDAL
    #define HUMAN_AVERAGE_DELAY_SWITCHING_PEDALS 0.0      // Time to switch foot from brake to gas or vice versa, in seconds. During this time, neither pedal is pressed.
#else
    #define HUMAN_TIME_TO_PRESS_FULL_PEDAL 0.1          // Time to go from 0 to full brake/gas pedal in seconds once the foot is on the pedal. Going from setting p0 to p1 takes time = |p1 - p0| * HUMAN_TIME_TO_PRESS_FULL_PEDAL
    #define HUMAN_TIME_TO_RELEASE_FULL_PEDAL 0.1        // Time to go from full brake/gas pedal to 0 in seconds once the foot is off the pedal. Going from setting p0 to p1 takes time = |p1 - p0| * HUMAN_TIME_TO_RELEASE_FULL_PEDAL
    #define HUMAN_AVERAGE_DELAY_SWITCHING_PEDALS 0.1      // Time to switch foot from brake to gas or vice versa, in seconds. During this time, neither pedal is pressed.
#endif


#define TRAFFIC_VIOLATIONS_LOG_QUEUE_SIZE 10
#define COLLISIONS_MAX_PER_CAR 2

#define OVERSPEEDING_DETECTION_PROBABILITY 0.05 // Probability of getting caught for speeding in a given second. Will be mulitplied by dt to determine probability of getting caught in a given simulation step. This is a very rough parameter to simulate the deterrence effect of speeding tickets on speeding behavior, and can be tuned to achieve realistic law enforcement levels.
#define OVERSPEEDING_THRESHOLD_MPS 2.2352 // 5 mph in m/s. Speeding is defined as exceeding the speed limit by more than this threshold.

#define TAILGATING_MIN_SPEED_MPS 4.4704
#define TAILGATING_TIME_HEADWAY_THRESHOLD 1.5 // in seconds. Tailgating is defined as having less than this time headway to the lead car while traveling above TAILGATING_MIN_SPEED_MPS and not braking.
#define TAILGATING_DETECTION_PROBABILITY 0.05 // Probability of getting caught for tailgating in a given second while tailgating. This is a rough parameter to simulate the deterrence effect of tailgating tickets on tailgating behavior, and can be tuned to achieve realistic law enforcement levels.