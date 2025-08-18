#include "ai.h"
#include "utils.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MPH_60 26.8224
#define MPH_15 6.7056
#define MPH_15_SQUARED 44.96507136


MetersPerSecondSquared control_compute_pd_accel(ControlError error, double k) {
    return -2 * sqrt(k) * error.speed - k * error.position;
}


MetersPerSecondSquared control_speed_compute_pd_accel(ControlError error, double k) {
    return -sqrt(k) * error.speed;
}


MetersPerSecondSquared control_compute_const_accel(ControlError error) {
    // if (error.position * error.speed > 0) {
    //     fprintf(stderr, "Error: Moving away from equilibrium (Δv ⋅ Δx ≥ 0)\n");
    //     exit(EXIT_FAILURE);
    // }
    return error.speed * error.speed / (2 * error.position + 1e-9); // Avoid division by zero
}


MetersPerSecondSquared control_compute_bounded_accel(ControlError error, MetersPerSecondSquared max_accel, MetersPerSecond max_speed) {
    if (max_accel <= 0.0) {
        fprintf(stderr, "Error: max_accel must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (max_speed <= 0.0) {
        fprintf(stderr, "Error: max_speed must be positive\n");
        exit(EXIT_FAILURE);
    }
    // if (error.position * error.speed > 0) {
    //     fprintf(stderr, "Error: Moving away from equilibrium (Δv ⋅ Δx ≥ 0)\n");
    //     exit(EXIT_FAILURE);
    // }
    MetersPerSecond v_0 = max_speed / 4;
    double k = (max_accel / v_0) * (max_accel / v_0);
    // double ratio = (error.speed * error.speed) / (error.position * error.position + 1e-9);
    // Meters x_0 = v_0 / sqrt(k);
    // if (ratio < 1.5 * k) {
    //     return 0;
    // }
    // printf("ratio and k: %f, %f\n", ratio, k);
    // Meters braking_distance;
    // if (fabs(error.speed) <= v_0) {
    //     braking_distance = fabs(error.speed) / sqrt(k);
    // } else {
    //     braking_distance = (error.speed * error.speed - v_0 * v_0) / (2 * max_accel) + x_0;     // linear phase + PD phase
    // }

    // printf("braking_distance and position error and difference: %f, %f, %f\n", braking_distance, error.position, fabs(error.position) - braking_distance);
    // printf("v_0 and speed: %f, %f\n", v_0, fabs(error.speed));
    
    MetersPerSecondSquared accel;
    // if (braking_distance * 1.01 < fabs(error.position)) {    // that 1.1 is a buffer to start braking early
    // if (ratio < k / 4) {
    //     // printf("Too far. Not braking yet.\n");
    //     accel = 0;
    // } else {
    //     // printf("Applying PD control.\n");
    //     accel = control_compute_pd_accel(error, k);
    // }
    accel = control_compute_pd_accel(error, k);
    // printf("accel and max_accel: %f, %f\n", accel, max_accel);
    accel = fclamp(accel, -max_accel, max_accel);
    // accel = error.speed > 0 ? fclamp(accel, -max_accel, 0) : fclamp(accel, 0, max_accel);
    return accel;
}


MetersPerSecondSquared control_speed_compute_bounded_accel(ControlError error, MetersPerSecondSquared max_accel, MetersPerSecond max_speed) {
    if (max_accel <= 0.0) {
        fprintf(stderr, "Error: max_accel must be positive\n");
        exit(EXIT_FAILURE);
    }
    if (max_speed <= 0.0) {
        fprintf(stderr, "Error: max_speed must be positive\n");
        exit(EXIT_FAILURE);
    }
    double k = (4 * max_accel / max_speed) * (4 * max_accel / max_speed);
    MetersPerSecondSquared accel = control_speed_compute_pd_accel(error, k);
    accel = fclamp(accel, -max_accel, max_accel);
    return accel;
}


// Target may be moving or stationary
MetersPerSecondSquared car_compute_acceleration_chase_target(const Car* car, Meters position_target, MetersPerSecond speed_target, Meters position_target_overshoot_buffer, MetersPerSecond speed_limit, bool use_preferred_accel_profile) {
    Meters position_current = car_get_lane_progress_meters(car);
    MetersPerSecond speed_current = car_get_speed(car);
    LOG_TRACE("position_current: %f, speed_current: %f", position_current, speed_current);
    LOG_TRACE("position_target: %f, speed_target: %f", position_target, speed_target);
    Meters position_error = position_current - position_target;     // in target frame
    MetersPerSecond speed_error = speed_current - speed_target;     // in target frame
    LOG_TRACE("Position error: %f, Speed error: %f", position_error, speed_error);
    ControlError error = {position_error, speed_error};
    CarAccelProfile capable = car->capabilities.accel_profile;
    CarAccelProfile preferred = use_preferred_accel_profile ? car_accel_profile_get_effective(car->capabilities.accel_profile, car->preferences.acceleration_profile) : capable;
    MetersPerSecondSquared accel;
    const MetersPerSecondSquared acc_eps = 0.1;

    if (position_current <= position_target) { // target in front of us
        LOG_TRACE("Target in front of us");
        if (speed_current >= 0 ) {    // we are travelling forward            // already catching up. we need to slow down at some point
            MetersPerSecondSquared const_accel = control_compute_const_accel(error);
            if (const_accel < -preferred.max_deceleration) {    // can't brake in time with preferred profile. So, we brake max.
                LOG_TRACE("Can't brake in time with preferred profile. So, we brake max.");
                accel = fmax(const_accel, -capable.max_deceleration);
            } else {
                accel = control_compute_bounded_accel(error, preferred.max_deceleration, MPH_60);
                if (accel > 0) {   // we are still too far to brake if the braking controller is suggesting acceleration. Cruise to a higher speed using the acceleration controller.
                    LOG_TRACE("We are still too far to brake if the braking controller is suggesting acceleration. Cruise to a higher speed using the acceleration controller.");
                    accel = control_speed_compute_bounded_accel((ControlError){0, speed_current - speed_limit}, preferred.max_acceleration, MPH_60);
                    LOG_TRACE("Accel: %f", accel);
                } else {
                    LOG_TRACE("We are not too far to brake. Just brake smoothly.");
                }
            }
        } else {    // we are in reverse gear, even though the target is ahead. We need to bring speed to 0 using brakes as soon as possible within comfort.
            accel = control_speed_compute_bounded_accel((ControlError){0, speed_current}, capable.max_deceleration, MPH_60) + acc_eps;
        }
    } else {    // target already left behind. We are in overshot situation.
        LOG_TRACE("We are in overshot situation.");

        bool crashed = position_current > position_target + position_target_overshoot_buffer;
        bool crashed_and_still_moving_ahead = crashed && speed_current >= 0;

        if (crashed_and_still_moving_ahead) {
            LOG_TRACE("CRASHED! Breaking hard.");
            // exit(EXIT_FAILURE);
            accel = -capable.max_deceleration;
        } else { // not crashed yet or backing out of a crash, but target still left behind
            LOG_TRACE("Not crashed yet (or backing out of a crash), target position behind us");
            if (speed_current > 0) {    // and we are moving forward (and yet to crash)
                LOG_TRACE("And we are moving forward.");
                if (speed_current > speed_target) { // and moving forward at a faster pace than the target speed. Gap decreasing from the final crash point and increasing from where we should be.
                    MetersPerSecondSquared const_acc_to_prevent_crash = control_compute_const_accel((ControlError){position_current - (position_target + position_target_overshoot_buffer), speed_current});
                    if (const_acc_to_prevent_crash < -preferred.max_deceleration) { // our preferred profile can't avoid the crash
                        LOG_TRACE("Our preferred profile can't avoid the crash");
                        accel = fmax(const_acc_to_prevent_crash, -capable.max_deceleration);
                        if (const_acc_to_prevent_crash < -capable.max_deceleration) {
                            LOG_TRACE("OH NO!!!!!: Can't brake in time with capable profile. We are going to crash! Unless some lane change or some other event saves us.");
                        }
                    } else {
                        // we don't press full brakes if overshot only by little (e.g., we were trying to maintain a particular distance from the next vehicle, but overshot such that current distance is desired minus small eps), but allow full brakes if overshot by much.
                        LOG_TRACE("We don't press full brakes if overshot only by little, but allow full brakes if overshot by much.");
                        accel = control_speed_compute_bounded_accel(error, capable.max_deceleration, MPH_60) - acc_eps;
                    }
                } else {    // we are still moving forward but gap decreasing as our target (probably d distance behind the leading vehile when we are trying to tail) chases us. Just brake smoothly, unless chasing us too fast... like the next car suddenly sped up.
                    LOG_TRACE("We are still moving forward but gap decreasing as our target chases us.");
                    MetersPerSecondSquared const_acc = control_compute_const_accel(error);
                    if (const_acc > preferred.max_deceleration) {   // target chasing us too fast.
                        LOG_TRACE("Target chasing us too fast.");
                        accel = fmin(const_acc, capable.max_deceleration);
                    } else {
                        LOG_TRACE("Target not chasing us too fast.");
                        accel = control_compute_bounded_accel(error, preferred.max_deceleration, MPH_60);
                    }
                }
            } else {    // we are in reverse gear. We want to get back to the target as soon as possible.
                LOG_TRACE("We are in reverse gear. We want to get back to the target as soon as possible.");
                if (speed_current < speed_target) { // we are reversing fast enough. Need to break at some point.
                    LOG_TRACE("We are reversing fast enough. Need to break at some point.");
                    MetersPerSecondSquared const_acc = control_compute_const_accel(error);
                    if (const_acc > preferred.max_deceleration) {   // can't brake slowly
                        LOG_TRACE("Can't brake slowly");
                        accel = fmin(const_acc, capable.max_deceleration);
                    } else {    // can brake smoothly
                        accel = control_compute_bounded_accel(error, capable.max_deceleration, MPH_60);
                        if (accel < 0) {   // we are too far. Let us reverse faster.
                            LOG_TRACE("We are too far. Let us reverse faster.");
                            accel = control_speed_compute_bounded_accel((ControlError){0, speed_current + speed_limit}, preferred.max_acceleration_reverse_gear, MPH_60); 
                        } else {
                            LOG_TRACE("Braking smoothly");
                        }
                    }
                } else {
                    // were reversing too slow. Gap increasing as our target is also reversing. We might even crash if gap increases target + buffer. Just reverse full speed to catch up to the speed of the target, unless the error is too low.
                    LOG_TRACE("Were reversing too slow. Gap increasing as our target is also reversing.");
                    accel = control_speed_compute_bounded_accel((ControlError){0, speed_current - (speed_target - MPH_60)}, capable.max_acceleration_reverse_gear, MPH_60);
                }
            }
        }
    }
    return accel;
}

MetersPerSecondSquared car_compute_acceleration_adjust_speed(const Car* car, MetersPerSecond speed_target, bool use_preferred_accel_profile) {
    MetersPerSecond speed_current = car_get_speed(car);
    MetersPerSecondSquared accel;
    CarAccelProfile profile = use_preferred_accel_profile ? car_accel_profile_get_effective(car->capabilities.accel_profile, car->preferences.acceleration_profile) : car->capabilities.accel_profile;
    ControlError error = {0, speed_current - speed_target};
    MetersPerSecondSquared acc_eps = 0.1;
    if (speed_target > 0) {
        if (speed_target < speed_current) {
            // we are going too fast. Slow down.
            accel = control_speed_compute_bounded_accel(error, profile.max_deceleration, MPH_60);
        } else if (speed_current < speed_target) {
            if (speed_current < 0) {
                // first we need to stop if we are in reverse gear
                accel = control_speed_compute_bounded_accel((ControlError){0, speed_current}, profile.max_deceleration, MPH_60) + acc_eps;
            } else {
                // we are going too slow. Speed up.
                accel = control_speed_compute_bounded_accel(error, profile.max_acceleration, MPH_60);
            }
        } else {
            accel = 0; // already at target speed
        }
    } else if (speed_target < 0) {
        if (speed_target < speed_current) {
            if (speed_current < 0) {
                // continue to speed up in reverse gear
                accel = control_speed_compute_bounded_accel(error, profile.max_acceleration_reverse_gear, MPH_60);
            } else {
                // first, we need to stop if we are in forward gear
                accel = control_speed_compute_bounded_accel((ControlError){0, speed_current}, profile.max_deceleration, MPH_60) - acc_eps;
            }
        } else if (speed_current < speed_target) {
            // brake to slow down in reverse gear
            accel = control_speed_compute_bounded_accel(error, profile.max_deceleration, MPH_60);
        } else {
            accel = 0; // already at target speed
        }
    } else {
        if (speed_current == 0) {
            accel = 0; // already at target speed
        } else {
            accel = control_speed_compute_bounded_accel((ControlError){0, speed_current}, profile.max_deceleration, MPH_60); // brake to stop
        }
    }
    return accel;
}

MetersPerSecondSquared car_compute_acceleration_adjust_speed_linear(const Car* car, MetersPerSecond speed_target, bool use_preferred_accel_profile) {
    MetersPerSecond speed_current = car_get_speed(car);
    MetersPerSecondSquared accel;
    CarAccelProfile profile = use_preferred_accel_profile ? car_accel_profile_get_effective(car->capabilities.accel_profile, car->preferences.acceleration_profile) : car->capabilities.accel_profile;

    if (speed_target > 0) {
        if (speed_target < speed_current) {
            // we are going too fast. Slow down.
            accel = -profile.max_deceleration;
        } else if (speed_current < speed_target) {
            if (speed_current < 0) {
                // first we need to stop if we are in reverse gear
                accel = profile.max_deceleration; // braking while reversing = acceleration in positive direction
            } else {
                // we are going too slow. Speed up.
                accel = profile.max_acceleration;
            }
        } else {
            accel = 0; // already at target speed
        }
    } else if (speed_target < 0) {
        if (speed_target < speed_current) {
            if (speed_current < 0) {
                // continue to speed up in reverse gear
                accel = -profile.max_acceleration_reverse_gear;
            } else {
                // first, we need to stop if we are in forward gear
                accel = -profile.max_deceleration;
            }
        } else if (speed_current < speed_target) {
            // brake to slow down in reverse gear
            accel = profile.max_deceleration;
        } else {
            accel = 0; // already at target speed
        }
    } else {
        if (speed_current == 0) {
            accel = 0; // already at target speed
        } else {
            accel = speed_current > 0 ? -profile.max_deceleration : profile.max_deceleration; // brake to stop
        }
    }
    return accel;
}

MetersPerSecondSquared car_compute_acceleration_cruise(const Car* car, MetersPerSecond speed_target, bool use_preferred_accel_profile) {
    if (speed_target < 0) {
        fprintf(stderr, "Error: speed_target must be non-negative\n");
        exit(EXIT_FAILURE);
    }
    return car_compute_acceleration_adjust_speed(car, speed_target, use_preferred_accel_profile);
}

MetersPerSecondSquared car_compute_acceleration_stop(const Car* car, bool use_preferred_accel_profile) {
    return car_compute_acceleration_adjust_speed(car, 0, use_preferred_accel_profile);
}

MetersPerSecondSquared car_compute_acceleration_stop_max_brake(const Car* car, bool use_preferred_accel_profile) {
    CarAccelProfile profile = use_preferred_accel_profile ? car_accel_profile_get_effective(car->capabilities.accel_profile, car->preferences.acceleration_profile) : car->capabilities.accel_profile;
    MetersPerSecond speed = car_get_speed(car);
    return speed >= 0 ? -profile.max_deceleration : profile.max_deceleration;
}

MetersPerSecondSquared car_compute_acceleration_adaptive_cruise(const Car* car, const SituationalAwareness* situation, MetersPerSecond speed_target, MetersPerSecond lead_car_distance_in_seconds_target, Meters min_distance,  bool use_preferred_accel_profile) {
    if (speed_target < 0) {
        LOG_ERROR("Error: Car %d speed_target must be non-negative. Received: %f", car_get_id(car), speed_target);
        exit(EXIT_FAILURE);
    }
    const Car* lead_car = situation->lead_vehicle;
    // First, check if there is a lead car
    if (!lead_car) {
        // No lead car, just cruise at the target speed
        return car_compute_acceleration_adjust_speed(car, speed_target, use_preferred_accel_profile);
    } else {
        // There is a lead car, compute the acceleration to maintain a safe following distance
        MetersPerSecond car_speed = car_get_speed(car);
        Meters car_position = car_get_lane_progress_meters(car);

        Meters target_distance_from_next_car = fmax(car_speed, 0) * lead_car_distance_in_seconds_target + min_distance;
        Meters car_lengths_offset = (car_get_length(car) + car_get_length(situation->lead_vehicle)) / 2;
        target_distance_from_next_car += car_lengths_offset;
        Meters position_target = car_position + situation->distance_to_lead_vehicle - target_distance_from_next_car;
        Meters position_target_overshoot_buffer = target_distance_from_next_car - car_lengths_offset - from_feet(1.0); // let's call being within 1.0 feet of the next car as a crash.
        MetersPerSecond speed_at_target = car_get_speed(situation->lead_vehicle);
        return car_compute_acceleration_chase_target(car, position_target, speed_at_target, position_target_overshoot_buffer, speed_target, use_preferred_accel_profile);
    }
}


BrakingDistance car_compute_braking_distance(const Car* car) {
    BrakingDistance braking_distance;
    CarAccelProfile capable = car->capabilities.accel_profile;
    CarAccelProfile preferred = car_accel_profile_get_effective(car->capabilities.accel_profile, car->preferences.acceleration_profile);
    MetersPerSecond speed = car_get_speed(car);
    braking_distance.capable = speed * speed / (2 * capable.max_deceleration);
    braking_distance.preferred = speed * speed / (2 * preferred.max_deceleration);
    // for pid braking distance, when speed is greater than MPH_60/4, we apply preferred.max_deceleration until speed is less than MPH_60/4. Then exponential decay kicks in and v^2/x^2 ratio becomes k at all times. Therefore stopping distance = distance to reach speed MPH_60/4 + distance to stop from there.
    MetersPerSecond v_0 = MPH_15;
    MetersPerSecondSquared a_0 = preferred.max_deceleration;
    double k = (a_0 / v_0) * (a_0 / v_0);
    if (fabs(speed) <= v_0) {
        braking_distance.preferred_smooth = fabs(speed) / sqrt(k);
    } else {
        Meters x_0 = v_0 / sqrt(k);
        braking_distance.preferred_smooth = (speed * speed - v_0 * v_0) / (2 * a_0) + x_0;     // linear phase + PD phase
    }
    
    return braking_distance;
}