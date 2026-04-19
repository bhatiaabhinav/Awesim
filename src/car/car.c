#include "utils.h"
#include "car.h"
#include "logging.h"
#include "math.h"
#include "sim.h"
#include "map.h"
#include "ai.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static void reset_stats(Car* car) {
    memset(&car->travel_stats, 0, sizeof(CarTravelStats));
}


void car_init(Car* car, Dimensions3D dimensions, CarCapabilities capabilities, Liters fuel_tank_capacity, CarPersonality preferences) {
    car->dimensions = dimensions;
    car->fuel_tank_capacity = fuel_tank_capacity;
    car->capabilities = capabilities;
    car->preferences = preferences;
    car->lane_id = ID_NULL;
    car->lane_progress = 0.0;
    car->lane_progress_meters = 0.0;
    car->cached_half_length = dimensions.y * 0.5;
    car->lane_rank = -1;
    car->speed = 0.0;
    car->damage = 0.0;
    car->fuel_level = fuel_tank_capacity; // Start with a full tank
    car->center = (Coordinates){0.0, 0.0};
    car->orientation = 0.0;
    car->true_speed_vector = (Vec2D){0.0, 0.0};
    car->true_acceleration_vector = (Vec2D){0.0, 0.0};
    for (int i = 0; i < 4; i++) {
        car->corners[i] = (Coordinates){0.0, 0.0};
    }
    car->acceleration = 0.0;
    car->indicator_turn = INDICATOR_NONE;
    car->indicator_lane = INDICATOR_NONE;
    car->request_indicated_lane = false;
    car->request_indicated_turn = false;
    car->auto_turn_off_indicators = true; // Default to true, can be changed later
    car->prev_lane_id = ID_NULL; // Initialize previous lane ID to NULL
    car->recent_forward_movement = 0.0; // Initialize recent forward movement to 0
    car->lowest_speed_in_stop_zone = __FLT_MAX__; // No stop zone visit yet
    car->lowest_speed_post_stop_line_before_lane_end = __FLT_MAX__; // No stop line crossing yet
    reset_stats(car);
}

void car_free(Car* self) {
    if (self) {
        free(self);
    }
}

CarId car_get_id(const Car* self) { return self->id; }
Dimensions3D car_get_dimensions(const Car* self) { return self->dimensions; }
Meters car_get_length(const Car* self) { return self->dimensions.y; }
Meters car_get_width(const Car* self) { return self->dimensions.x; }
Meters car_get_height(const Car* self) { return self->dimensions.z; }
Liters car_get_fuel_tank_capacity(const Car* self) { return self->fuel_tank_capacity; }
CarCapabilities car_get_capabilities(const Car* self) { return self->capabilities; }
CarPersonality car_get_preferences(const Car* self) { return self->preferences; }
LaneId car_get_lane_id(const Car* self) { return self->lane_id; }
Lane* car_get_lane(const Car* self, Map* map) {
    if (self->lane_id < 0 || self->lane_id >= map->num_lanes) {
        LOG_ERROR("Car (id=%d) is not on a valid lane (id=%d).", self->id, self->lane_id);
        return NULL;
    }
    return map_get_lane(map, self->lane_id);
}
double car_get_lane_progress(const Car* self) { return self->lane_progress; }
Meters car_get_lane_progress_meters(const Car* self) { return self->lane_progress_meters; }
int car_get_lane_rank(const Car* self) { return self->lane_rank; }
MetersPerSecond car_get_speed(const Car* self) { return self->speed; }
double car_get_damage(const Car* self) { return self->damage; }
Liters car_get_fuel_level(const Car* self) { return self->fuel_level; }
Coordinates car_get_center(const Car* self) { return self->center; }
Radians car_get_orientation(const Car* self) { return self->orientation; }
Coordinates* car_get_corners(const Car* self) { return (Coordinates*)self->corners; }
Coordinates car_get_corner_front_right(const Car* self) { return self->corners[0]; }
Coordinates car_get_corner_front_left(const Car* self) { return self->corners[1]; }
Coordinates car_get_corner_rear_left(const Car* self) { return self->corners[2]; }
Coordinates car_get_corner_rear_right(const Car* self) { return self->corners[3]; }

MetersPerSecondSquared car_get_acceleration(const Car* self) { return self->acceleration; }
CarIndicator car_get_indicator_turn(const Car* self) { return self->indicator_turn; }
CarIndicator car_get_indicator_lane(const Car* self) { return self->indicator_lane; }
bool car_get_request_indicated_lane(const Car* self) { return self->request_indicated_lane; }
bool car_get_request_indicated_turn(const Car* self) { return self->request_indicated_turn; }
bool car_get_auto_turn_off_indicators(const Car* self) { return self->auto_turn_off_indicators; }


void car_set_fuel_tank_capacity(Car* self, Liters capacity) {
    if (capacity <= 0.0) {
        LOG_WARN("Fuel tank capacity must be positive. You are trying to set it to %f. Ignoring.", capacity);
        return;
    }
    self->fuel_tank_capacity = capacity;
    // If the current fuel level exceeds the new capacity, clip it.
    if (self->fuel_level > capacity) {
        LOG_WARN("Current fuel level (%f liters) exceeds new fuel tank capacity (%f liters). Clipping fuel level.", self->fuel_level, capacity);
        self->fuel_level = capacity;
    }
}

void car_set_lane(Car* self, const Lane* lane) {
    self->prev_lane_id = self->lane_id; // Store the lane ID at the previous timestep
    self->lane_id = lane->id;
}

void car_set_lane_progress(Car* self, double progress, Meters progress_meters, Meters recent_forward_movement) {
    if (progress < 0.0 || progress > 1.0) {
        LOG_WARN("Car %d, lane %d : Lane progress must be between 0.0 and 1.0. You are trying to set it to %f. Clipping.", self->id, self->lane_id, progress);
    }
    progress = fclamp(progress, 0.0, 1.0);
    self->lane_progress = progress;
    if (progress_meters < 0.0) {
        progress_meters = 0.0;
    }
    self->lane_progress_meters = progress_meters;
    self->recent_forward_movement = recent_forward_movement;
}

void car_set_lane_rank(Car* self, int rank) {
    self->lane_rank = rank;
}

void car_set_speed(Car* self, MetersPerSecond speed) {
    if (speed > self->capabilities.top_speed) {
        LOG_WARN("Speed exceeds car's top speed. You are trying to set it to %f. Clipping.", speed);
        self->speed = self->capabilities.top_speed;
    } else {
        self->speed = speed;
    }
}

void car_set_acceleration(Car* self, MetersPerSecondSquared acceleration) {
    CarAccelProfile* profile = &self->capabilities.accel_profile;
    if (self->speed >= 0.0) {
        if (acceleration > profile->max_acceleration) acceleration = profile->max_acceleration;     // gas
        if (acceleration < -profile->max_deceleration) acceleration = -profile->max_deceleration;   // braking
    } else {    // reverse gear
        if (acceleration < -profile->max_acceleration_reverse_gear) acceleration = -profile->max_acceleration_reverse_gear;                                                     // gas with reverse gear
        if (acceleration > profile->max_deceleration) acceleration = profile->max_deceleration;     // braking
    }
    self->acceleration = acceleration;
}

void car_set_damage(Car* self, const double damage) {
    if (damage < 0.0 || damage > 1.0) {
        LOG_WARN("Damage must be between 0.0 and 1.0. You are trying to set it to %f. Clipping.", damage);
    }
    self->damage = fclamp(damage, 0.0, 1.0);
}

void car_set_fuel_level(Car* self, Liters fuel_level) {
    if (fuel_level < 0.0) {
        LOG_WARN("Fuel level cannot be negative. You are trying to set it to %f. Clipping.", fuel_level);
    }
    if (fuel_level > self->fuel_tank_capacity) {
        LOG_WARN("Fuel level cannot exceed fuel tank capacity (%f liters). You are trying to set it to %f. Clipping.", self->fuel_tank_capacity, fuel_level);
    }
    self->fuel_level = fclamp(fuel_level, 0.0, self->fuel_tank_capacity);
}

void car_set_indicator_turn(Car* self, CarIndicator indicator) {
    self->indicator_turn = indicator;
}

void car_set_indicator_lane(Car* self, CarIndicator indicator) {
    self->indicator_lane = indicator;
}

void car_set_request_indicated_lane(Car* self, bool request) {
    self->request_indicated_lane = request;
}
void car_set_request_indicated_turn(Car* self, bool request) {
    self->request_indicated_turn = request;
}

void car_set_indicator_turn_and_request(Car* self, CarIndicator indicator) {
    car_set_indicator_turn(self, indicator);
    if (indicator != INDICATOR_NONE) {
        car_set_request_indicated_turn(self, true);
    } else {
        car_set_request_indicated_turn(self, false);
    }
}

void car_set_indicator_lane_and_request(Car* self, CarIndicator indicator) {
    car_set_indicator_lane(self, indicator);
    if (indicator != INDICATOR_NONE) {
        car_set_request_indicated_lane(self, true);
    } else {
        car_set_request_indicated_lane(self, false);
    }
}

void car_set_auto_turn_off_indicators(Car* self, bool auto_turn_off) {
    self->auto_turn_off_indicators = auto_turn_off;
}


Meters car_get_recent_forward_movement(const Car* self) {
    return self->recent_forward_movement;
}

LaneId car_get_prev_lane_id(const Car* self) {
    return self->prev_lane_id;
}

void car_reset_all_control_variables(Car* self) {
    car_set_indicator_lane(self, INDICATOR_NONE);
    car_set_request_indicated_lane(self, false);
    car_set_indicator_turn(self, INDICATOR_NONE);
    car_set_request_indicated_turn(self, false);
    car_set_acceleration(self, 0);
}

static void car_geometry_from_given_position_speed_acc(Simulation* sim, Car* car, Meters lane_progress_meters, MetersPerSecond speed, MetersPerSecondSquared acceleration, Coordinates* out_center, Radians* out_orientation, Vec2D* out_speed_vector, Vec2D* out_acceleration_vector, Coordinates out_corners[4]) {
    Lane* lane = car_get_lane(car, sim_get_map(sim));
    double lane_progress = lane_progress_meters / lane->length;
    Coordinates center;
    Radians orientation;
    double cos_orient, sin_orient; // Computed once, reused for tangent + corners
    if (lane->type == LINEAR_LANE) {
        // Use cached orientation/cos/sin — no atan2 or trig needed
        orientation = lane->cached_orientation;
        cos_orient = lane->cached_cos_orientation;
        sin_orient = lane->cached_sin_orientation;
        Vec2D start = lane->start_point;
        Vec2D end = lane->end_point;
        Vec2D delta = vec_sub(end, start);
        center = vec_add(start, vec_scale(delta, lane_progress));
        if (out_center) *out_center = center;
        if (out_orientation) *out_orientation = orientation;
        
        // Linear Lane Vectors — reuse cached cos/sin
        if (out_speed_vector) {
            out_speed_vector->x = cos_orient * speed;
            out_speed_vector->y = sin_orient * speed;
        }
        if (out_acceleration_vector) {
            out_acceleration_vector->x = cos_orient * acceleration;
            out_acceleration_vector->y = sin_orient * acceleration;
        }
        
    } else if (lane->type == QUARTER_ARC_LANE) {
        // Use cached trig values for start/end angles
        double theta = lane->start_angle + lane_progress * (lane->end_angle - lane->start_angle);
        double cos_theta = cos(theta);
        double sin_theta = sin(theta);
        Vec2D theoretical_pos = {
            lane->center.x + lane->radius * cos_theta,
            lane->center.y + lane->radius * sin_theta
        };

        // Use precomputed theoretical start/end positions
        Vec2D offset_start = vec_sub(lane->cached_actual_start, lane->cached_theoretical_start);
        Vec2D offset_end = vec_sub(lane->cached_actual_end, lane->cached_theoretical_end);
        double p = lane_progress;
        Vec2D current_offset = {
            offset_start.x * (1.0 - p) + offset_end.x * p,
            offset_start.y * (1.0 - p) + offset_end.y * p
        };

        // Apply offset
        center.x = theoretical_pos.x + current_offset.x;
        center.y = theoretical_pos.y + current_offset.y;
        if (out_center) *out_center = center;

        // Orientation: use precomputed angular diffs (stored in cached_cos/sin_orientation)
        double diff_start = lane->cached_cos_orientation; // repurposed: angular offset at start
        double diff_end = lane->cached_sin_orientation;   // repurposed: angular offset at end

        // Theoretical orientation on the arc
        double dir_sign = (lane->direction == DIRECTION_CCW) ? 1.0 : -1.0;
        double theoretical_orient = theta + dir_sign * M_PI_2;
        double current_orient_offset = diff_start * (1.0 - p) + diff_end * p;
        orientation = angle_normalize(theoretical_orient + current_orient_offset);
        if (out_orientation) *out_orientation = orientation;

        // Compute cos/sin of orientation once for tangent, normal, and corners
        if (out_speed_vector || out_acceleration_vector || out_corners) {
            cos_orient = cos(orientation);
            sin_orient = sin(orientation);
        }

        // Speed and Acceleration Vectors
        if (out_speed_vector || out_acceleration_vector) {
            double omega_theoretical = (speed / lane->radius) * dir_sign;
            double omega_correction = (diff_end - diff_start) * (speed / lane->length);
            double omega_total = omega_theoretical + omega_correction;

            // 2. Speed Vector
            if (out_speed_vector) {
                double effective_speed_scalar = speed * (1.0 + (diff_end - diff_start) * (lane->radius / lane->length) * dir_sign);
                out_speed_vector->x = cos_orient * effective_speed_scalar;
                out_speed_vector->y = sin_orient * effective_speed_scalar;
            }
            
            // 3. Acceleration Vector
            if (out_acceleration_vector) {
                double centripetal = speed * omega_total;
                out_acceleration_vector->x = cos_orient * acceleration + (-sin_orient) * centripetal;
                out_acceleration_vector->y = sin_orient * acceleration + cos_orient * centripetal;
            }
        }
    } else {
        LOG_ERROR("Cannot update geometry for car %d: unknown lane type.", car->id);
        return;
    }

    if (out_corners) {
        double half_width = car->dimensions.x * 0.5;
        double half_length = car->dimensions.y * 0.5;
        // Inline rotation: rotated = (x*cos - y*sin, x*sin + y*cos) + center
        // Front-right (half_length, -half_width)
        out_corners[0].x = center.x + half_length * cos_orient - (-half_width) * sin_orient;
        out_corners[0].y = center.y + half_length * sin_orient + (-half_width) * cos_orient;
        // Front-left (half_length, half_width)
        out_corners[1].x = center.x + half_length * cos_orient - half_width * sin_orient;
        out_corners[1].y = center.y + half_length * sin_orient + half_width * cos_orient;
        // Rear-left (-half_length, half_width)
        out_corners[2].x = center.x + (-half_length) * cos_orient - half_width * sin_orient;
        out_corners[2].y = center.y + (-half_length) * sin_orient + half_width * cos_orient;
        // Rear-right (-half_length, -half_width)
        out_corners[3].x = center.x + (-half_length) * cos_orient - (-half_width) * sin_orient;
        out_corners[3].y = center.y + (-half_length) * sin_orient + (-half_width) * cos_orient;
    }
}


void car_update_geometry(Simulation* sim, Car* car) {
    car_geometry_from_given_position_speed_acc(sim, car, car->lane_progress_meters, car->speed, car->acceleration, &car->center, &car->orientation, &car->true_speed_vector, &car->true_acceleration_vector, car->corners);
}


Coordinates coords_compute_relative_to_car(const Car* self, Coordinates global_coords) {
    Meters dx = global_coords.x - self->center.x;
    Meters dy = global_coords.y - self->center.y;
    double cos_theta = cos(-self->orientation);
    double sin_theta = sin(-self->orientation);
    Coordinates rel_coords;
    rel_coords.x = dx * cos_theta - dy * sin_theta;
    rel_coords.y = dx * sin_theta + dy * cos_theta;
    return rel_coords;
}

Vec2D vec2d_rotate_to_car_frame(const Car* self, Vec2D global_vec) {
    double cos_theta = cos(-self->orientation);
    double sin_theta = sin(-self->orientation);
    Vec2D rel_vec;
    rel_vec.x = global_vec.x * cos_theta - global_vec.y * sin_theta;
    rel_vec.y = global_vec.x * sin_theta + global_vec.y * cos_theta;
    return rel_vec;
}

bool car_noisy_perceive_other(const Car* self, const Car* other, Simulation* sim, double* progress_out, Meters* progress_meters_out, MetersPerSecond* speed_out, MetersPerSecondSquared* acceleration_out, Meters* rel_distance_out, Radians* rel_orientation_out, Coordinates* rel_position_out, Vec2D* rel_speed_vector_out, Vec2D* rel_acceleration_vector_out, bool consider_blind_spot, double dropout_probability_multiplier, bool noisy) {
    if (!self || !other || !sim) {
        return false;
    }
    if (noisy) {
        double net_probability = sim->perception_noise_dropout_probability * dropout_probability_multiplier;
        if (net_probability > 1e-9 && rand_0_to_1() < net_probability) {
            return false; // Not perceived
        }
    }
    Map* map = sim_get_map(sim);  // cache once
    Lane* other_lane = car_get_lane(other, map);

    // Blind spot check — only fetch self_lane and adjacency when needed
    if (noisy && consider_blind_spot) {
        double base_blindspot_drop_probability = sim->perception_noise_blind_spot_dropout_base_probability * dropout_probability_multiplier;
        if (base_blindspot_drop_probability > 1e-9) {
            Lane* self_lane = car_get_lane(self, map);
            Lane* self_adjacent_left = lane_get_adjacent_left(self_lane, map);
            if (self_adjacent_left && self_adjacent_left->direction != self_lane->direction) {
                self_adjacent_left = NULL;
            }
            Lane* self_adjacent_right = lane_get_adjacent_right(self_lane, map);
            if (self_adjacent_right && self_adjacent_right->direction != self_lane->direction) {
                self_adjacent_right = NULL;
            }
            bool in_adjacent_lane = other_lane == self_adjacent_left || other_lane == self_adjacent_right;
            if (in_adjacent_lane) {
                Meters hypothetical_position = calculate_hypothetical_position_on_lane_change(self, self_lane, other_lane, map);
                Meters ego_len = car_get_length(self);
                Meters other_len = car_get_length(other);
                Meters bs_start_rel = -ego_len * 0.5 - 3.0;
                Meters bs_end_rel = ego_len / 4.0;
                Meters bs_len = bs_end_rel - bs_start_rel;
                Meters other_distance = other->lane_progress_meters - hypothetical_position;
                Meters other_start = other_distance - other_len / 2.0;
                Meters other_end = other_distance + other_len / 2.0;
                if (other_start >= bs_start_rel && other_end <= bs_end_rel) {
                    double size_ratio = other_len / bs_len;
                    double drop_probability = base_blindspot_drop_probability * (1.0 - size_ratio);
                    if (drop_probability > 1e-9 && rand_0_to_1() < drop_probability) {
                        return false;
                    }
                }
            }
        }
    }
    
    // Same-lane check via integer lane_id comparison — avoids needing car_get_lane(self) when blind spot is skipped
    bool same_lane = (self->lane_id == other->lane_id);

    // Only compute cosine_alignment and true_distance when noise parameters require them
    bool distance_noise_on = noisy && sim->perception_noise_distance_std_dev_percent > 1e-9;
    bool speed_noise_on = noisy && sim->perception_noise_speed_std_dev > 1e-9;

    Meters noisy_progress_meters;
    double cosine_alignment;
    if (distance_noise_on || speed_noise_on) {
        cosine_alignment = same_lane ? 1.0 : fabs(cos(self->orientation - other->orientation));
    } else {
        cosine_alignment = 0.0; // unused
    }

    if (distance_noise_on) {
        Meters true_distance = same_lane ? fabs(other->lane_progress_meters - self->lane_progress_meters) : vec_magnitude(vec_sub(other->center, self->center));
        double distance_noise_stddev = sim->perception_noise_distance_std_dev_percent * true_distance;
        distance_noise_stddev *= cosine_alignment;
        noisy_progress_meters = fclamp(rand_gaussian(other->lane_progress_meters, distance_noise_stddev), 0.0, other_lane->length);
    } else {
        noisy_progress_meters = other->lane_progress_meters;
    }
    if (progress_meters_out) *progress_meters_out = noisy_progress_meters;
    if (progress_out) *progress_out = noisy_progress_meters / other_lane->length;

    MetersPerSecond true_speed = other->speed;
    if (speed_noise_on) {
        MetersPerSecond speed_noise_std = cosine_alignment * sim->perception_noise_speed_std_dev * true_speed;
        MetersPerSecond noisy_speed = rand_gaussian(true_speed, speed_noise_std);
        if (speed_out) *speed_out = noisy_speed;
    } else {
        if (speed_out) *speed_out = true_speed;
    }

    MetersPerSecondSquared true_accel = other->acceleration;
    MetersPerSecondSquared noisy_accel = true_accel; // No noise for acceleration for now
    if (acceleration_out) *acceleration_out = noisy_accel;

    double still_need_rel_distance_out = rel_distance_out != NULL;
    if (still_need_rel_distance_out && same_lane) {
        *rel_distance_out = fabs(noisy_progress_meters - self->lane_progress_meters);
        still_need_rel_distance_out = false;
    }


    if (still_need_rel_distance_out || rel_orientation_out || rel_position_out || rel_speed_vector_out || rel_acceleration_vector_out) {
        Radians other_orientation_perceived;
        Coordinates other_center_perceived;
        Vec2D other_speed_vector_perceived;
        Vec2D other_acceleration_vector_perceived;
        MetersPerSecond noisy_speed_for_geom = speed_out ? *speed_out : true_speed;
        car_geometry_from_given_position_speed_acc(sim, (Car*)other, 
            noisy_progress_meters,
            noisy_speed_for_geom,
            noisy_accel,
            (rel_position_out || still_need_rel_distance_out) ? &other_center_perceived : NULL,
            (rel_orientation_out) ? &other_orientation_perceived : NULL,
            (rel_speed_vector_out) ? &other_speed_vector_perceived : NULL,
            rel_acceleration_vector_out ? &other_acceleration_vector_perceived : NULL,
            NULL);

        if (still_need_rel_distance_out) {
            *rel_distance_out = vec_magnitude(vec_sub(other_center_perceived, self->center));
        }
        if (rel_orientation_out) {
            *rel_orientation_out = angle_normalize(other_orientation_perceived - self->orientation);
        }
        if (rel_position_out) {
            *rel_position_out = coords_compute_relative_to_car(self, other_center_perceived);
        }
        if (rel_speed_vector_out) {
            *rel_speed_vector_out = vec2d_rotate_to_car_frame(self, other_speed_vector_perceived);
        }
        if (rel_acceleration_vector_out) {
            *rel_acceleration_vector_out = vec2d_rotate_to_car_frame(self, other_acceleration_vector_perceived);;
        }
    }



    return true;
}


bool car_is_colliding(const Car* car1, const Car* car2) {

    // very cheap filter: check if the distance between the centers is larger than 15 meters
    // Bus diagonal is ~12.25m. Two buses touching would have centers ~12.25m apart.
    // We use 15m (225 sq) to include a safety margin and allow for slightly larger future vehicles.
    double dx = car1->center.x - car2->center.x;
    double dy = car1->center.y - car2->center.y;
    double dist_sq = dx * dx + dy * dy;
    if (dist_sq > 225) { // 15^2 = 225
        return false;
    }

    RotatedRect2D car1_rect = {
        car1->center,
        (Dimensions) {car1->dimensions.y, car1->dimensions.x},
        car1->orientation
    };
    RotatedRect2D car2_rect = {
        car2->center,
        (Dimensions) {car2->dimensions.y, car2->dimensions.x},
        car2->orientation
    };
    bool colliding = rotated_rects_intersect(car1_rect, car2_rect);
    // if (colliding) {
    //     LOG_DEBUG("Collision detected between Car %d and Car %d", car1->id, car2->id);
    // }
    return colliding;
}
