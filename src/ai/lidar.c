#include "ai.h"
#include "logging.h"
#include <stdlib.h>
#include <string.h>


void lidar_capture(Lidar* lidar, Simulation* sim) {
    lidar_capture_exclude_objects(lidar, sim, (void*[]){ NULL } );
}


void lidar_capture_exclude_objects(Lidar* lidar, Simulation* sim, void** exclude_objects) {
    // Algo: starting from lidar orientation - fov/2 radians to orientation+fov/2 radians, scan "num_points" number of lines uniformly, and get unit vector. For each line, construct a line segment from origin to origin + max_depth * unit_vector.

    lidar_clear(lidar);

    for (int point_id = 0; point_id < lidar->num_points; point_id++) {
        double scan_progress = lidar->num_points == 1 ? 0.5 : (double) point_id / (lidar->num_points - 1);  // when num_points is 1, just point in the center
        Radians ray_orientation = lidar->orientation + lidar->fov * (scan_progress - 0.5);
        Vec2D ray_unit_vector = angle_to_unit_vector(ray_orientation);
        LineSegment ray = line_segment_create(lidar->position, vec_add(lidar->position, vec_scale(ray_unit_vector, lidar->max_depth)));
        Meters closest_distance = lidar->max_depth;
        
        // now cycle through all cars.
        for (CarId car_id = 0; car_id < sim_get_num_cars(sim); car_id++) {
            Car* car = sim_get_car(sim, car_id);
            if (object_array_contains_object(exclude_objects, car)) continue;
            for (int edge_idx = 0; edge_idx < 4; edge_idx++) {
                LineSegment car_edge = line_segment_create(car->corners[edge_idx], car->corners[(edge_idx + 1) % 4]);
                Coordinates intersection_point;
                if (line_segments_intersect(ray, car_edge, &intersection_point)) {
                    Meters distance = vec_distance(lidar->position, intersection_point);
                    if (distance < closest_distance) {
                        closest_distance = distance;
                    }
                }
            }
        }

        lidar_set_value_at(lidar, point_id, closest_distance);
    }
}


Lidar* lidar_malloc(Coordinates position, Radians orientation, int num_points, Radians fov, Meters max_depth) {
    Lidar* frame = (Lidar*) malloc(sizeof(Lidar));
    if (frame == NULL) {
        LOG_ERROR("Failed to allocate memory for Lidar");
        return NULL;
    }
    if (num_points <= 0) {
        LOG_ERROR("Number of points must be positive in lidar_malloc");
        free(frame);
        return NULL;
    }
    frame->position = position;
    frame->orientation = orientation;
    frame->num_points = num_points;
    frame->fov = fov;
    frame->max_depth = max_depth;
    frame->data = (double*) malloc(sizeof(double) * num_points);
    if (frame->data == NULL) {
        LOG_ERROR("Failed to allocate memory for depth data");
        free(frame);
        return NULL;
    }
    memset(frame->data, 0, sizeof(double) * num_points);
    return frame;
}

void lidar_free(Lidar* frame) {
    if (frame != NULL) {
        if (frame->data != NULL) {
            free(frame->data);
        }
        free(frame);
    }
}

void lidar_clear(Lidar* frame) {
    if (frame != NULL && frame->data != NULL) {
        memset(frame->data, 0, sizeof(double) * frame->num_points);
    }
}

double lidar_get_value_at(const Lidar* frame, int index) {
    if (index < 0 || index >= frame->num_points) {
        LOG_ERROR("Index out of bounds in lidar_get_value_at: index=%d", index);
        exit(EXIT_FAILURE);
    }
    return frame->data[index];
}

void lidar_set_value_at(Lidar* frame, int index, double value) {
    if (index < 0 || index >= frame->num_points) {
        LOG_ERROR("Index out of bounds in lidar_set_value_at: index=%d", index);
        exit(EXIT_FAILURE);
    }
    if (value < 0.0 || value > frame->max_depth) {
        LOG_ERROR("Depth value out of bounds in lidar_set_value_at: value=%f", value);
        exit(EXIT_FAILURE);
    }
    frame->data[index] = value;
}

void lidar_copy(const Lidar* src, Lidar* dest) {
    if (src->num_points != dest->num_points) {
        LOG_ERROR("Lidar copy failed: source and destination num_points do not match");
        return;
    }
    dest->fov = src->fov;
    dest->max_depth = src->max_depth;
    memcpy(dest->data, src->data, sizeof(double) * src->num_points);
}