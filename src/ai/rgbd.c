#include "ai.h"
#include "logging.h"
#include <stdlib.h>
#include <string.h>


bool should_exclude_object(void** exclude_objects, const void* object) {
    if (exclude_objects == NULL) return false;
    for (int i = 0; exclude_objects[i] != NULL; i++) {
        if (exclude_objects[i] == object) {
            return true;
        }
    }
    return false;
}

void rgbcam_capture(RGBCamera* camera, Simulation* sim, void** exclude_objects) {
    // 3 rows captured by the cam.
    // the top row will capture intersection lights or other road signs
    // the middle row will capture other cars
    // the bottom row will capture road features

    // Algo: Starting from cam orientation - fov/2 radians to orientation+fov/2 radians, scan "width" number of lines uniformly, and get unit vector. For each line, construct a line segment from origin to origin + max_depth * unit_vector.

    rgbcam_clear(camera);

    const int max_depth_meters = 1000; // 1 km

    for (int ray_id = 0; ray_id < camera->width; ray_id++) {
        double scan_progress =  camera->width == 1 ? 0.5 : (double) ray_id / (camera->width - 1);   // when width is 1, just point in the center
        Radians ray_orientation = camera->orientation + camera->fov * (scan_progress - 0.5);
        Vec2D ray_unit_vector = angle_to_unit_vector(ray_orientation);
        LineSegment ray = line_segment_create(camera->position, vec_add(camera->position, vec_scale(ray_unit_vector, max_depth_meters)));
        Meters closest_point_distance = INFINITY;
        RGB closest_point_rgb = {255, 255, 255}; // Background color (white)
        // now cycle through all cars.
        for (CarId car_id = 0; car_id < sim_get_num_cars(sim); car_id++) {
            Car* car = sim_get_car(sim, car_id);
            if (should_exclude_object(exclude_objects, car)) continue;
            for (int edge_idx = 0; edge_idx < 4; edge_idx++) {
                LineSegment car_edge = line_segment_create(car->corners[edge_idx], car->corners[(edge_idx + 1) % 4]);
                Coordinates intersection_point;
                if (line_segments_intersect(ray, car_edge, &intersection_point)) {
                    Meters distance = vec_distance(camera->position, intersection_point);
                    if (distance < closest_point_distance) {
                        closest_point_distance = distance;
                        closest_point_rgb = (RGB) {74, 103, 65};
                    }
                }
            }
        }
        rgbcam_set_pixel_at(camera, 1, ray_id, closest_point_rgb);

        // now we need a way to capture info about where we are on the road.. like where does the road end. In that case, 
    }
}


void lidar_capture(Lidar* lidar, Simulation* sim, void** exclude_objects) {
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
            if (should_exclude_object(exclude_objects, car)) continue;
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

RGBCamera* rgbcam_malloc(Coordinates position, Radians orientation, int width, int height, Radians fov) {
    RGBCamera* frame = (RGBCamera*) malloc(sizeof(RGBCamera));
    if (frame == NULL) {
        LOG_ERROR("Failed to allocate memory for RGBCamera");
        return NULL;
    }
    if (width <= 0 || height <= 0) {
        LOG_ERROR("Width and height must be positive in rgbcam_malloc");
        free(frame);
        return NULL;
    }
    frame->position = position;
    frame->orientation = orientation;
    frame->width = width;
    frame->height = height;
    frame->fov = fov;
    frame->data = (uint8_t*) malloc(sizeof(uint8_t) * 3 * width * height); // CHW format
    if (frame->data == NULL) {
        LOG_ERROR("Failed to allocate memory for RGB data");
        free(frame);
        return NULL;
    }
    memset(frame->data, 0, sizeof(uint8_t) * 3 * width * height);
    return frame;
}

void rgbcam_free(RGBCamera* frame) {
    if (frame != NULL) {
        if (frame->data != NULL) {
            free(frame->data);
        }
        free(frame);
    }
}

void rgbcam_clear(RGBCamera* frame) {
    if (frame != NULL && frame->data != NULL) {
        memset(frame->data, 0, sizeof(uint8_t) * 3 * frame->width * frame->height);
    }
}

static int rgbcam_get_index(const RGBCamera* frame, int channel, int row, int col) {
    // check bounds
    if (channel < 0 || channel >= 3 || row < 0 || row >= frame->height || col < 0 || col >= frame->width) {
        LOG_ERROR("Index out of bounds in rgbcam_get_index: channel=%d, row=%d, col=%d", channel, row, col);
        exit(EXIT_FAILURE);
    }
    return channel * (frame->width * frame->height) + row * frame->width + col;
}

uint8_t rgbcam_get_value_at(const RGBCamera* frame, int channel, int row, int col) {
    int index = rgbcam_get_index(frame, channel, row, col);
    return frame->data[index];
}

void rgbcam_set_value_at(RGBCamera* frame, int channel, int row, int col, uint8_t value) {
    int index = rgbcam_get_index(frame, channel, row, col);
    frame->data[index] = value;
}

RGB rgbcam_get_pixel_at(const RGBCamera* frame, int row, int col) {
    RGB pixel;
    pixel.r = rgbcam_get_value_at(frame, 0, row, col);
    pixel.g = rgbcam_get_value_at(frame, 1, row, col);
    pixel.b = rgbcam_get_value_at(frame, 2, row, col);
    return pixel;
}

void rgbcam_set_pixel_at(RGBCamera* frame, int row, int col, const RGB pixel) {
    rgbcam_set_value_at(frame, 0, row, col, pixel.r);
    rgbcam_set_value_at(frame, 1, row, col, pixel.g);
    rgbcam_set_value_at(frame, 2, row, col, pixel.b);
}

void rgbcam_copy(const RGBCamera* src, RGBCamera* dest) {
    if (src->width != dest->width || src->height != dest->height) {
        LOG_ERROR("RGBCamera copy failed: source and destination dimensions do not match");
        return;
    }
    dest->fov = src->fov;
    memcpy(dest->data, src->data, sizeof(uint8_t) * 3 * src->width * src->height);
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