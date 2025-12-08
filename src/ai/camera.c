#include "ai.h"
#include "logging.h"
#include <stdlib.h>
#include <string.h>

void rgbcam_capture(RGBCamera* camera, Simulation* sim, void** exclude_objects) {
    // pinhole camera model. Assume the camera screen is vertical, and the camera is at (position.x, position.y, z_altitude), looking towards orientation direction in the XY plane. The camera has a horizontal fov of fov radians, and vertical fov determined by width, height, and fov. In the pinhole model, there is a screen plane at distance d = (width/2) / tan(fov/2) from the camera center along the orientation direction. Each pixel corresponds to a ray from the camera center through the pixel on the screen plane, and extending to max_distance.

    rgbcam_clear(camera);

    int H = camera->height;
    int W = camera->width;
    double tan_fov_2 = tan(camera->fov / 2.0);
    double d = (W / 2.0) / tan_fov_2;

    double cos_o = cos(camera->orientation);
    double sin_o = sin(camera->orientation);

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            // Compute coords of pixel (h, w) in camera space. In its own space, camera is at origin, looking along +X axis (0 orientation), screen plane is at distance d along +X axis.
            double x_screen = d;
            double y_screen = (W / 2.0) - w;
            double z_screen = (H / 2.0) - h;

            // rotate around Z axis by orientation
            double x_rotated = x_screen * cos_o - y_screen * sin_o;
            double y_rotated = x_screen * sin_o + y_screen * cos_o;

            // create ray from camera position to pixel on screen plane
            Coordinates3D ray_start = { camera->position.x, camera->position.y, camera->z_altitude };
            Coordinates3D ray_dir = { x_rotated, y_rotated, z_screen };
            ray_dir = vec3d_normalize(ray_dir);
            LineSegment3D ray = line_segment3d_create(ray_start, vec3d_add(ray_start, vec3d_scale(ray_dir, camera->max_distance)));

            // setup for finding closest intersection
            Meters closest_distance = camera->max_distance;
            RGB pixel_color = {135, 206, 235}; // default sky color

            // scan for intersections with each (quad3d) side of each car
            for (CarId car_id = 0; car_id < sim_get_num_cars(sim); car_id++) {
                Car* car = sim_get_car(sim, car_id);
                if (object_array_contains_object(exclude_objects, car)) continue;
                // quick check, if the car's center is farther than max_distance + 10 m, skip
                Meters center_distance = vec_distance(camera->position, car->center);
                if (center_distance > camera->max_distance + meters(10.0)) {
                    continue;
                }
                
                // we know base of the car (in XY plane) has 2D corners car->corners[0..3] (front-right, front-left, rear-left, rear-right. (i.e., all anticlockwise)), and we assume each car is 3 meters tall. Let's create the 6 sides of the car as Quad3D objects.

                Coordinates3D b0 = {car->corners[0].x, car->corners[0].y, 0.0};
                Coordinates3D b1 = {car->corners[1].x, car->corners[1].y, 0.0};
                Coordinates3D b2 = {car->corners[2].x, car->corners[2].y, 0.0};
                Coordinates3D b3 = {car->corners[3].x, car->corners[3].y, 0.0};

                Coordinates3D t0 = {car->corners[0].x, car->corners[0].y, 3.0};
                Coordinates3D t1 = {car->corners[1].x, car->corners[1].y, 3.0};
                Coordinates3D t2 = {car->corners[2].x, car->corners[2].y, 3.0};
                Coordinates3D t3 = {car->corners[3].x, car->corners[3].y, 3.0};

                Quad3D sides[] = {
                    { {b0, b3, b2, b1} }, // Bottom
                    { {t0, t1, t2, t3} }, // Top
                    { {b0, b1, t1, t0} }, // Front
                    { {b1, b2, t2, t1} }, // Left
                    { {b2, b3, t3, t2} }, // Back
                    { {b3, b0, t0, t3} }  // Right
                };

                Coordinates3D intersection_point;
                for (int side_idx = 0; side_idx < 6; side_idx++) {
                    if (line_segment3d_intersect_quad(ray, sides[side_idx], &intersection_point)) {
                        Meters distance = vec3d_distance(ray_start, intersection_point);
                        if (distance < closest_distance) {
                            closest_distance = distance;
                            if (side_idx == 2) { // Front
                                pixel_color = (RGB) {255, 255, 100}; // Headlights
                            } else if (side_idx == 4) { // Back
                                bool is_braking = false;
                                if (car->speed > 0 && car->acceleration < -0.01) {
                                    is_braking = true;
                                } else if (car->speed < 0 && car->acceleration > 0.01) {
                                    is_braking = true;
                                }

                                if (is_braking) {
                                    pixel_color = (RGB) {255, 0, 0}; // Bright Red
                                } else {
                                    pixel_color = (RGB) {100, 0, 0}; // Dark Red
                                }
                            } else if (side_idx == 3) { // Left
                                RGB side_color = {74, 103, 65}; // Default Olive Green
                                CarIndicator ind = INDICATOR_NONE;
                                RGB ind_color = {0,0,0};
                                if (car->indicator_turn == INDICATOR_LEFT) {
                                    ind = INDICATOR_LEFT;
                                    ind_color = (RGB){255, 0, 0};
                                } else if (car->indicator_lane == INDICATOR_LEFT) {
                                    ind = INDICATOR_LEFT;
                                    ind_color = (RGB){255, 140, 0};
                                }

                                if (ind != INDICATOR_NONE) {
                                    // Mix colors: average
                                    side_color.r = (side_color.r + ind_color.r) / 2;
                                    side_color.g = (side_color.g + ind_color.g) / 2;
                                    side_color.b = (side_color.b + ind_color.b) / 2;
                                }
                                pixel_color = side_color;
                            } else if (side_idx == 5) { // Right
                                RGB side_color = {0, 0, 255}; // Default Blue
                                CarIndicator ind = INDICATOR_NONE;
                                RGB ind_color = {0,0,0};
                                if (car->indicator_turn == INDICATOR_RIGHT) {
                                    ind = INDICATOR_RIGHT;
                                    ind_color = (RGB){255, 0, 0};
                                } else if (car->indicator_lane == INDICATOR_RIGHT) {
                                    ind = INDICATOR_RIGHT;
                                    ind_color = (RGB){255, 140, 0};
                                }

                                if (ind != INDICATOR_NONE) {
                                    // Mix colors: average
                                    side_color.r = (side_color.r + ind_color.r) / 2;
                                    side_color.g = (side_color.g + ind_color.g) / 2;
                                    side_color.b = (side_color.b + ind_color.b) / 2;
                                }
                                pixel_color = side_color;
                            } else {
                                pixel_color = (RGB) {74, 103, 65}; // Car color
                            }
                        }
                    }
                }
            }

            // now scan all lanes for road surface intersection
            Map* map = sim_get_map(sim);
            for (RoadId road_id = 0; road_id < map_get_num_roads(map); road_id++) {
                Road* road = map_get_road(map, road_id);
                if (object_array_contains_object(exclude_objects, road)) continue;
                // if it is not a straight road, skip
                if (road_get_type(road) == STRAIGHT) {
                    // we know the center, length and width of the road and its direction, so we can compute its 4 corners
                    Coordinates road_center = road_get_center(road);
                    Meters delta_x, delta_y;
                    bool is_horizontal = (road_get_direction(road) == DIRECTION_EAST || road_get_direction(road) == DIRECTION_WEST);
                    if (is_horizontal) {
                        delta_x = road_get_length(road) / 2.0;
                        delta_y = road_get_width(road) / 2.0;
                    } else {
                        delta_x = road_get_width(road) / 2.0;
                        delta_y = road_get_length(road) / 2.0;
                    }
                    Coordinates3D r0 = { road_center.x - delta_x, road_center.y - delta_y, 0.0 };
                    Coordinates3D r1 = { road_center.x + delta_x, road_center.y - delta_y, 0.0 };
                    Coordinates3D r2 = { road_center.x + delta_x, road_center.y + delta_y, 0.0 };
                    Coordinates3D r3 = { road_center.x - delta_x, road_center.y + delta_y, 0.0 };

                    Quad3D road_surface = { {r0, r1, r2, r3} };
                    Coordinates3D intersection_point;
                    if (line_segment3d_intersect_quad(ray, road_surface, &intersection_point)) {
                        Meters distance = vec3d_distance(ray_start, intersection_point);
                        if (distance < closest_distance) {
                            closest_distance = distance;
                            if (road_get_direction(road) == DIRECTION_WEST || road_get_direction(road) == DIRECTION_SOUTH) {
                                pixel_color = (RGB) {100, 100, 100}; // Darker Road color
                            } else {
                                pixel_color = (RGB) {128, 128, 128}; // Road color
                            }
                        }
                    }
                } else if (road_get_type(road) == TURN) {
                    Coordinates center = road_get_center(road);
                    Meters radius = road->radius;
                    Meters width = road_get_width(road);
                    Meters r_out = radius + width / 2.0;
                    Meters r_in = radius - width / 2.0;
                    Quadrant quad = road->quadrant;

                    double min_x, max_x, min_y, max_y;
                    if (quad == QUADRANT_TOP_RIGHT) {
                        min_x = center.x; max_x = center.x + r_out;
                        min_y = center.y; max_y = center.y + r_out;
                    } else if (quad == QUADRANT_TOP_LEFT) {
                        min_x = center.x - r_out; max_x = center.x;
                        min_y = center.y; max_y = center.y + r_out;
                    } else if (quad == QUADRANT_BOTTOM_LEFT) {
                        min_x = center.x - r_out; max_x = center.x;
                        min_y = center.y - r_out; max_y = center.y;
                    } else { // QUADRANT_BOTTOM_RIGHT
                        min_x = center.x; max_x = center.x + r_out;
                        min_y = center.y - r_out; max_y = center.y;
                    }

                    Coordinates3D q0 = {min_x, min_y, 0.0};
                    Coordinates3D q1 = {max_x, min_y, 0.0};
                    Coordinates3D q2 = {max_x, max_y, 0.0};
                    Coordinates3D q3 = {min_x, max_y, 0.0};

                    Quad3D turn_surface = { {q0, q1, q2, q3} };
                    Coordinates3D intersection_point;
                    if (line_segment3d_intersect_quad(ray, turn_surface, &intersection_point)) {
                        double dist_sq = (intersection_point.x - center.x) * (intersection_point.x - center.x) +
                                         (intersection_point.y - center.y) * (intersection_point.y - center.y);
                        if (dist_sq >= r_in * r_in && dist_sq <= r_out * r_out) {
                             Meters distance = vec3d_distance(ray_start, intersection_point);
                             if (distance < closest_distance) {
                                 closest_distance = distance;
                                 if (road_get_direction(road) == DIRECTION_CW) {
                                     pixel_color = (RGB) {100, 100, 100}; // Darker Road color
                                 } else {
                                     pixel_color = (RGB) {128, 128, 128}; // Road color
                                 }
                             }
                        }
                    }
                }
            }

            // go through intersections
            for (IntersectionId intersection_id = 0; intersection_id < map_get_num_intersections(map); intersection_id++) {
                Intersection* intersection = map_get_intersection(map, intersection_id);
                if (object_array_contains_object(exclude_objects, intersection)) continue;
                // we know center and 'dimensions' of intersection.
                Coordinates intersection_center = intersection->center;
                Dimensions dims = intersection->dimensions;
                Coordinates3D i0 = { intersection_center.x - dims.x / 2.0, intersection_center.y - dims.y / 2.0, 0.0 };
                Coordinates3D i1 = { intersection_center.x + dims.x / 2.0, intersection_center.y - dims.y / 2.0, 0.0 };
                Coordinates3D i2 = { intersection_center.x + dims.x / 2.0, intersection_center.y + dims.y / 2.0, 0.0 };
                Coordinates3D i3 = { intersection_center.x - dims.x / 2.0, intersection_center.y + dims.y / 2.0, 0.0 };

                Quad3D intersection_surface = { {i0, i1, i2, i3} };
                Coordinates3D intersection_point;
                if (line_segment3d_intersect_quad(ray, intersection_surface, &intersection_point)) {
                    Meters distance = vec3d_distance(ray_start, intersection_point);
                    if (distance < closest_distance) {
                        closest_distance = distance;
                        pixel_color = (RGB) {150, 150, 150}; // Intersection color
                    }
                }
            }

            // set pixel color
            rgbcam_set_pixel_at(camera, h, w, pixel_color);
        }
    }
}


RGBCamera* rgbcam_malloc(Coordinates position, Meters z_altitude, Radians orientation, int width, int height, Radians fov, Meters max_distance) {
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
    frame->z_altitude = z_altitude;
    frame->orientation = orientation;
    frame->width = width;
    frame->height = height;
    frame->fov = fov;
    frame->max_distance = max_distance;
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

// HWC format
static int rgbcam_get_index(const RGBCamera* frame, int channel, int row, int col) {
    // check bounds
    if (channel < 0 || channel >= 3 || row < 0 || row >= frame->height || col < 0 || col >= frame->width) {
        LOG_ERROR("Index out of bounds in rgbcam_get_index: channel=%d, row=%d, col=%d", channel, row, col);
        exit(EXIT_FAILURE);
    }
    return row * (frame->width * 3) + col * 3 + channel;
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
