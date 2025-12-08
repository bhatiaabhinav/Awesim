#include "ai.h"
#include "logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define MIN_Z_NEAR 0.1
#define TURN_SEGMENTS 10
#define EPSILON 1e-9
#define ROAD_THICKNESS 0.5
#define TRAFFIC_LIGHT_SIZE 2.0
#define TRAFFIC_LIGHT_HEIGHT 8.0

typedef struct {
    double x, y, z;
} Vec3;

typedef struct {
    double x, y; // Screen coordinates (pixels)
    double z;    // Camera space Z (forward)
} ProjectedVertex;

// Z-buffer array
static double* z_buffer = NULL;
static int z_buffer_size = 0;

static void ensure_z_buffer(int size) {
    if (size > z_buffer_size) {
        if (z_buffer) free(z_buffer);
        z_buffer = (double*)malloc(sizeof(double) * size);
        z_buffer_size = size;
    }
}

static void clear_z_buffer(int width, int height, double max_dist) {
    for (int i = 0; i < width * height; i++) {
        z_buffer[i] = max_dist;
    }
}

static void rasterize_triangle(RGBCamera* camera, ProjectedVertex v0, ProjectedVertex v1, ProjectedVertex v2, RGB color) {
    // Bounding box
    int min_x = (int)floor(fmin(v0.x, fmin(v1.x, v2.x)));
    int max_x = (int)ceil(fmax(v0.x, fmax(v1.x, v2.x)));
    int min_y = (int)floor(fmin(v0.y, fmin(v1.y, v2.y)));
    int max_y = (int)ceil(fmax(v0.y, fmax(v1.y, v2.y)));

    // Clip to screen
    min_x = fmax(min_x, 0);
    max_x = fmin(max_x, camera->width - 1);
    min_y = fmax(min_y, 0);
    max_y = fmin(max_y, camera->height - 1);

    // Precompute denominator
    double denom = (v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y);
    if (fabs(denom) < EPSILON) return; // Skip degenerate triangles

    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            double w0, w1, w2;
            double px = x + 0.5;
            double py = y + 0.5;

            w0 = ((v1.y - v2.y) * (px - v2.x) + (v2.x - v1.x) * (py - v2.y)) / denom;
            w1 = ((v2.y - v0.y) * (px - v2.x) + (v0.x - v2.x) * (py - v2.y)) / denom;
            w2 = 1.0 - w0 - w1;

            if (w0 >= -EPSILON && w1 >= -EPSILON && w2 >= -EPSILON) {
                // Perspective correct interpolation for Z
                // We interpolate 1/z
                double z_inv = w0 * (1.0 / v0.z) + w1 * (1.0 / v1.z) + w2 * (1.0 / v2.z);
                if (z_inv <= EPSILON) continue; // Avoid division by zero or negative Z
                double z = 1.0 / z_inv;

                // Calculate actual distance from camera (Euclidean)
                double tan_fov_2 = tan(camera->fov / 2.0);
                double d = (camera->width / 2.0) / tan_fov_2;
                
                // Use pixel center for reconstruction too
                double y_cam = ((camera->width / 2.0) - px) * z / d;
                double z_cam = ((camera->height / 2.0) - py) * z / d;
                
                double dist = sqrt(y_cam * y_cam + z_cam * z_cam + z * z);

                int idx = y * camera->width + x;
                if (dist < z_buffer[idx]) {
                    z_buffer[idx] = dist;
                    rgbcam_set_pixel_at(camera, y, x, color);
                }
            }
        }
    }
}

static void project_and_draw_triangle(RGBCamera* camera, Vec3 p0, Vec3 p1, Vec3 p2, RGB color) {
    double tan_fov_2 = tan(camera->fov / 2.0);
    double d = (camera->width / 2.0) / tan_fov_2;

    ProjectedVertex v[3];
    Vec3 ps[3] = {p0, p1, p2};

    for(int i=0; i<3; i++) {
        v[i].z = ps[i].x; // Camera X is forward distance
        
        double y_screen = d * ps[i].y / ps[i].x;
        double z_screen = d * ps[i].z / ps[i].x;
        
        v[i].x = (camera->width / 2.0) - y_screen;
        v[i].y = (camera->height / 2.0) - z_screen;
    }
    rasterize_triangle(camera, v[0], v[1], v[2], color);
}

static Vec3 intersect_near_plane(Vec3 a, Vec3 b) {
    double t = (MIN_Z_NEAR - a.x) / (b.x - a.x);
    Vec3 res;
    res.x = MIN_Z_NEAR;
    res.y = a.y + t * (b.y - a.y);
    res.z = a.z + t * (b.z - a.z);
    return res;
}

static void clip_and_rasterize_triangle(RGBCamera* camera, Vec3 p0, Vec3 p1, Vec3 p2, RGB color) {
    Vec3* in_pts[3];
    Vec3* out_pts[3];
    int n_in = 0;
    int n_out = 0;

    if (p0.x >= MIN_Z_NEAR) in_pts[n_in++] = &p0; else out_pts[n_out++] = &p0;
    if (p1.x >= MIN_Z_NEAR) in_pts[n_in++] = &p1; else out_pts[n_out++] = &p1;
    if (p2.x >= MIN_Z_NEAR) in_pts[n_in++] = &p2; else out_pts[n_out++] = &p2;

    if (n_in == 0) {
        return; // All outside
    } else if (n_in == 3) {
        project_and_draw_triangle(camera, p0, p1, p2, color);
    } else if (n_in == 1) {
        // 1 inside, 2 outside. Clip to 1 triangle.
        Vec3 v_in = *in_pts[0];
        Vec3 v_out1 = *out_pts[0];
        Vec3 v_out2 = *out_pts[1];
        
        Vec3 i1 = intersect_near_plane(v_in, v_out1);
        Vec3 i2 = intersect_near_plane(v_in, v_out2);
        
        project_and_draw_triangle(camera, v_in, i1, i2, color);
    } else if (n_in == 2) {
        // 2 inside, 1 outside. Clip to 2 triangles (quad).
        Vec3 v_in1 = *in_pts[0];
        Vec3 v_in2 = *in_pts[1];
        Vec3 v_out = *out_pts[0];
        
        Vec3 i1 = intersect_near_plane(v_in1, v_out);
        Vec3 i2 = intersect_near_plane(v_in2, v_out);
        
        project_and_draw_triangle(camera, v_in1, v_in2, i2, color);
        project_and_draw_triangle(camera, v_in1, i2, i1, color);
    }
}

static void rasterize_quad(RGBCamera* camera, Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3, RGB color) {
    // Split into 2 triangles: 0-1-2 and 0-2-3
    clip_and_rasterize_triangle(camera, p0, p1, p2, color);
    clip_and_rasterize_triangle(camera, p0, p2, p3, color);
}

// Helper to transform world to camera
static Vec3 to_cam(Coordinates3D p, Coordinates3D cam_pos, double cos_o, double sin_o) {
    double dx = p.x - cam_pos.x;
    double dy = p.y - cam_pos.y;
    double dz = p.z - cam_pos.z;
    
    Vec3 v;
    v.x = dx * cos_o + dy * sin_o;
    v.y = -dx * sin_o + dy * cos_o;
    v.z = dz;
    return v;
}

static void rasterize_thick_quad(RGBCamera* camera, Coordinates3D p0, Coordinates3D p1, Coordinates3D p2, Coordinates3D p3, RGB color, Coordinates3D cam_pos, double cos_o, double sin_o) {
    // Top
    rasterize_quad(camera, to_cam(p0, cam_pos, cos_o, sin_o), to_cam(p1, cam_pos, cos_o, sin_o), to_cam(p2, cam_pos, cos_o, sin_o), to_cam(p3, cam_pos, cos_o, sin_o), color);
    
    // Bottom vertices (World Space)
    Coordinates3D p0_b = {p0.x, p0.y, p0.z - ROAD_THICKNESS};
    Coordinates3D p1_b = {p1.x, p1.y, p1.z - ROAD_THICKNESS};
    Coordinates3D p2_b = {p2.x, p2.y, p2.z - ROAD_THICKNESS};
    Coordinates3D p3_b = {p3.x, p3.y, p3.z - ROAD_THICKNESS};

    // Sides
    rasterize_quad(camera, to_cam(p0, cam_pos, cos_o, sin_o), to_cam(p1, cam_pos, cos_o, sin_o), to_cam(p1_b, cam_pos, cos_o, sin_o), to_cam(p0_b, cam_pos, cos_o, sin_o), color);
    rasterize_quad(camera, to_cam(p1, cam_pos, cos_o, sin_o), to_cam(p2, cam_pos, cos_o, sin_o), to_cam(p2_b, cam_pos, cos_o, sin_o), to_cam(p1_b, cam_pos, cos_o, sin_o), color);
    rasterize_quad(camera, to_cam(p2, cam_pos, cos_o, sin_o), to_cam(p3, cam_pos, cos_o, sin_o), to_cam(p3_b, cam_pos, cos_o, sin_o), to_cam(p2_b, cam_pos, cos_o, sin_o), color);
    rasterize_quad(camera, to_cam(p3, cam_pos, cos_o, sin_o), to_cam(p0, cam_pos, cos_o, sin_o), to_cam(p0_b, cam_pos, cos_o, sin_o), to_cam(p3_b, cam_pos, cos_o, sin_o), color);
}

static void rasterize_cube(RGBCamera* camera, Coordinates3D center, double size, RGB color_top, RGB color_bottom, RGB color_front, RGB color_back, RGB color_left, RGB color_right, Coordinates3D cam_pos, double cos_o, double sin_o) {
    double s = size / 2.0;
    Coordinates3D p0 = {center.x - s, center.y - s, center.z - s};
    Coordinates3D p1 = {center.x + s, center.y - s, center.z - s};
    Coordinates3D p2 = {center.x + s, center.y + s, center.z - s};
    Coordinates3D p3 = {center.x - s, center.y + s, center.z - s};
    Coordinates3D p4 = {center.x - s, center.y - s, center.z + s};
    Coordinates3D p5 = {center.x + s, center.y - s, center.z + s};
    Coordinates3D p6 = {center.x + s, center.y + s, center.z + s};
    Coordinates3D p7 = {center.x - s, center.y + s, center.z + s};

    // Top
    rasterize_quad(camera, to_cam(p4, cam_pos, cos_o, sin_o), to_cam(p5, cam_pos, cos_o, sin_o), to_cam(p6, cam_pos, cos_o, sin_o), to_cam(p7, cam_pos, cos_o, sin_o), color_top);
    // Bottom
    rasterize_quad(camera, to_cam(p0, cam_pos, cos_o, sin_o), to_cam(p3, cam_pos, cos_o, sin_o), to_cam(p2, cam_pos, cos_o, sin_o), to_cam(p1, cam_pos, cos_o, sin_o), color_bottom);
    // Front (y-s)
    rasterize_quad(camera, to_cam(p0, cam_pos, cos_o, sin_o), to_cam(p1, cam_pos, cos_o, sin_o), to_cam(p5, cam_pos, cos_o, sin_o), to_cam(p4, cam_pos, cos_o, sin_o), color_front);
    // Back (y+s)
    rasterize_quad(camera, to_cam(p2, cam_pos, cos_o, sin_o), to_cam(p3, cam_pos, cos_o, sin_o), to_cam(p7, cam_pos, cos_o, sin_o), to_cam(p6, cam_pos, cos_o, sin_o), color_back);
    // Left (x-s)
    rasterize_quad(camera, to_cam(p3, cam_pos, cos_o, sin_o), to_cam(p0, cam_pos, cos_o, sin_o), to_cam(p4, cam_pos, cos_o, sin_o), to_cam(p7, cam_pos, cos_o, sin_o), color_left);
    // Right (x+s)
    rasterize_quad(camera, to_cam(p1, cam_pos, cos_o, sin_o), to_cam(p2, cam_pos, cos_o, sin_o), to_cam(p6, cam_pos, cos_o, sin_o), to_cam(p5, cam_pos, cos_o, sin_o), color_right);
}

void rgbcam_capture_efficient(RGBCamera* camera, Simulation* sim, void** exclude_objects) {
    // Clear to sky color (gradient from top to horizon)
    RGB sky_top = {135, 206, 235}; // Sky Blue
    RGB sky_horizon = {255, 255, 255}; // White at horizon

    for (int y = 0; y < camera->height; y++) {
        double t = (double)y / (camera->height / 2.0);
        if (t > 1.0) t = 1.0;

        uint8_t r = (uint8_t)(sky_top.r * (1.0 - t) + sky_horizon.r * t);
        uint8_t g = (uint8_t)(sky_top.g * (1.0 - t) + sky_horizon.g * t);
        uint8_t b = (uint8_t)(sky_top.b * (1.0 - t) + sky_horizon.b * t);

        for (int x = 0; x < camera->width; x++) {
            int idx = (y * camera->width + x) * 3;
            camera->data[idx + 0] = r;
            camera->data[idx + 1] = g;
            camera->data[idx + 2] = b;
        }
    }

    ensure_z_buffer(camera->width * camera->height);
    clear_z_buffer(camera->width, camera->height, camera->max_distance);

    double cos_o = cos(camera->orientation);
    double sin_o = sin(camera->orientation);
    Coordinates3D cam_pos = { camera->position.x, camera->position.y, camera->z_altitude };

    // 0. Ground
    {
        double ground_half_size = 2500.0;
        double ground_z = -0.33; // Slightly below 0 to avoid z-fighting with road surface
        Coordinates3D g0 = { camera->position.x - ground_half_size, camera->position.y - ground_half_size, ground_z };
        Coordinates3D g1 = { camera->position.x + ground_half_size, camera->position.y - ground_half_size, ground_z };
        Coordinates3D g2 = { camera->position.x + ground_half_size, camera->position.y + ground_half_size, ground_z };
        Coordinates3D g3 = { camera->position.x - ground_half_size, camera->position.y + ground_half_size, ground_z };
        
        RGB ground_color = {34, 139, 34}; // Forest Green
        rasterize_quad(camera, to_cam(g0, cam_pos, cos_o, sin_o), to_cam(g1, cam_pos, cos_o, sin_o), to_cam(g2, cam_pos, cos_o, sin_o), to_cam(g3, cam_pos, cos_o, sin_o), ground_color);
    }

    // 1. Cars
    for (CarId car_id = 0; car_id < sim_get_num_cars(sim); car_id++) {
        Car* car = sim_get_car(sim, car_id);
        if (object_array_contains_object(exclude_objects, car)) continue;

        Meters center_distance = vec_distance(camera->position, car->center);
        if (center_distance > camera->max_distance + 10.0) continue;

        Coordinates3D b[4], t[4];
        for(int i=0; i<4; i++) {
            b[i] = (Coordinates3D){car->corners[i].x, car->corners[i].y, 0.0};
            t[i] = (Coordinates3D){car->corners[i].x, car->corners[i].y, car->dimensions.z};
        }

        // Transform all vertices
        Vec3 vb[4], vt[4];
        for(int i=0; i<4; i++) {
            vb[i] = to_cam(b[i], cam_pos, cos_o, sin_o);
            vt[i] = to_cam(t[i], cam_pos, cos_o, sin_o);
        }

        // Define sides (indices)
        // Bottom: b0, b3, b2, b1
        // Top: t0, t1, t2, t3
        // Front: b0, b1, t1, t0
        // Left: b1, b2, t2, t1
        // Back: b2, b3, t3, t2
        // Right: b3, b0, t0, t3

        // Colors
        RGB colors[6];
        // Bottom (not visible usually, but set to car color)
        colors[0] = (RGB){74, 103, 65}; 
        // Top
        colors[1] = (RGB){74, 103, 65};
        // Front (Headlights)
        colors[2] = (RGB){255, 255, 100};
        // Left
        {
            RGB side_color = {74, 103, 65};
            if (car->indicator_turn == INDICATOR_LEFT || car->indicator_lane == INDICATOR_LEFT) {
                RGB ind_color = (car->indicator_turn == INDICATOR_LEFT) ? (RGB){255, 0, 0} : (RGB){255, 140, 0};
                side_color.r = (side_color.r + ind_color.r) / 2;
                side_color.g = (side_color.g + ind_color.g) / 2;
                side_color.b = (side_color.b + ind_color.b) / 2;
            }
            colors[3] = side_color;
        }
        // Back
        {
            bool is_braking = (car->speed > 0 && car->acceleration < -0.01) || (car->speed < 0 && car->acceleration > 0.01);
            colors[4] = is_braking ? (RGB){255, 0, 0} : (RGB){100, 0, 0};
        }
        // Right
        {
            RGB side_color = {0, 0, 255};
            if (car->indicator_turn == INDICATOR_RIGHT || car->indicator_lane == INDICATOR_RIGHT) {
                RGB ind_color = (car->indicator_turn == INDICATOR_RIGHT) ? (RGB){255, 0, 0} : (RGB){255, 140, 0};
                side_color.r = (side_color.r + ind_color.r) / 2;
                side_color.g = (side_color.g + ind_color.g) / 2;
                side_color.b = (side_color.b + ind_color.b) / 2;
            }
            colors[5] = side_color;
        }

        // Rasterize sides
        rasterize_quad(camera, vb[0], vb[3], vb[2], vb[1], colors[0]); // Bottom
        rasterize_quad(camera, vt[0], vt[1], vt[2], vt[3], colors[1]); // Top
        rasterize_quad(camera, vb[0], vb[1], vt[1], vt[0], colors[2]); // Front
        rasterize_quad(camera, vb[1], vb[2], vt[2], vt[1], colors[3]); // Left
        rasterize_quad(camera, vb[2], vb[3], vt[3], vt[2], colors[4]); // Back
        rasterize_quad(camera, vb[3], vb[0], vt[0], vt[3], colors[5]); // Right
    }

    // 2. Roads
    Map* map = sim_get_map(sim);
    for (RoadId road_id = 0; road_id < map_get_num_roads(map); road_id++) {
        Road* road = map_get_road(map, road_id);
        if (object_array_contains_object(exclude_objects, road)) continue;

        if (road_get_type(road) == STRAIGHT) {
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

            RGB color = (road_get_direction(road) == DIRECTION_WEST || road_get_direction(road) == DIRECTION_SOUTH) ? 
                        (RGB){100, 100, 100} : (RGB){128, 128, 128};
            
            rasterize_thick_quad(camera, r0, r1, r2, r3, color, cam_pos, cos_o, sin_o);

        } else if (road_get_type(road) == TURN) {
            Coordinates center = road_get_center(road);
            Meters radius = road->radius;
            Meters width = road_get_width(road);
            Meters r_out = radius + width / 2.0;
            Meters r_in = radius - width / 2.0;
            Quadrant quad = road->quadrant;

            double start_angle, end_angle;
            if (quad == QUADRANT_TOP_RIGHT) { start_angle = 0; end_angle = M_PI_2; }
            else if (quad == QUADRANT_TOP_LEFT) { start_angle = M_PI_2; end_angle = M_PI; }
            else if (quad == QUADRANT_BOTTOM_LEFT) { start_angle = M_PI; end_angle = 3*M_PI_2; }
            else { start_angle = 3*M_PI_2; end_angle = 2*M_PI; }

            RGB color = (road_get_direction(road) == DIRECTION_CW) ? (RGB){100, 100, 100} : (RGB){128, 128, 128};

            for (int i = 0; i < TURN_SEGMENTS; i++) {
                double a1 = start_angle + (end_angle - start_angle) * i / (double)TURN_SEGMENTS;
                double a2 = start_angle + (end_angle - start_angle) * (i + 1) / (double)TURN_SEGMENTS;

                Coordinates3D p0 = { center.x + r_in * cos(a1), center.y + r_in * sin(a1), 0.0 };
                Coordinates3D p1 = { center.x + r_out * cos(a1), center.y + r_out * sin(a1), 0.0 };
                Coordinates3D p2 = { center.x + r_out * cos(a2), center.y + r_out * sin(a2), 0.0 };
                Coordinates3D p3 = { center.x + r_in * cos(a2), center.y + r_in * sin(a2), 0.0 };

                rasterize_thick_quad(camera, p0, p1, p2, p3, color, cam_pos, cos_o, sin_o);
            }
        }
    }

    // 3. Intersections
    for (IntersectionId intersection_id = 0; intersection_id < map_get_num_intersections(map); intersection_id++) {
        Intersection* intersection = map_get_intersection(map, intersection_id);
        if (object_array_contains_object(exclude_objects, intersection)) continue;

        Coordinates intersection_center = intersection->center;
        Dimensions dims = intersection->dimensions;
        Coordinates3D i0 = { intersection_center.x - dims.x / 2.0, intersection_center.y - dims.y / 2.0, 0.0 };
        Coordinates3D i1 = { intersection_center.x + dims.x / 2.0, intersection_center.y - dims.y / 2.0, 0.0 };
        Coordinates3D i2 = { intersection_center.x + dims.x / 2.0, intersection_center.y + dims.y / 2.0, 0.0 };
        Coordinates3D i3 = { intersection_center.x - dims.x / 2.0, intersection_center.y + dims.y / 2.0, 0.0 };

        rasterize_thick_quad(camera, i0, i1, i2, i3, (RGB){150, 150, 150}, cam_pos, cos_o, sin_o);

        // Traffic Light
        Coordinates3D tl_center = { intersection_center.x, intersection_center.y, TRAFFIC_LIGHT_HEIGHT };
        RGB c_ns = {0, 0, 0};
        RGB c_ew = {0, 0, 0};
        
        IntersectionState state = intersection_get_state(intersection);
        if (state == NS_GREEN_EW_RED) {
            c_ns = (RGB){0, 255, 0}; // Green
            c_ew = (RGB){255, 0, 0}; // Red
        } else if (state == NS_YELLOW_EW_RED) {
            c_ns = (RGB){255, 255, 0}; // Yellow
            c_ew = (RGB){255, 0, 0}; // Red
        } else if (state == NS_RED_EW_GREEN) {
            c_ns = (RGB){255, 0, 0}; // Red
            c_ew = (RGB){0, 255, 0}; // Green
        } else if (state == EW_YELLOW_NS_RED) {
            c_ns = (RGB){255, 0, 0}; // Red
            c_ew = (RGB){255, 255, 0}; // Yellow
        } else if (state == ALL_RED_BEFORE_NS_GREEN || state == ALL_RED_BEFORE_EW_GREEN || state == FOUR_WAY_STOP) {
            c_ns = (RGB){255, 0, 0}; // Red
            c_ew = (RGB){255, 0, 0}; // Red
        }

        // Cube faces: Top, Bottom, Front (S), Back (N), Left (W), Right (E)
        // Note: Front is -Y (South), Back is +Y (North), Left is -X (West), Right is +X (East)
        // So Front/Back are NS facing, Left/Right are EW facing.
        rasterize_cube(camera, tl_center, TRAFFIC_LIGHT_SIZE, 
            (RGB){50, 50, 50}, (RGB){50, 50, 50}, // Top, Bottom (Dark Grey)
            c_ns, c_ns, // Front (S), Back (N) -> NS lights
            c_ew, c_ew, // Left (W), Right (E) -> EW lights
            cam_pos, cos_o, sin_o);
    }
}
