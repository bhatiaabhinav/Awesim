#include "ai.h"
#include "logging.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// --- Benchmarking ---
// #define ENABLE_BENCHMARKING // Uncomment to enable benchmarking

#ifdef ENABLE_BENCHMARKING
static int bench_frame_count = 0;
static double bench_time_sky = 0;
static double bench_time_ground = 0;
static double bench_time_cars = 0;
static double bench_time_roads = 0;
static double bench_time_intersections = 0;
static double bench_time_total = 0;
static double bench_time_hud = 0;
static double bench_time_downsample = 0;

#define BENCH_START(var) double var##_start = get_sys_time_seconds();
#define BENCH_END(var) var += get_sys_time_seconds() - var##_start;
#else
#define BENCH_START(var)
#define BENCH_END(var)
#endif

// --- Optimization Switches ---
// Uncomment to enable optimizations/approximations
// #define OPT_SIMPLE_CARS          // Render cars as simple boxes
// #define OPT_NO_CAR_LIGHTS        // Do not render car lights/indicators
// #define OPT_SIMPLE_LIGHTS        // Render lights as flat quads instead of thick ones
// #define OPT_NO_LANE_MARKINGS     // Do not render lane markings
// #define OPT_SIMPLE_LANE_MARKINGS // Reduce segments for curved lanes
// #define OPT_NO_DASHED_LINES      // Render dashed lines as solid (faster)
// #define OPT_SIMPLE_INTERSECTIONS // Do not render intersection edges/lights
// #define OPT_NO_HUD               // Do not render HUD

// --- Rendering Options ---
#define OPT_SPAN_CLIPPING        // Enable per-row span clipping (much faster)
// #define OPT_Z_BUFFER_EUCLIDEAN   // Use Euclidean distance for Z-buffer (slower, but maybe more precise)

// --- Configuration Constants ---

// Camera & Rendering
#define MIN_Z_NEAR 0.1
#define EPSILON 1e-9
#define TURN_SEGMENTS 10

// Sky
#define SKY_TOP_RGB (RGB){135, 206, 235}
#define SKY_HORIZON_RGB (RGB){255, 255, 255}

// Ground
#define GROUND_SIZE 2500.0
#define GROUND_Z -0.33
#define GROUND_RGB (RGB){34, 139, 34}

// Roads & Lanes
#define ROAD_THICKNESS 0.5
#define LANE_MARKING_WIDTH 0.15
#define LANE_MARKING_Z 0.1
#define LANE_MARKING_THICKNESS 0.1
#define DASH_LENGTH 3.0
#define DASH_GAP 9.0
#define ROAD_COLOR_DARK_RGB (RGB){100, 100, 100}
#define ROAD_COLOR_LIGHT_RGB (RGB){128, 128, 128}

// Cars
#define CAR_RENDER_DIST_OFFSET 10.0
#define CAR_BODY_RGB (RGB){74, 103, 65}
#define CAR_HEADLIGHT_RGB (RGB){255, 255, 100}
#define CAR_INDICATOR_RED_RGB (RGB){255, 0, 0}
#define CAR_INDICATOR_ORANGE_RGB (RGB){255, 140, 0}
#define CAR_BRAKE_RGB (RGB){255, 0, 0}
#define CAR_TAILLIGHT_RGB (RGB){100, 0, 0}
#define CAR_SIDE_RIGHT_RGB (RGB){74, 103, 128} // more blueish to differentiate from left side

// Intersections
#define INTERSECTION_RGB (RGB){175, 175, 175}
#define INTERSECTION_EDGE_WIDTH 0.3
#define INTERSECTION_EDGE_RGB (RGB){255, 255, 255}

// Traffic Lights
#define TRAFFIC_LIGHT_SIZE 2.0
#define TRAFFIC_LIGHT_HEIGHT 6.0
#define TL_POLE_WIDTH 0.4
#define TL_POLE_RGB (RGB){100, 100, 100}
#define TL_BODY_RGB (RGB){50, 50, 50}
#define TL_RED_RGB (RGB){255, 30, 30}
#define TL_GREEN_RGB (RGB){50, 250, 50}
#define TL_YELLOW_RGB (RGB){220, 180, 20}

// HUD
#define HUD_HEIGHT 20
#define HUD_TEXT_SCALE 2
#define HUD_BG_RGB (RGB){0, 0, 0}
#define HUD_TICK_RGB (RGB){255, 255, 255}
#define HUD_TEXT_RGB (RGB){255, 255, 255}
#define HUD_NORTH_RGB (RGB){255, 50, 50}
#define HUD_NEEDLE_RGB (RGB){255, 255, 0}

#ifndef M_PI_4
#define M_PI_4 0.78539816339744830962
#endif

typedef struct {
    double x, y, z;
} Vec3;

typedef struct {
    double x, y; // Screen coordinates (pixels)
    double z;    // Camera space Z (forward)
} ProjectedVertex;

static void ensure_z_buffer(RGBCamera* camera) {
    int size = camera->render_width * camera->render_height;
    if (size > camera->z_buffer_size) {
        if (camera->z_buffer) free(camera->z_buffer);
        camera->z_buffer = (float*)malloc(sizeof(float) * size);
        camera->z_buffer_size = size;
    }
}

static void clear_z_buffer(RGBCamera* camera) {
    int size = camera->render_width * camera->render_height;
#ifdef OPT_Z_BUFFER_EUCLIDEAN
    // Initialize to infinity for Euclidean distance (closest wins)
    float val = FLT_MAX;
    for(int i=0; i<size; i++) camera->z_buffer[i] = val;
#else
    // Clear to 0.0f (representing infinite distance in 1/z space)
    // This avoids artificial clipping at max_distance
    memset(camera->z_buffer, 0, sizeof(float) * size);
#endif
}

#define EPSILON_F 1e-5f
#define SLOPE_EPSILON 1e-9f

static void rasterize_triangle(RGBCamera* camera, ProjectedVertex v0, ProjectedVertex v1, ProjectedVertex v2, RGB color) {
    // Use double for setup to avoid precision issues (cracks/holes)
    double v0x = v0.x; double v0y = v0.y; double v0z = v0.z;
    double v1x = v1.x; double v1y = v1.y; double v1z = v1.z;
    double v2x = v2.x; double v2y = v2.y; double v2z = v2.z;

    // Bounding box
    int min_x = (int)floor(fmin(v0x, fmin(v1x, v2x)));
    int max_x = (int)ceil(fmax(v0x, fmax(v1x, v2x)));
    int min_y = (int)floor(fmin(v0y, fmin(v1y, v2y)));
    int max_y = (int)ceil(fmax(v0y, fmax(v1y, v2y)));

    // Clip to screen
    min_x = (int)fmax((double)min_x, 0.0);
    max_x = (int)fmin((double)max_x, (double)camera->render_width - 1.0);
    min_y = (int)fmax((double)min_y, 0.0);
    max_y = (int)fmin((double)max_y, (double)camera->render_height - 1.0);

    if (min_x > max_x || min_y > max_y) return;

    // Precompute denominator
    double denom = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y);
    if (fabs(denom) < EPSILON_F) return;

    double inv_denom = 1.0 / denom;
    
    // Barycentric constants
    // Using pixel centers (x + 0.5, y + 0.5)
    double w0_dx = (v1y - v2y) * inv_denom;
    double w1_dx = (v2y - v0y) * inv_denom;
    double w2_dx = -w0_dx - w1_dx;

    double w0_dy = (v2x - v1x) * inv_denom;
    double w1_dy = (v0x - v2x) * inv_denom;
    double w2_dy = -w0_dy - w1_dy;

    double w0_row_start = ((v1y - v2y) * (min_x + 0.5 - v2x) + (v2x - v1x) * (min_y + 0.5 - v2y)) * inv_denom;
    double w1_row_start = ((v2y - v0y) * (min_x + 0.5 - v2x) + (v0x - v2x) * (min_y + 0.5 - v2y)) * inv_denom;
    double w2_row_start = 1.0 - w0_row_start - w1_row_start;

    // 1/z interpolation
    double inv_z0 = 1.0 / v0z;
    double inv_z1 = 1.0 / v1z;
    double inv_z2 = 1.0 / v2z;
    
    double dz_dw0 = inv_z0 - inv_z2;
    double dz_dw1 = inv_z1 - inv_z2;
    double z_inv_base = inv_z2;
    
    double z_inv_dx = w0_dx * dz_dw0 + w1_dx * dz_dw1;
    double z_inv_dy = w0_dy * dz_dw0 + w1_dy * dz_dw1;
    
    double z_inv_row_start = w0_row_start * dz_dw0 + w1_row_start * dz_dw1 + z_inv_base;

    int row_offset = min_y * camera->render_width;
    float* z_buf_ptr = camera->z_buffer;
    uint8_t* img_data = camera->render_buffer;

    for (int y = min_y; y <= max_y; y++) {
        int x_start = min_x;
        int x_end = max_x;

#ifdef OPT_SPAN_CLIPPING
        // Span clipping: Calculate exact x range for this row
        // We solve for w >= -EPSILON_F to prevent cracks between triangles
        // We clamp the result to avoid integer overflow for large triangles
        
        double min_x_f = (double)min_x;
        double max_x_f = (double)max_x;
        
        // Clip against w0 >= -EPSILON_F
        if (w0_dx > SLOPE_EPSILON) {
            double f_lim = min_x_f + (-w0_row_start - EPSILON_F) / w0_dx;
            f_lim = fmax(min_x_f - 1.0, fmin(max_x_f + 1.0, f_lim));
            int x_lim = (int)ceil(f_lim);
            if (x_lim > x_start) x_start = x_lim;
        } else if (w0_dx < -SLOPE_EPSILON) {
            double f_lim = min_x_f + (-w0_row_start - EPSILON_F) / w0_dx;
            f_lim = fmax(min_x_f - 1.0, fmin(max_x_f + 1.0, f_lim));
            int x_lim = (int)floor(f_lim);
            if (x_lim < x_end) x_end = x_lim;
        } else if (w0_row_start < -EPSILON_F) {
            x_start = x_end + 1; // Invalid
        }

        // Clip against w1 >= -EPSILON_F
        if (w1_dx > SLOPE_EPSILON) {
            double f_lim = min_x_f + (-w1_row_start - EPSILON_F) / w1_dx;
            f_lim = fmax(min_x_f - 1.0, fmin(max_x_f + 1.0, f_lim));
            int x_lim = (int)ceil(f_lim);
            if (x_lim > x_start) x_start = x_lim;
        } else if (w1_dx < -SLOPE_EPSILON) {
            double f_lim = min_x_f + (-w1_row_start - EPSILON_F) / w1_dx;
            f_lim = fmax(min_x_f - 1.0, fmin(max_x_f + 1.0, f_lim));
            int x_lim = (int)floor(f_lim);
            if (x_lim < x_end) x_end = x_lim;
        } else if (w1_row_start < -EPSILON_F) {
            x_start = x_end + 1; // Invalid
        }

        // Clip against w2 >= -EPSILON_F
        if (w2_dx > SLOPE_EPSILON) {
            double f_lim = min_x_f + (-w2_row_start - EPSILON_F) / w2_dx;
            f_lim = fmax(min_x_f - 1.0, fmin(max_x_f + 1.0, f_lim));
            int x_lim = (int)ceil(f_lim);
            if (x_lim > x_start) x_start = x_lim;
        } else if (w2_dx < -SLOPE_EPSILON) {
            double f_lim = min_x_f + (-w2_row_start - EPSILON_F) / w2_dx;
            f_lim = fmax(min_x_f - 1.0, fmin(max_x_f + 1.0, f_lim));
            int x_lim = (int)floor(f_lim);
            if (x_lim < x_end) x_end = x_lim;
        } else if (w2_row_start < -EPSILON_F) {
            x_start = x_end + 1; // Invalid
        }
#endif

        if (x_start <= x_end) {
            // Clamp to screen bounds
            if (x_start < min_x) x_start = min_x;
            if (x_end > max_x) x_end = max_x;

            int dx_start = x_start - min_x;
            // Convert to float for inner loop
            float z_inv = (float)(z_inv_row_start + dx_start * z_inv_dx);
            float z_inv_dx_f = (float)z_inv_dx;
            
#ifndef OPT_SPAN_CLIPPING
            float w0 = (float)(w0_row_start + dx_start * w0_dx);
            float w1 = (float)(w1_row_start + dx_start * w1_dx);
            float w0_dx_f = (float)w0_dx;
            float w1_dx_f = (float)w1_dx;
#endif

            int idx = row_offset + x_start;

            for (int x = x_start; x <= x_end; x++) {
#ifndef OPT_SPAN_CLIPPING
                float w2 = 1.0f - w0 - w1;
                if (w0 >= -EPSILON_F && w1 >= -EPSILON_F && w2 >= -EPSILON_F) {
#endif
                    // Depth test
#ifdef OPT_Z_BUFFER_EUCLIDEAN
                    float z_curr = 1.0f / z_inv;
                    if (z_curr < z_buf_ptr[idx]) {
                        z_buf_ptr[idx] = z_curr;
#else
                    if (z_inv > z_buf_ptr[idx]) {
                        z_buf_ptr[idx] = z_inv;
#endif
                        // Direct pixel write
                        int p = idx * 3;
                        img_data[p] = color.r;
                        img_data[p+1] = color.g;
                        img_data[p+2] = color.b;
                    }
#ifndef OPT_SPAN_CLIPPING
                }
                w0 += w0_dx_f;
                w1 += w1_dx_f;
#endif
                z_inv += z_inv_dx_f;
                idx++;
            }
        }

        w0_row_start += w0_dy;
        w1_row_start += w1_dy;
        w2_row_start += w2_dy;
        z_inv_row_start += z_inv_dy;
        row_offset += camera->render_width;
    }
}

static void project_and_draw_triangle(RGBCamera* camera, Vec3 p0, Vec3 p1, Vec3 p2, RGB color) {
    double tan_fov_2 = tan(camera->fov / 2.0);
    double d = (camera->render_width / 2.0) / tan_fov_2;

    ProjectedVertex v[3];
    Vec3 ps[3] = {p0, p1, p2};

    for(int i=0; i<3; i++) {
        v[i].z = ps[i].x; // Camera X is forward distance
        
        double y_screen = d * ps[i].y / ps[i].x;
        double z_screen = d * ps[i].z / ps[i].x;
        
        v[i].x = (camera->render_width / 2.0) - y_screen;
        v[i].y = (camera->render_height / 2.0) - z_screen;
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

static void rasterize_box(RGBCamera* camera, Coordinates3D center, double size_x, double size_y, double size_z, RGB color, Coordinates3D cam_pos, double cos_o, double sin_o) {
    double sx = size_x / 2.0;
    double sy = size_y / 2.0;
    double sz = size_z / 2.0;
    
    Coordinates3D p0 = {center.x - sx, center.y - sy, center.z - sz};
    Coordinates3D p1 = {center.x + sx, center.y - sy, center.z - sz};
    Coordinates3D p2 = {center.x + sx, center.y + sy, center.z - sz};
    Coordinates3D p3 = {center.x - sx, center.y + sy, center.z - sz};
    Coordinates3D p4 = {center.x - sx, center.y - sy, center.z + sz};
    Coordinates3D p5 = {center.x + sx, center.y - sy, center.z + sz};
    Coordinates3D p6 = {center.x + sx, center.y + sy, center.z + sz};
    Coordinates3D p7 = {center.x - sx, center.y + sy, center.z + sz};

    // Top
    rasterize_quad(camera, to_cam(p4, cam_pos, cos_o, sin_o), to_cam(p5, cam_pos, cos_o, sin_o), to_cam(p6, cam_pos, cos_o, sin_o), to_cam(p7, cam_pos, cos_o, sin_o), color);
    // Bottom
    rasterize_quad(camera, to_cam(p0, cam_pos, cos_o, sin_o), to_cam(p3, cam_pos, cos_o, sin_o), to_cam(p2, cam_pos, cos_o, sin_o), to_cam(p1, cam_pos, cos_o, sin_o), color);
    // Front (y-sy)
    rasterize_quad(camera, to_cam(p0, cam_pos, cos_o, sin_o), to_cam(p1, cam_pos, cos_o, sin_o), to_cam(p5, cam_pos, cos_o, sin_o), to_cam(p4, cam_pos, cos_o, sin_o), color);
    // Back (y+sy)
    rasterize_quad(camera, to_cam(p2, cam_pos, cos_o, sin_o), to_cam(p3, cam_pos, cos_o, sin_o), to_cam(p7, cam_pos, cos_o, sin_o), to_cam(p6, cam_pos, cos_o, sin_o), color);
    // Left (x-sx)
    rasterize_quad(camera, to_cam(p3, cam_pos, cos_o, sin_o), to_cam(p0, cam_pos, cos_o, sin_o), to_cam(p4, cam_pos, cos_o, sin_o), to_cam(p7, cam_pos, cos_o, sin_o), color);
    // Right (x+sx)
    rasterize_quad(camera, to_cam(p1, cam_pos, cos_o, sin_o), to_cam(p2, cam_pos, cos_o, sin_o), to_cam(p6, cam_pos, cos_o, sin_o), to_cam(p5, cam_pos, cos_o, sin_o), color);
}

static void rasterize_line_segment(RGBCamera* camera, Coordinates3D p0, Coordinates3D p1, double width, RGB color, Coordinates3D cam_pos, double cos_o, double sin_o) {
    double dx = p1.x - p0.x;
    double dy = p1.y - p0.y;
    double len = sqrt(dx*dx + dy*dy);
    if (len < EPSILON) return;
    
    double nx = -dy / len * (width / 2.0);
    double ny = dx / len * (width / 2.0);
    
    Coordinates3D v0 = {p0.x + nx, p0.y + ny, p0.z};
    Coordinates3D v1 = {p1.x + nx, p1.y + ny, p1.z};
    Coordinates3D v2 = {p1.x - nx, p1.y - ny, p1.z};
    Coordinates3D v3 = {p0.x - nx, p0.y - ny, p0.z};
    
    // Top
    rasterize_quad(camera, to_cam(v0, cam_pos, cos_o, sin_o), to_cam(v1, cam_pos, cos_o, sin_o), to_cam(v2, cam_pos, cos_o, sin_o), to_cam(v3, cam_pos, cos_o, sin_o), color);

    // Bottom vertices (World Space)
    Coordinates3D v0_b = {v0.x, v0.y, v0.z - LANE_MARKING_THICKNESS};
    Coordinates3D v1_b = {v1.x, v1.y, v1.z - LANE_MARKING_THICKNESS};
    Coordinates3D v2_b = {v2.x, v2.y, v2.z - LANE_MARKING_THICKNESS};
    Coordinates3D v3_b = {v3.x, v3.y, v3.z - LANE_MARKING_THICKNESS};

    // Sides
    rasterize_quad(camera, to_cam(v0, cam_pos, cos_o, sin_o), to_cam(v1, cam_pos, cos_o, sin_o), to_cam(v1_b, cam_pos, cos_o, sin_o), to_cam(v0_b, cam_pos, cos_o, sin_o), color);
    rasterize_quad(camera, to_cam(v1, cam_pos, cos_o, sin_o), to_cam(v2, cam_pos, cos_o, sin_o), to_cam(v2_b, cam_pos, cos_o, sin_o), to_cam(v1_b, cam_pos, cos_o, sin_o), color);
    rasterize_quad(camera, to_cam(v2, cam_pos, cos_o, sin_o), to_cam(v3, cam_pos, cos_o, sin_o), to_cam(v3_b, cam_pos, cos_o, sin_o), to_cam(v2_b, cam_pos, cos_o, sin_o), color);
    rasterize_quad(camera, to_cam(v3, cam_pos, cos_o, sin_o), to_cam(v0, cam_pos, cos_o, sin_o), to_cam(v0_b, cam_pos, cos_o, sin_o), to_cam(v3_b, cam_pos, cos_o, sin_o), color);
}

static void rasterize_dashed_line(RGBCamera* camera, Coordinates3D start, Coordinates3D end, double width, RGB color, Coordinates3D cam_pos, double cos_o, double sin_o) {
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    double len = sqrt(dx*dx + dy*dy);
    if (len < EPSILON) return;
    
    double ux = dx / len;
    double uy = dy / len;
    
    double current_dist = 0;
    while (current_dist < len) {
        double seg_len = fmin(DASH_LENGTH, len - current_dist);
        Coordinates3D p0 = {start.x + ux * current_dist, start.y + uy * current_dist, start.z};
        Coordinates3D p1 = {start.x + ux * (current_dist + seg_len), start.y + uy * (current_dist + seg_len), start.z};
        
        rasterize_line_segment(camera, p0, p1, width, color, cam_pos, cos_o, sin_o);
        
        current_dist += DASH_LENGTH + DASH_GAP;
    }
}

static void render_lane_markings(RGBCamera* camera, Lane* lane, Map* map, Coordinates3D cam_pos, double cos_o, double sin_o) {
#ifdef OPT_NO_LANE_MARKINGS
    return;
#endif

    RGB white = {255, 255, 255};
    RGB yellow = {255, 255, 0};
    
    Lane* adj_left = lane_get_adjacent_left(lane, map);
    bool draw_left = false;
    bool dashed_left = false;
    RGB color_left = white;
    
    if (adj_left && adj_left->direction != lane->direction) {
        draw_left = true; color_left = yellow; dashed_left = false;
    } else if (!adj_left) {
        draw_left = true; color_left = white; dashed_left = false;
    } else {
        draw_left = true; color_left = white; dashed_left = true;
    }
    
    Lane* adj_right = lane_get_adjacent_right(lane, map);
    bool draw_right = false;
    bool dashed_right = false;
    RGB color_right = white;
    
    if (!adj_right) {
        draw_right = true; color_right = white; dashed_right = false;
    } else if (adj_right->direction == lane->direction) {
        draw_right = true; color_right = white; dashed_right = true;
    }

#ifdef OPT_NO_DASHED_LINES
    dashed_left = false;
    dashed_right = false;
#endif

    if (lane->type == LINEAR_LANE) {
        Coordinates start = lane->start_point;
        Coordinates end = lane->end_point;
        double dx = end.x - start.x;
        double dy = end.y - start.y;
        double len = sqrt(dx*dx + dy*dy);
        if (len < EPSILON) return;
        
        double ux = dx / len;
        double uy = dy / len;
        double px = -uy; 
        double py = ux;
        
        double offset_left = dashed_left ? 0.0 : LANE_MARKING_WIDTH / 2.0;
        double offset_right = dashed_right ? 0.0 : LANE_MARKING_WIDTH / 2.0;

        if (draw_left) {
            Coordinates3D p0 = {start.x + px * (lane->width/2 - offset_left), start.y + py * (lane->width/2 - offset_left), LANE_MARKING_Z};
            Coordinates3D p1 = {end.x + px * (lane->width/2 - offset_left), end.y + py * (lane->width/2 - offset_left), LANE_MARKING_Z};
            if (dashed_left) rasterize_dashed_line(camera, p0, p1, LANE_MARKING_WIDTH, color_left, cam_pos, cos_o, sin_o);
            else rasterize_line_segment(camera, p0, p1, LANE_MARKING_WIDTH, color_left, cam_pos, cos_o, sin_o);
        }
        
        if (draw_right) {
            Coordinates3D p0 = {start.x - px * (lane->width/2 - offset_right), start.y - py * (lane->width/2 - offset_right), LANE_MARKING_Z};
            Coordinates3D p1 = {end.x - px * (lane->width/2 - offset_right), end.y - py * (lane->width/2 - offset_right), LANE_MARKING_Z};
            if (dashed_right) rasterize_dashed_line(camera, p0, p1, LANE_MARKING_WIDTH, color_right, cam_pos, cos_o, sin_o);
            else rasterize_line_segment(camera, p0, p1, LANE_MARKING_WIDTH, color_right, cam_pos, cos_o, sin_o);
        }
    } else if (lane->type == QUARTER_ARC_LANE) {
        double r_left, r_right;
        double offset_left = dashed_left ? 0.0 : LANE_MARKING_WIDTH / 2.0;
        double offset_right = dashed_right ? 0.0 : LANE_MARKING_WIDTH / 2.0;

        if (lane->direction == DIRECTION_CW) {
            r_left = lane->radius + lane->width/2 - offset_left;
            r_right = lane->radius - lane->width/2 + offset_right;
        } else {
            r_left = lane->radius - lane->width/2 + offset_left;
            r_right = lane->radius + lane->width/2 - offset_right;
        }
        
#ifdef OPT_SIMPLE_LANE_MARKINGS
        int segments = TURN_SEGMENTS;
#else
        int segments = TURN_SEGMENTS * 2;
#endif

        // Calculate offsets for smooth interpolation (fix for asymmetric intersections)
        Vec2D theoretical_start = {
            lane->center.x + lane->radius * cos(lane->start_angle),
            lane->center.y + lane->radius * sin(lane->start_angle)
        };
        Vec2D theoretical_end = {
            lane->center.x + lane->radius * cos(lane->end_angle),
            lane->center.y + lane->radius * sin(lane->end_angle)
        };
        
        Lane* incoming = lane_get_connection_incoming_straight(lane, map);
        Lane* outgoing = lane_get_connection_straight(lane, map);
        
        Vec2D actual_start = incoming ? incoming->end_point : lane->start_point;
        Vec2D actual_end = outgoing ? outgoing->start_point : lane->end_point;
        
        Vec2D offset_start = vec_sub(actual_start, theoretical_start);
        Vec2D offset_end = vec_sub(actual_end, theoretical_end);

        double step = (lane->end_angle - lane->start_angle) / segments;
        
        if (draw_left) {
             double dist_accum = 0;
             bool drawing = true;
             for(int i=0; i<segments; i++) {
                double a1 = lane->start_angle + i * step;
                double a2 = lane->start_angle + (i+1) * step;
                double seg_len = fabs(step * r_left);

                double p1_ratio = (double)i / segments;
                double p2_ratio = (double)(i+1) / segments;
                Vec2D off1 = vec_add(vec_scale(offset_start, 1.0 - p1_ratio), vec_scale(offset_end, p1_ratio));
                Vec2D off2 = vec_add(vec_scale(offset_start, 1.0 - p2_ratio), vec_scale(offset_end, p2_ratio));
                
                if (!dashed_left || drawing) {
                    Coordinates3D p0 = {lane->center.x + r_left * cos(a1) + off1.x, lane->center.y + r_left * sin(a1) + off1.y, LANE_MARKING_Z};
                    Coordinates3D p1 = {lane->center.x + r_left * cos(a2) + off2.x, lane->center.y + r_left * sin(a2) + off2.y, LANE_MARKING_Z};
                    rasterize_line_segment(camera, p0, p1, LANE_MARKING_WIDTH, color_left, cam_pos, cos_o, sin_o);
                }
                
                if (dashed_left) {
                    dist_accum += seg_len;
                    if (drawing && dist_accum >= DASH_LENGTH) { drawing = false; dist_accum = 0; }
                    else if (!drawing && dist_accum >= DASH_GAP) { drawing = true; dist_accum = 0; }
                }
             }
        }
        
        if (draw_right) {
             double dist_accum = 0;
             bool drawing = true;
             for(int i=0; i<segments; i++) {
                double a1 = lane->start_angle + i * step;
                double a2 = lane->start_angle + (i+1) * step;
                double seg_len = fabs(step * r_right);

                double p1_ratio = (double)i / segments;
                double p2_ratio = (double)(i+1) / segments;
                Vec2D off1 = vec_add(vec_scale(offset_start, 1.0 - p1_ratio), vec_scale(offset_end, p1_ratio));
                Vec2D off2 = vec_add(vec_scale(offset_start, 1.0 - p2_ratio), vec_scale(offset_end, p2_ratio));
                
                if (!dashed_right || drawing) {
                    Coordinates3D p0 = {lane->center.x + r_right * cos(a1) + off1.x, lane->center.y + r_right * sin(a1) + off1.y, LANE_MARKING_Z};
                    Coordinates3D p1 = {lane->center.x + r_right * cos(a2) + off2.x, lane->center.y + r_right * sin(a2) + off2.y, LANE_MARKING_Z};
                    rasterize_line_segment(camera, p0, p1, LANE_MARKING_WIDTH, color_right, cam_pos, cos_o, sin_o);
                }
                
                if (dashed_right) {
                    dist_accum += seg_len;
                    if (drawing && dist_accum >= DASH_LENGTH) { drawing = false; dist_accum = 0; }
                    else if (!drawing && dist_accum >= DASH_GAP) { drawing = true; dist_accum = 0; }
                }
             }
        }
    }
}

static void draw_pixel_safe(RGBCamera* camera, int x, int y, RGB color) {
    if (x >= 0 && x < camera->render_width && y >= 0 && y < camera->render_height) {
        int index = y * (camera->render_width * 3) + x * 3;
        camera->render_buffer[index + 0] = color.r;
        camera->render_buffer[index + 1] = color.g;
        camera->render_buffer[index + 2] = color.b;
    }
}

static void draw_line_2d(RGBCamera* camera, int x0, int y0, int x1, int y1, RGB color) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    while (1) {
        draw_pixel_safe(camera, x0, y0, color);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

static void draw_char_scaled(RGBCamera* camera, int x, int y, char c, RGB color, int scale) {
    // 5x5 bitmaps
    static const uint8_t font[][5] = {
        {0x11, 0x19, 0x15, 0x13, 0x11}, // N
        {0x1F, 0x10, 0x1F, 0x10, 0x1F}, // E
        {0x0E, 0x10, 0x0E, 0x01, 0x0E}, // S
        {0x11, 0x11, 0x15, 0x15, 0x0A}  // W
    };
    
    int idx = -1;
    if (c == 'N') idx = 0;
    else if (c == 'E') idx = 1;
    else if (c == 'S') idx = 2;
    else if (c == 'W') idx = 3;
    
    if (idx == -1) return;
    
    for(int r=0; r<5; r++) {
        for(int c=0; c<5; c++) {
            if ((font[idx][r] >> (4-c)) & 1) {
                for(int dy=0; dy<scale; dy++) {
                    for(int dx=0; dx<scale; dx++) {
                        draw_pixel_safe(camera, x + c*scale + dx, y + r*scale + dy, color);
                    }
                }
            }
        }
    }
}

static void draw_hud(RGBCamera* camera) {
    // Scale HUD elements based on render resolution vs base resolution (assuming base is width/height)
    // But wait, width/height are just the output dimensions.
    // If we are in SSAA mode, render_width > width.
    // We want the HUD to be drawn at high resolution so it gets anti-aliased too.
    
    int scale = camera->render_width / camera->width;
    if (scale < 1) scale = 1;

    int hud_height = HUD_HEIGHT * scale;
    RGB tick_color = HUD_TICK_RGB;
    RGB text_color = HUD_TEXT_RGB;
    RGB north_color = HUD_NORTH_RGB;

    // Background strip
    for(int y=0; y<hud_height; y++) {
        for(int x=0; x<camera->render_width; x++) {
            int idx = (y * camera->render_width + x) * 3;
            camera->render_buffer[idx+0] = camera->render_buffer[idx+0] / 2;
            camera->render_buffer[idx+1] = camera->render_buffer[idx+1] / 2;
            camera->render_buffer[idx+2] = camera->render_buffer[idx+2] / 2;
        }
    }

    double fov = camera->fov;
    double center_x = camera->render_width / 2.0;
    double pixels_per_rad = camera->render_width / fov;
    
    double orientation = fmod(camera->orientation, 2*M_PI);
    if (orientation < 0) orientation += 2*M_PI;

    double step = M_PI / 12.0; // 15 degrees

    for (int i = 0; i < 24; i++) {
        double angle = i * step;
        
        double diff = angle - orientation;
        while (diff <= -M_PI) diff += 2*M_PI;
        while (diff > M_PI) diff -= 2*M_PI;
        
        if (fabs(diff) < fov / 2.0) {
            int x = (int)(center_x - diff * pixels_per_rad);
            
            if (i % 6 == 0) {
                // Cardinal (N, E, S, W)
                char label = '?';
                if (i == 0) label = 'E';
                else if (i == 6) label = 'N';
                else if (i == 12) label = 'W';
                else if (i == 18) label = 'S';
                
                RGB c = (label == 'N') ? north_color : text_color;
                draw_char_scaled(camera, x - 5*scale, 5*scale, label, c, HUD_TEXT_SCALE * scale);
            } else if (i % 3 == 0) {
                // Intercardinal (NE, SE, SW, NW) -> Pipe
                draw_line_2d(camera, x, hud_height - 12*scale, x, hud_height - 4*scale, tick_color);
                draw_line_2d(camera, x+1, hud_height - 12*scale, x+1, hud_height - 4*scale, tick_color);
            } else {
                // Minor tick -> Period
                draw_line_2d(camera, x, hud_height - 6*scale, x, hud_height - 4*scale, tick_color);
                draw_line_2d(camera, x+1, hud_height - 6*scale, x+1, hud_height - 4*scale, tick_color);
            }
        }
    }
    
    // Center marker (Needle)
    draw_line_2d(camera, (int)center_x, hud_height - 10*scale, (int)center_x, hud_height, HUD_NEEDLE_RGB);
}

static void downsample(RGBCamera* src, RGBCamera* dest, int scale) {
    for (int y = 0; y < dest->height; y++) {
        for (int x = 0; x < dest->width; x++) {
            int r_sum = 0, g_sum = 0, b_sum = 0;
            for (int dy = 0; dy < scale; dy++) {
                for (int dx = 0; dx < scale; dx++) {
                    int sy = y * scale + dy;
                    int sx = x * scale + dx;
                    int idx = (sy * src->width + sx) * 3;
                    r_sum += src->data[idx + 0];
                    g_sum += src->data[idx + 1];
                    b_sum += src->data[idx + 2];
                }
            }
            int count = scale * scale;
            int idx = (y * dest->width + x) * 3;
            dest->data[idx + 0] = (uint8_t)(r_sum / count);
            dest->data[idx + 1] = (uint8_t)(g_sum / count);
            dest->data[idx + 2] = (uint8_t)(b_sum / count);
        }
    }
}


static void fill_background(uint8_t* buffer, int width, int height) {
    RGB sky_top = SKY_TOP_RGB;
    RGB sky_horizon = SKY_HORIZON_RGB;
    RGB ground_color = GROUND_RGB;
    
    uint8_t* ptr = buffer;
    int half_height = height / 2;
    
    // Sky (Top half)
    for (int y = 0; y < half_height; y++) {
        double t = (double)y / (double)half_height; // 0 at top, 1 at horizon
        if (t > 1.0) t = 1.0;
        uint8_t r = (uint8_t)(sky_top.r * (1.0 - t) + sky_horizon.r * t);
        uint8_t g = (uint8_t)(sky_top.g * (1.0 - t) + sky_horizon.g * t);
        uint8_t b = (uint8_t)(sky_top.b * (1.0 - t) + sky_horizon.b * t);
        
        for (int x = 0; x < width; x++) {
            *ptr++ = r;
            *ptr++ = g;
            *ptr++ = b;
        }
    }
    
    // Ground (Bottom half)
    uint8_t gr = ground_color.r;
    uint8_t gg = ground_color.g;
    uint8_t gb = ground_color.b;
    int pixels_remaining = (height - half_height) * width;
    for (int i = 0; i < pixels_remaining; i++) {
        *ptr++ = gr;
        *ptr++ = gg;
        *ptr++ = gb;
    }
}

static void render_scene_internal(RGBCamera* camera, Simulation* sim, void** exclude_objects) {
    BENCH_START(bench_time_sky);
    // Copy pre-computed background (sky + ground)
    if (camera->background_buffer) {
        memcpy(camera->render_buffer, camera->background_buffer, sizeof(uint8_t) * 3 * camera->render_width * camera->render_height);
    } else {
        // Fallback if background buffer is missing (e.g. SSAA high-res camera)
        fill_background(camera->render_buffer, camera->render_width, camera->render_height);
    }
    BENCH_END(bench_time_sky);

    ensure_z_buffer(camera);
    clear_z_buffer(camera);

    double cos_o = cos(camera->orientation);
    double sin_o = sin(camera->orientation);
    Coordinates3D cam_pos = { camera->position.x, camera->position.y, camera->z_altitude };

    // 0. Ground
    BENCH_START(bench_time_ground);
    // Ground already drawn via background copy
    BENCH_END(bench_time_ground);

    // 1. Cars
    BENCH_START(bench_time_cars);
    for (CarId car_id = 0; car_id < sim_get_num_cars(sim); car_id++) {
        Car* car = sim_get_car(sim, car_id);
        if (object_array_contains_object(exclude_objects, car)) continue;

        Meters center_distance = vec_distance(camera->position, car->center);
        if (center_distance > camera->max_distance + CAR_RENDER_DIST_OFFSET) continue;

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
        colors[0] = CAR_BODY_RGB; 
        // Top
        colors[1] = CAR_BODY_RGB;
        // Front (Headlights)
        colors[2] = CAR_BODY_RGB;
        // Left
        colors[3] = CAR_BODY_RGB;
        // Back
        RGB taillight_color;
        {
            bool is_braking = (car->speed > 0 && car->acceleration < -0.1) || (car->speed < 0 && car->acceleration > 0.1);
            taillight_color = is_braking ? CAR_BRAKE_RGB : CAR_TAILLIGHT_RGB;
            colors[4] = CAR_BODY_RGB;
        }
        // Right
        colors[5] = CAR_SIDE_RIGHT_RGB;

        // Rasterize sides
        rasterize_quad(camera, vb[0], vb[3], vb[2], vb[1], colors[0]); // Bottom
        rasterize_quad(camera, vt[0], vt[1], vt[2], vt[3], colors[1]); // Top
        rasterize_quad(camera, vb[0], vb[1], vt[1], vt[0], colors[2]); // Front
        rasterize_quad(camera, vb[1], vb[2], vt[2], vt[1], colors[3]); // Left
        rasterize_quad(camera, vb[2], vb[3], vt[3], vt[2], colors[4]); // Back
        rasterize_quad(camera, vb[3], vb[0], vt[0], vt[3], colors[5]); // Right

#ifdef OPT_SIMPLE_CARS
        continue;
#endif

#ifndef OPT_NO_CAR_LIGHTS
        // Calculate normals for lights (extrusion)
        #define V3_SUB(a, b) (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}
        #define V3_CROSS(a, b) (Vec3){ \
            a.y * b.z - a.z * b.y, \
            a.z * b.x - a.x * b.z, \
            a.x * b.y - a.y * b.x \
        }
        #define V3_LEN(v) sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
        #define V3_NORM(v) do { \
            double _l = V3_LEN(v); \
            if (_l > 1e-9) { v.x /= _l; v.y /= _l; v.z /= _l; } \
        } while(0)

        Vec3 n_front, n_back, n_left, n_right;
        {
            // Front: vb[0], vb[1], vt[1], vt[0] (Right to Left)
            Vec3 v_right_to_left = V3_SUB(vb[1], vb[0]);
            Vec3 v_up = V3_SUB(vt[0], vb[0]);
            Vec3 c = V3_CROSS(v_right_to_left, v_up);
            V3_NORM(c);
            n_front = c;
        }
        {
            // Back: vb[2], vb[3], vt[3], vt[2] (Left to Right)
            Vec3 v_left_to_right = V3_SUB(vb[3], vb[2]);
            Vec3 v_up = V3_SUB(vt[2], vb[2]);
            Vec3 c = V3_CROSS(v_left_to_right, v_up);
            V3_NORM(c);
            n_back = c;
        }
        {
            // Left: vb[1], vb[2], vt[2], vt[1] (Front to Back)
            Vec3 v_front_to_back = V3_SUB(vb[2], vb[1]);
            Vec3 v_up = V3_SUB(vt[1], vb[1]);
            Vec3 c = V3_CROSS(v_front_to_back, v_up);
            V3_NORM(c);
            n_left = c;
        }
        {
            // Right: vb[3], vb[0], vt[0], vt[3] (Back to Front)
            Vec3 v_back_to_front = V3_SUB(vb[0], vb[3]);
            Vec3 v_up = V3_SUB(vt[3], vb[3]);
            Vec3 c = V3_CROSS(v_back_to_front, v_up);
            V3_NORM(c);
            n_right = c;
        }

#ifdef OPT_SIMPLE_LIGHTS
        #define RASTERIZE_THICK_QUAD(p0, p1, p2, p3, normal, thickness, color) \
            rasterize_quad(camera, p0, p1, p2, p3, color)
#else
        #define RASTERIZE_THICK_QUAD(p0, p1, p2, p3, normal, thickness, color) do { \
            Vec3 _n = normal; \
            double _t = thickness; \
            Vec3 _q0 = {p0.x + _n.x*_t, p0.y + _n.y*_t, p0.z + _n.z*_t}; \
            Vec3 _q1 = {p1.x + _n.x*_t, p1.y + _n.y*_t, p1.z + _n.z*_t}; \
            Vec3 _q2 = {p2.x + _n.x*_t, p2.y + _n.y*_t, p2.z + _n.z*_t}; \
            Vec3 _q3 = {p3.x + _n.x*_t, p3.y + _n.y*_t, p3.z + _n.z*_t}; \
            rasterize_quad(camera, _q0, _q1, _q2, _q3, color); \
            rasterize_quad(camera, p0, p1, _q1, _q0, color); \
            rasterize_quad(camera, p1, p2, _q2, _q1, color); \
            rasterize_quad(camera, p2, p3, _q3, _q2, color); \
            rasterize_quad(camera, p3, p0, _q0, _q3, color); \
        } while(0)
#endif

        double light_thickness = 0.05;

        // Render Tail Lights (on top of Back)
        {
            Vec3 bl = vb[2];
            Vec3 br = vb[3];
            Vec3 tl = vt[2];
            Vec3 tr = vt[3];

            // Helper macro for interpolation
            #define LERP_V3(a, b, t) (Vec3){ \
                a.x + (b.x - a.x) * t, \
                a.y + (b.y - a.y) * t, \
                a.z + (b.z - a.z) * t \
            }
            
            // Bilinear interpolation
            #define POINT_ON_BACK(u, v) LERP_V3( \
                LERP_V3(bl, br, u), \
                LERP_V3(tl, tr, u), \
                v \
            )

            // Left Light
            Vec3 ll_p0 = POINT_ON_BACK(0.05, 0.5);
            Vec3 ll_p1 = POINT_ON_BACK(0.25, 0.5);
            Vec3 ll_p2 = POINT_ON_BACK(0.25, 0.8);
            Vec3 ll_p3 = POINT_ON_BACK(0.05, 0.8);
            RASTERIZE_THICK_QUAD(ll_p0, ll_p1, ll_p2, ll_p3, n_back, light_thickness, taillight_color);

            // Right Light
            Vec3 rl_p0 = POINT_ON_BACK(0.75, 0.5);
            Vec3 rl_p1 = POINT_ON_BACK(0.95, 0.5);
            Vec3 rl_p2 = POINT_ON_BACK(0.95, 0.8);
            Vec3 rl_p3 = POINT_ON_BACK(0.75, 0.8);
            RASTERIZE_THICK_QUAD(rl_p0, rl_p1, rl_p2, rl_p3, n_back, light_thickness, taillight_color);
            
            #undef POINT_ON_BACK
            #undef LERP_V3
        }

        // Render Headlights (on top of Front)
        {
            Vec3 bl = vb[0];
            Vec3 br = vb[1];
            Vec3 tl = vt[0];
            Vec3 tr = vt[1];

            #define LERP_V3(a, b, t) (Vec3){ \
                a.x + (b.x - a.x) * t, \
                a.y + (b.y - a.y) * t, \
                a.z + (b.z - a.z) * t \
            }
            
            #define POINT_ON_FACE(u, v) LERP_V3( \
                LERP_V3(bl, br, u), \
                LERP_V3(tl, tr, u), \
                v \
            )

            // Left Headlight
            Vec3 ll_p0 = POINT_ON_FACE(0.05, 0.5);
            Vec3 ll_p1 = POINT_ON_FACE(0.25, 0.5);
            Vec3 ll_p2 = POINT_ON_FACE(0.25, 0.8);
            Vec3 ll_p3 = POINT_ON_FACE(0.05, 0.8);
            RASTERIZE_THICK_QUAD(ll_p0, ll_p1, ll_p2, ll_p3, n_front, light_thickness, CAR_HEADLIGHT_RGB);

            // Right Headlight
            Vec3 rl_p0 = POINT_ON_FACE(0.75, 0.5);
            Vec3 rl_p1 = POINT_ON_FACE(0.95, 0.5);
            Vec3 rl_p2 = POINT_ON_FACE(0.95, 0.8);
            Vec3 rl_p3 = POINT_ON_FACE(0.75, 0.8);
            RASTERIZE_THICK_QUAD(rl_p0, rl_p1, rl_p2, rl_p3, n_front, light_thickness, CAR_HEADLIGHT_RGB);
            
            #undef POINT_ON_FACE
            #undef LERP_V3
        }

        // Render Indicators
        {

            bool left_on = (car->indicator_turn == INDICATOR_LEFT || car->indicator_lane == INDICATOR_LEFT);
            bool right_on = (car->indicator_turn == INDICATOR_RIGHT || car->indicator_lane == INDICATOR_RIGHT);
            
            if (left_on || right_on) {
                RGB ind_color_left = (car->indicator_turn == INDICATOR_LEFT) ? CAR_INDICATOR_RED_RGB : CAR_INDICATOR_ORANGE_RGB;
                RGB ind_color_right = (car->indicator_turn == INDICATOR_RIGHT) ? CAR_INDICATOR_RED_RGB : CAR_INDICATOR_ORANGE_RGB;
                
                #define LERP_V3(a, b, t) (Vec3){ \
                    a.x + (b.x - a.x) * t, \
                    a.y + (b.y - a.y) * t, \
                    a.z + (b.z - a.z) * t \
                }
                #define POINT_ON_QUAD(bl, br, tl, tr, u, v) LERP_V3( \
                    LERP_V3(bl, br, u), \
                    LERP_V3(tl, tr, u), \
                    v \
                )

                if (left_on) {
                    // 1. Back (Left side is u near 0)
                    {
                        Vec3 bl = vb[2], br = vb[3], tl = vt[2], tr = vt[3];
                        Vec3 p0 = POINT_ON_QUAD(bl, br, tl, tr, 0.0, 0.35);
                        Vec3 p1 = POINT_ON_QUAD(bl, br, tl, tr, 0.2, 0.35);
                        Vec3 p2 = POINT_ON_QUAD(bl, br, tl, tr, 0.2, 0.45);
                        Vec3 p3 = POINT_ON_QUAD(bl, br, tl, tr, 0.0, 0.45);
                        RASTERIZE_THICK_QUAD(p0, p1, p2, p3, n_back, light_thickness, ind_color_left);
                    }
                    // 2. Front (Left side is u near 1)
                    {
                        Vec3 bl = vb[0], br = vb[1], tl = vt[0], tr = vt[1];
                        Vec3 p0 = POINT_ON_QUAD(bl, br, tl, tr, 0.8, 0.35);
                        Vec3 p1 = POINT_ON_QUAD(bl, br, tl, tr, 1.0, 0.35);
                        Vec3 p2 = POINT_ON_QUAD(bl, br, tl, tr, 1.0, 0.45);
                        Vec3 p3 = POINT_ON_QUAD(bl, br, tl, tr, 0.8, 0.45);
                        RASTERIZE_THICK_QUAD(p0, p1, p2, p3, n_front, light_thickness, ind_color_left);
                    }
                    // 3. Side (Left Face: b1, b2, t2, t1. u=0 is Front)
                    {
                        Vec3 bl = vb[1], br = vb[2], tl = vt[1], tr = vt[2];
                        Vec3 p0 = POINT_ON_QUAD(bl, br, tl, tr, 0.1, 0.4);
                        Vec3 p1 = POINT_ON_QUAD(bl, br, tl, tr, 0.2, 0.4);
                        Vec3 p2 = POINT_ON_QUAD(bl, br, tl, tr, 0.2, 0.5);
                        Vec3 p3 = POINT_ON_QUAD(bl, br, tl, tr, 0.1, 0.5);
                        RASTERIZE_THICK_QUAD(p0, p1, p2, p3, n_left, light_thickness, ind_color_left);
                    }
                }

                if (right_on) {
                    // 1. Back (Right side is u near 1)
                    {
                        Vec3 bl = vb[2], br = vb[3], tl = vt[2], tr = vt[3];
                        Vec3 p0 = POINT_ON_QUAD(bl, br, tl, tr, 0.8, 0.35);
                        Vec3 p1 = POINT_ON_QUAD(bl, br, tl, tr, 1.0, 0.35);
                        Vec3 p2 = POINT_ON_QUAD(bl, br, tl, tr, 1.0, 0.45);
                        Vec3 p3 = POINT_ON_QUAD(bl, br, tl, tr, 0.8, 0.45);
                        RASTERIZE_THICK_QUAD(p0, p1, p2, p3, n_back, light_thickness, ind_color_right);
                    }
                    // 2. Front (Right side is u near 0)
                    {
                        Vec3 bl = vb[0], br = vb[1], tl = vt[0], tr = vt[1];
                        Vec3 p0 = POINT_ON_QUAD(bl, br, tl, tr, 0.0, 0.35);
                        Vec3 p1 = POINT_ON_QUAD(bl, br, tl, tr, 0.2, 0.35);
                        Vec3 p2 = POINT_ON_QUAD(bl, br, tl, tr, 0.2, 0.45);
                        Vec3 p3 = POINT_ON_QUAD(bl, br, tl, tr, 0.0, 0.45);
                        RASTERIZE_THICK_QUAD(p0, p1, p2, p3, n_front, light_thickness, ind_color_right);
                    }
                    // 3. Side (Right Face: b3, b0, t0, t3. u=0 is Rear, u=1 is Front)
                    {
                        Vec3 bl = vb[3], br = vb[0], tl = vt[3], tr = vt[0];
                        Vec3 p0 = POINT_ON_QUAD(bl, br, tl, tr, 0.8, 0.4);
                        Vec3 p1 = POINT_ON_QUAD(bl, br, tl, tr, 0.9, 0.4);
                        Vec3 p2 = POINT_ON_QUAD(bl, br, tl, tr, 0.9, 0.5);
                        Vec3 p3 = POINT_ON_QUAD(bl, br, tl, tr, 0.8, 0.5);
                        RASTERIZE_THICK_QUAD(p0, p1, p2, p3, n_right, light_thickness, ind_color_right);
                    }
                }
                
                #undef POINT_ON_QUAD
                #undef LERP_V3
            }
        }
        
        #undef V3_SUB
        #undef V3_CROSS
        #undef V3_LEN
        #undef V3_NORM
        #undef RASTERIZE_THICK_QUAD
#endif
    }
    BENCH_END(bench_time_cars);

    // 2. Roads
    BENCH_START(bench_time_roads);
    Map* map = sim_get_map(sim);
    for (RoadId road_id = 0; road_id < map_get_num_roads(map); road_id++) {
        Road* road = map_get_road(map, road_id);
        if (object_array_contains_object(exclude_objects, road)) continue;

        if (road_get_type(road) == STRAIGHT) {
            Coordinates road_center = road_get_center(road);
            
            // Pruning: Check if road is within max_distance
            // Approximate road extent as length/2
            double road_len = road_get_length(road);
            if (vec_distance(camera->position, road_center) > camera->max_distance + road_len / 2.0) continue;

            Meters delta_x, delta_y;
            bool is_horizontal = (road_get_direction(road) == DIRECTION_EAST || road_get_direction(road) == DIRECTION_WEST);
            if (is_horizontal) {
                delta_x = road_len / 2.0;
                delta_y = road_get_width(road) / 2.0;
            } else {
                delta_x = road_get_width(road) / 2.0;
                delta_y = road_len / 2.0;
            }
            Coordinates3D r0 = { road_center.x - delta_x, road_center.y - delta_y, 0.0 };
            Coordinates3D r1 = { road_center.x + delta_x, road_center.y - delta_y, 0.0 };
            Coordinates3D r2 = { road_center.x + delta_x, road_center.y + delta_y, 0.0 };
            Coordinates3D r3 = { road_center.x - delta_x, road_center.y + delta_y, 0.0 };

            RGB color = (road_get_direction(road) == DIRECTION_WEST || road_get_direction(road) == DIRECTION_SOUTH) ? 
                        ROAD_COLOR_DARK_RGB : ROAD_COLOR_LIGHT_RGB;
            
            rasterize_thick_quad(camera, r0, r1, r2, r3, color, cam_pos, cos_o, sin_o);

        } else if (road_get_type(road) == TURN) {
            Coordinates center = road_get_center(road);
            Meters radius = road->radius;
            Meters width = road_get_width(road);
            
            // Pruning: Check if turn is within max_distance
            // Approximate turn extent as radius + width/2
            if (vec_distance(camera->position, center) > camera->max_distance + radius + width / 2.0) continue;

            Meters r_out = radius + width / 2.0;
            Meters r_in = radius - width / 2.0;
            Quadrant quad = road->quadrant;

            double start_angle, end_angle;
            if (quad == QUADRANT_TOP_RIGHT) { start_angle = 0; end_angle = M_PI_2; }
            else if (quad == QUADRANT_TOP_LEFT) { start_angle = M_PI_2; end_angle = M_PI; }
            else if (quad == QUADRANT_BOTTOM_LEFT) { start_angle = M_PI; end_angle = 3*M_PI_2; }
            else { start_angle = 3*M_PI_2; end_angle = 2*M_PI; }

            RGB color = (road_get_direction(road) == DIRECTION_CW) ? ROAD_COLOR_DARK_RGB : ROAD_COLOR_LIGHT_RGB;

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

        // Render Lanes
        for (int i = 0; i < road->num_lanes; i++) {
            Lane* lane = map_get_lane(map, road->lane_ids[i]);
            render_lane_markings(camera, lane, map, cam_pos, cos_o, sin_o);
        }
    }
    BENCH_END(bench_time_roads);

    // 3. Intersections
    BENCH_START(bench_time_intersections);
    for (IntersectionId intersection_id = 0; intersection_id < map_get_num_intersections(map); intersection_id++) {
        Intersection* intersection = map_get_intersection(map, intersection_id);
        if (object_array_contains_object(exclude_objects, intersection)) continue;

        Coordinates intersection_center = intersection->center;
        Dimensions dims = intersection->dimensions;
        Coordinates3D i0 = { intersection_center.x - dims.x / 2.0, intersection_center.y - dims.y / 2.0, 0.0 };
        Coordinates3D i1 = { intersection_center.x + dims.x / 2.0, intersection_center.y - dims.y / 2.0, 0.0 };
        Coordinates3D i2 = { intersection_center.x + dims.x / 2.0, intersection_center.y + dims.y / 2.0, 0.0 };
        Coordinates3D i3 = { intersection_center.x - dims.x / 2.0, intersection_center.y + dims.y / 2.0, 0.0 };

        rasterize_thick_quad(camera, i0, i1, i2, i3, INTERSECTION_RGB, cam_pos, cos_o, sin_o);

#ifndef OPT_SIMPLE_INTERSECTIONS
        // Intersection Edges
        RGB edge_color = INTERSECTION_EDGE_RGB;
        double edge_width = INTERSECTION_EDGE_WIDTH;
        double half_w = edge_width / 2.0;

        // Bottom (y is min)
        Coordinates3D b0 = {i0.x, i0.y + half_w, LANE_MARKING_Z};
        Coordinates3D b1 = {i1.x, i1.y + half_w, LANE_MARKING_Z};
        rasterize_line_segment(camera, b0, b1, edge_width, edge_color, cam_pos, cos_o, sin_o);

        // Right (x is max)
        Coordinates3D r0 = {i1.x - half_w, i1.y, LANE_MARKING_Z};
        Coordinates3D r1 = {i2.x - half_w, i2.y, LANE_MARKING_Z};
        rasterize_line_segment(camera, r0, r1, edge_width, edge_color, cam_pos, cos_o, sin_o);

        // Top (y is max)
        Coordinates3D t0 = {i2.x, i2.y - half_w, LANE_MARKING_Z};
        Coordinates3D t1 = {i3.x, i3.y - half_w, LANE_MARKING_Z};
        rasterize_line_segment(camera, t0, t1, edge_width, edge_color, cam_pos, cos_o, sin_o);

        // Left (x is min)
        Coordinates3D l0 = {i3.x + half_w, i3.y, LANE_MARKING_Z};
        Coordinates3D l1 = {i0.x + half_w, i0.y, LANE_MARKING_Z};
        rasterize_line_segment(camera, l0, l1, edge_width, edge_color, cam_pos, cos_o, sin_o);

        // Traffic Light Pole
        double pole_height = TRAFFIC_LIGHT_HEIGHT - TRAFFIC_LIGHT_SIZE / 2.0;
        Coordinates3D pole_center = { intersection_center.x, intersection_center.y, pole_height / 2.0 };
        rasterize_box(camera, pole_center, TL_POLE_WIDTH, TL_POLE_WIDTH, pole_height, TL_POLE_RGB, cam_pos, cos_o, sin_o);

        // Traffic Light
        Coordinates3D tl_center = { intersection_center.x, intersection_center.y, TRAFFIC_LIGHT_HEIGHT };
        RGB c_ns = {0, 0, 0};
        RGB c_ew = {0, 0, 0};
        
        IntersectionState state = intersection_get_state(intersection);
        if (state == NS_GREEN_EW_RED) {
            c_ns = TL_GREEN_RGB; // Green
            c_ew = TL_RED_RGB; // Red
        } else if (state == NS_YELLOW_EW_RED) {
            c_ns = TL_YELLOW_RGB; // Yellow
            c_ew = TL_RED_RGB; // Red
        } else if (state == NS_RED_EW_GREEN) {
            c_ns = TL_RED_RGB; // Red
            c_ew = TL_GREEN_RGB; // Green
        } else if (state == EW_YELLOW_NS_RED) {
            c_ns = TL_RED_RGB; // Red
            c_ew = TL_YELLOW_RGB; // Yellow
        } else if (state == ALL_RED_BEFORE_NS_GREEN || state == ALL_RED_BEFORE_EW_GREEN || state == FOUR_WAY_STOP) {
            c_ns = TL_RED_RGB; // Red
            c_ew = TL_RED_RGB; // Red
        } else if (state == T_JUNC_NS || state == T_JUNC_EW) {
            // For T-junctions, we can set all to green
            c_ns = TL_GREEN_RGB;
            c_ew = TL_GREEN_RGB;
        }

        // Cube faces: Top, Bottom, Front (S), Back (N), Left (W), Right (E)
        // Note: Front is -Y (South), Back is +Y (North), Left is -X (West), Right is +X (East)
        // So Front/Back are NS facing, Left/Right are EW facing.
        rasterize_cube(camera, tl_center, TRAFFIC_LIGHT_SIZE, 
            TL_BODY_RGB, TL_BODY_RGB, // Top, Bottom (Dark Grey)
            c_ns, c_ns, // Front (S), Back (N) -> NS lights
            c_ew, c_ew, // Left (W), Right (E) -> EW lights
            cam_pos, cos_o, sin_o);
#endif
    }
    BENCH_END(bench_time_intersections);
}

void rgbcam_capture(RGBCamera* camera, Simulation* sim) {
    void* exclude_objects[2] = { NULL, NULL };
    if (camera->attached_car != NULL) {
        exclude_objects[0] = (void*)camera->attached_car;
    }
    rgbcam_capture_exclude_objects(camera, sim, exclude_objects);
}

void rgbcam_capture_exclude_objects(RGBCamera* camera, Simulation* sim, void** exclude_objects) {
    BENCH_START(bench_time_total);
    // here, if the camera is attached to a car, we need to update the camera's position and orientation based on the car's current state
    if (camera->attached_car != NULL) {
        Car* car = camera->attached_car;
        double length = car->dimensions.y;
        double width = car->dimensions.x;
        double height = car->dimensions.z;
        
        double local_x = 0;
        double local_y = 0;
        double local_z = 0;
        double local_yaw = 0;

        switch (camera->attached_car_camera_type) {
            case CAR_CAMERA_MAIN_FORWARD:
            case CAR_CAMERA_WIDE_FORWARD:
            case CAR_CAMERA_NARROW_FORWARD:
                local_x = length / 4.0;
                local_z = height * 0.8;
                local_yaw = 0;
                break;
            case CAR_CAMERA_SIDE_LEFT_FORWARDVIEW:
                local_x = 0;
                local_y = width / 2.0;
                local_z = height * 0.8;
                local_yaw = M_PI / 4.0;
                break;
            case CAR_CAMERA_SIDE_RIGHT_FORWARDVIEW:
                local_x = 0;
                local_y = -width / 2.0;
                local_z = height * 0.8;
                local_yaw = -M_PI / 4.0;
                break;
            case CAR_CAMERA_SIDE_LEFT_REARVIEW:
                local_x = length / 4.0;
                local_y = width / 2.0;
                local_z = height * 0.6;
                local_yaw = M_PI - M_PI / 6.0; // 150 deg
                break;
            case CAR_CAMERA_SIDE_RIGHT_REARVIEW:
                local_x = length / 4.0;
                local_y = -width / 2.0;
                local_z = height * 0.6;
                local_yaw = -M_PI + M_PI / 6.0; // -150 deg
                break;
            case CAR_CAMERA_REARVIEW:
                local_x = -length / 2.0;
                local_z = height * 0.4;
                local_yaw = M_PI;
                break;
            case CAR_CAMERA_FRONT_BUMPER:
                local_x = length / 2.0;
                local_z = height * 0.3;
                local_yaw = 0;
                break;
            default:
                // Default to main forward camera
                local_x = length / 4.0;
                local_z = height * 0.8;
                local_yaw = 0;
                break;
        }

        // Rotate local position by car orientation
        double cos_theta = cos(car->orientation);
        double sin_theta = sin(car->orientation);
        
        double world_offset_x = local_x * cos_theta - local_y * sin_theta;
        double world_offset_y = local_x * sin_theta + local_y * cos_theta;

        camera->position.x = car->center.x + world_offset_x;
        camera->position.y = car->center.y + world_offset_y;
        camera->z_altitude = local_z;
        
        camera->orientation = angle_normalize(car->orientation + local_yaw);
    }

    // Ensure buffers are correct (handles manual AA level changes)
    int needed_level = (camera->aa_level > 1) ? camera->aa_level : 1;
    rgbcam_set_aa_level(camera, needed_level);

    // Render scene
    // We render directly into the render_buffer using render_width/render_height
    // render_scene_internal will use camera->render_width/height/buffer
    render_scene_internal(camera, sim, exclude_objects);

#ifndef OPT_NO_HUD
    BENCH_START(bench_time_hud);
    draw_hud(camera);
    BENCH_END(bench_time_hud);
#endif

    // Post-processing and downsampling
    if (camera->aa_level > 1) {
        // Downsample from render_buffer to data
        // We need a temporary camera struct to pass to downsample as source
        // But downsample takes RGBCamera* src, RGBCamera* dest
        // src is 'camera' (with high res data), dest needs to be a wrapper around orig_data
        
        // Wait, render_scene_internal now uses render_buffer.
        // So camera->render_buffer contains the high-res image.
        // camera->data is the destination low-res buffer.
        
        // We need to adapt downsample to work with this setup or create a temp struct
        // Let's create a temp struct for the source (high res)
        RGBCamera src_cam = *camera;
        src_cam.width = camera->render_width;
        src_cam.height = camera->render_height;
        src_cam.data = camera->render_buffer;
        
        BENCH_START(bench_time_downsample);
        downsample(&src_cam, camera, camera->aa_level);
        BENCH_END(bench_time_downsample);
    } else {
        // AA level 1: Swap pointers (Double buffering)
        // camera->render_buffer has the new frame
        // camera->data has the old frame buffer
        
        uint8_t* temp = camera->data;
        camera->data = camera->render_buffer;
        camera->render_buffer = temp;
    }
    
    BENCH_END(bench_time_total);

#ifdef ENABLE_BENCHMARKING
    bench_frame_count++;
    if (bench_frame_count >= 1000) {
        printf("--- Camera Benchmark (Avg over %d frames) ---\n", bench_frame_count);
        printf("Sky:           %.6f ms\n", (bench_time_sky / bench_frame_count) * 1000.0);
        printf("Ground:        %.6f ms\n", (bench_time_ground / bench_frame_count) * 1000.0);
        printf("Cars:          %.6f ms\n", (bench_time_cars / bench_frame_count) * 1000.0);
        printf("Roads:         %.6f ms\n", (bench_time_roads / bench_frame_count) * 1000.0);
        printf("Intersections: %.6f ms\n", (bench_time_intersections / bench_frame_count) * 1000.0);
        printf("HUD:           %.6f ms\n", (bench_time_hud / bench_frame_count) * 1000.0);
        printf("Downsample:    %.6f ms\n", (bench_time_downsample / bench_frame_count) * 1000.0);
        printf("TOTAL:         %.6f ms\n", (bench_time_total / bench_frame_count) * 1000.0);
        printf("---------------------------------------------\n");
        
        bench_frame_count = 0;
        bench_time_sky = 0;
        bench_time_ground = 0;
        bench_time_cars = 0;
        bench_time_roads = 0;
        bench_time_intersections = 0;
        bench_time_total = 0;
        bench_time_hud = 0;
        bench_time_downsample = 0;
    }
#endif
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
    frame->aa_level = 1;
    frame->z_buffer = NULL;
    frame->z_buffer_size = 0;
    frame->data = (uint8_t*) malloc(sizeof(uint8_t) * 3 * width * height); // CHW format
    if (frame->data == NULL) {
        LOG_ERROR("Failed to allocate memory for RGB data");
        free(frame);
        return NULL;
    }
    memset(frame->data, 0, sizeof(uint8_t) * 3 * width * height);

    // Pre-compute background (sky + ground)
    frame->background_buffer = (uint8_t*) malloc(sizeof(uint8_t) * 3 * width * height);
    if (frame->background_buffer == NULL) {
        LOG_ERROR("Failed to allocate memory for background buffer");
        free(frame->data);
        free(frame);
        return NULL;
    }

    // Fill background buffer
    fill_background(frame->background_buffer, width, height);

    // Initialize render buffer (same size as data initially, AA=1)
    frame->render_width = width;
    frame->render_height = height;
    frame->render_buffer = (uint8_t*) malloc(sizeof(uint8_t) * 3 * width * height);
    if (frame->render_buffer == NULL) {
        LOG_ERROR("Failed to allocate memory for render buffer");
        free(frame->background_buffer);
        free(frame->data);
        free(frame);
        return NULL;
    }
    // Initialize render buffer with background
    memcpy(frame->render_buffer, frame->background_buffer, sizeof(uint8_t) * 3 * width * height);

    return frame;
}

void rgbcam_set_aa_level(RGBCamera* camera, int level) {
    if (camera == NULL || level < 1) return;
    
    int target_width = camera->width * level;
    int target_height = camera->height * level;

    if (camera->aa_level == level && 
        camera->render_width == target_width && 
        camera->render_height == target_height && 
        camera->render_buffer != NULL) {
        return;
    }

    camera->aa_level = level;
    camera->render_width = target_width;
    camera->render_height = target_height;

    // Reallocate render buffer
    if (camera->render_buffer != NULL) {
        free(camera->render_buffer);
    }
    camera->render_buffer = (uint8_t*) malloc(sizeof(uint8_t) * 3 * camera->render_width * camera->render_height);
    if (camera->render_buffer == NULL) {
        LOG_ERROR("Failed to allocate memory for render buffer in rgbcam_set_aa_level");
        // Try to recover with level 1
        camera->aa_level = 1;
        camera->render_width = camera->width;
        camera->render_height = camera->height;
        camera->render_buffer = (uint8_t*) malloc(sizeof(uint8_t) * 3 * camera->render_width * camera->render_height);
    }

    // Reallocate background buffer to match render resolution
    if (camera->background_buffer != NULL) {
        free(camera->background_buffer);
    }
    camera->background_buffer = (uint8_t*) malloc(sizeof(uint8_t) * 3 * camera->render_width * camera->render_height);
    if (camera->background_buffer != NULL) {
        fill_background(camera->background_buffer, camera->render_width, camera->render_height);
    }

    // Reset Z-buffer so it gets reallocated on next render
    if (camera->z_buffer != NULL) {
        free(camera->z_buffer);
        camera->z_buffer = NULL;
        camera->z_buffer_size = 0;
    }
}

void rgbcam_free(RGBCamera* frame) {
    if (frame != NULL) {
        if (frame->data != NULL) {
            free(frame->data);
        }
        if (frame->z_buffer != NULL) {
            free(frame->z_buffer);
        }
        if (frame->background_buffer != NULL) {
            free(frame->background_buffer);
        }
        if (frame->render_buffer != NULL) {
            free(frame->render_buffer);
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

void rgbcam_attach_to_car(RGBCamera* camera, Car* car, CarCameraType camera_type) {
    if (camera == NULL) return;
    camera->attached_car = car;
    camera->attached_car_camera_type = camera_type;

    double deg_to_rad = M_PI / 180.0;

    switch (camera_type) {
        case CAR_CAMERA_MAIN_FORWARD:
            camera->fov = 50.0 * deg_to_rad;
            camera->max_distance = 150.0;
            break;
        case CAR_CAMERA_WIDE_FORWARD:
            camera->fov = 120.0 * deg_to_rad;
            camera->max_distance = 60.0;
            break;
        case CAR_CAMERA_SIDE_LEFT_FORWARDVIEW:
        case CAR_CAMERA_SIDE_RIGHT_FORWARDVIEW:
            camera->fov = 90.0 * deg_to_rad;
            camera->max_distance = 80.0;
            break;
        case CAR_CAMERA_SIDE_LEFT_REARVIEW:
        case CAR_CAMERA_SIDE_RIGHT_REARVIEW:
            camera->fov = 75.0 * deg_to_rad;
            camera->max_distance = 100.0;
            break;
        case CAR_CAMERA_REARVIEW:
            camera->fov = 140.0 * deg_to_rad;
            camera->max_distance = 60.0;
            break;
        case CAR_CAMERA_FRONT_BUMPER:
            camera->fov = 150.0 * deg_to_rad;
            camera->max_distance = 20.0;
            break;
        case CAR_CAMERA_NARROW_FORWARD:
            camera->fov = 35.0 * deg_to_rad;
            camera->max_distance = 250.0;
            break;
        default:
            LOG_WARN("Unknown CarCameraType %d in rgbcam_attach_to_car. Defaulting to main forward camera", camera_type);
            camera->fov = 50.0 * deg_to_rad;
            camera->max_distance = 150.0;
            break;
    }
}


RGB rgb(uint8_t r, uint8_t g, uint8_t b) {
    return (RGB){ r, g, b };
}

const RGB RGB_WHITE = { 255, 255, 255 };
const RGB RGB_BLACK = { 0, 0, 0 };
const RGB RGB_RED   = { 255, 0, 0 };
const RGB RGB_GREEN = { 0, 255, 0 };
const RGB RGB_BLUE  = { 0, 0, 255 };
const RGB RGB_YELLOW = { 255, 255, 0 };
const RGB RGB_CYAN = { 0, 255, 255 };
const RGB RGB_MAGENTA = { 255, 0, 255 };
const RGB RGB_GRAY = { 128, 128, 128 };
const RGB RGB_ORANGE = { 255, 165, 0 };