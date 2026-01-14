#include "ai.h"
#include "logging.h"
#include "utils.h"
#include "map.h"
#include "sim.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Constants for colors
static const RGB COLOR_ROAD_DEFAULT = {255, 255, 255};
static const RGB COLOR_ROAD_DARKER = {200, 200, 200};
// static const RGB COLOR_BACKGROUND = {0, 0, 0};
static const RGB COLOR_CAR = {255, 0, 0}; // Red for car

// #define RENDER_DEBUG_GRAPH

#ifdef RENDER_DEBUG_GRAPH
static const RGB COLOR_GRAPH_NODE = {165, 42, 42}; // brown
static const RGB COLOR_GRAPH_EDGE = {255, 165, 0}; // Orange
#endif

// Helper to set a pixel
static void set_pixel(uint8_t* buffer, int width, int height, int x, int y, RGB color) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    int idx = (y * width + x) * 3;
    buffer[idx+0] = color.r;
    buffer[idx+1] = color.g;
    buffer[idx+2] = color.b;
}

// Helper to draw a line
static void draw_line(uint8_t* buffer, int width, int height, int x0, int y0, int x1, int y1, RGB color, int thickness) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (1) {
        // Draw with thickness
        for (int tx = -thickness/2; tx <= thickness/2; tx++) {
            for (int ty = -thickness/2; ty <= thickness/2; ty++) {
                set_pixel(buffer, width, height, x0 + tx, y0 + ty, color);
            }
        }
        
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// Helper to draw a filled square
static void draw_filled_square(uint8_t* buffer, int width, int height, int cx, int cy, int size, RGB color) {
    int half_size = size / 2;
    for (int y = cy - half_size; y <= cy + half_size; y++) {
        for (int x = cx - half_size; x <= cx + half_size; x++) {
            set_pixel(buffer, width, height, x, y, color);
        }
    }
}

// Helper to draw a filled triangle (for car)
static void draw_filled_triangle(uint8_t* buffer, int width, int height, int x1, int y1, int x2, int y2, int x3, int y3, RGB color) {
    // Bounding box
    int min_x = fmin(x1, fmin(x2, x3));
    int max_x = fmax(x1, fmax(x2, x3));
    int min_y = fmin(y1, fmin(y2, y3));
    int max_y = fmax(y1, fmax(y2, y3));

    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            // Barycentric coordinates
            double w1 = (double)((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3));
            double w2 = (double)((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3));
            double w3 = 1.0 - w1 - w2;

            if (w1 >= 0 && w2 >= 0 && w3 >= 0) {
                set_pixel(buffer, width, height, x, y, color);
            }
        }
    }
}

// Helper to draw an arc
// static void draw_arc(uint8_t* buffer, int width, int height, int cx, int cy, int radius, double start_angle, double end_angle, RGB color, int thickness) {
//     // Normalize angles
//     while (start_angle < 0) start_angle += 2 * M_PI;
//     while (end_angle < 0) end_angle += 2 * M_PI;
//     while (start_angle >= 2 * M_PI) start_angle -= 2 * M_PI;
//     while (end_angle >= 2 * M_PI) end_angle -= 2 * M_PI;

//     // Determine if we cross 0/2PI
//     bool cross_zero = end_angle < start_angle;

//     // Iterate over bounding box of the circle
//     int min_x = cx - radius - thickness;
//     int max_x = cx + radius + thickness;
//     int min_y = cy - radius - thickness;
//     int max_y = cy + radius + thickness;

//     for (int y = min_y; y <= max_y; y++) {
//         for (int x = min_x; x <= max_x; x++) {
//             double dist = sqrt(pow(x - cx, 2) + pow(y - cy, 2));
//             if (dist >= radius - thickness/2.0 && dist <= radius + thickness/2.0) {
//                 double angle = atan2(y - cy, x - cx); // -PI to PI
//                 if (angle < 0) angle += 2 * M_PI; // 0 to 2PI

//                 // Check if angle is within range
//                 bool in_range = false;
//                 if (cross_zero) {
//                     if (angle >= start_angle || angle <= end_angle) in_range = true;
//                 } else {
//                     if (angle >= start_angle && angle <= end_angle) in_range = true;
//                 }

//                 if (in_range) {
//                     set_pixel(buffer, width, height, x, y, color);
//                 }
//             }
//         }
//     }
// }

// Convert world coords to pixel coords
static void world_to_pixel(MiniMap* minimap, Coordinates world, int* px, int* py) {
    // Map world X to pixel X (left to right)
    // Map world Y to pixel Y (bottom to top, so invert Y)
    *px = (int)((world.x - minimap->min_coord.x) * minimap->scale_x);
    *py = minimap->height - 1 - (int)((world.y - minimap->min_coord.y) * minimap->scale_y);
}

static void minimap_render_static(MiniMap* minimap, Map* map) {
    if (!minimap || !minimap->static_data || !map) return;

    // Clear background
    memset(minimap->static_data, 0, 3 * minimap->width * minimap->height);

    // Calculate thickness based on resolution (e.g., 1 meter width)
    int thickness = (int)(minimap->scale_x * 4.0); // 4 meters wide roads roughly
    if (thickness < 1) thickness = 1;

    for (int i = 0; i < map->num_roads; i++) {
        Road* road = map_get_road(map, i);
        RGB color = COLOR_ROAD_DEFAULT;
        
        // Darker shade for Westbound and Southbound
        if (road->direction == DIRECTION_WEST || road->direction == DIRECTION_SOUTH) {
            color = COLOR_ROAD_DARKER;
        }

        if (road->type == STRAIGHT) {
            int x0, y0, x1, y1;
            world_to_pixel(minimap, road->start_point, &x0, &y0);
            world_to_pixel(minimap, road->end_point, &x1, &y1);
            draw_line(minimap->static_data, minimap->width, minimap->height, x0, y0, x1, y1, color, thickness);
        } else if (road->type == TURN) {
            // Use segment drawing similar to render_lane.c
            int num_segments = 20; // Sufficient for smooth curves on minimap
            double step = (road->end_angle - road->start_angle) / num_segments;
            
            for (int j = 0; j < num_segments; j++) {
                double theta1 = road->start_angle + j * step;
                double theta2 = road->start_angle + (j + 1) * step;
                
                Coordinates p1_world = {
                    road->center.x + road->radius * cos(theta1),
                    road->center.y + road->radius * sin(theta1)
                };
                Coordinates p2_world = {
                    road->center.x + road->radius * cos(theta2),
                    road->center.y + road->radius * sin(theta2)
                };
                
                int x1, y1, x2, y2;
                world_to_pixel(minimap, p1_world, &x1, &y1);
                world_to_pixel(minimap, p2_world, &x2, &y2);
                
                draw_line(minimap->static_data, minimap->width, minimap->height, x1, y1, x2, y2, color, thickness);
            }
        }
    }
}

MiniMap* minimap_malloc(int width, int height, Map* map) {
    MiniMap* minimap = (MiniMap*)malloc(sizeof(MiniMap));
    if (!minimap) return NULL;

    minimap->width = width;
    minimap->height = height;
    minimap->data = (uint8_t*)malloc(3 * width * height);
    minimap->static_data = (uint8_t*)malloc(3 * width * height);
    minimap->marked_car_id = ID_NULL;
    minimap->marked_path = NULL;
    minimap->debug_planner = path_planner_create(map, false);
    
    for(int i=0; i<8; i++) {
        minimap->marked_landmarks[i] = (Coordinates){0,0};
        minimap->marked_landmark_colors[i] = (RGB){0,0,0};
    }

    if (!minimap->data || !minimap->static_data) {
        if (minimap->data) free(minimap->data);
        if (minimap->static_data) free(minimap->static_data);
        free(minimap);
        return NULL;
    }

    // Calculate extent
    double min_x = DBL_MAX, min_y = DBL_MAX;
    double max_x = -DBL_MAX, max_y = -DBL_MAX;

    for (int i = 0; i < map->num_roads; i++) {
        Road* road = map_get_road(map, i);
        // Check start, end, and center (for turns)
        // Simplified: just check start and end for all, and maybe center +/- radius for turns
        Coordinates pts[] = {road->start_point, road->end_point};
        for (int j = 0; j < 2; j++) {
            if (pts[j].x < min_x) min_x = pts[j].x;
            if (pts[j].y < min_y) min_y = pts[j].y;
            if (pts[j].x > max_x) max_x = pts[j].x;
            if (pts[j].y > max_y) max_y = pts[j].y;
        }
        if (road->type == TURN) {
             if (road->center.x - road->radius < min_x) min_x = road->center.x - road->radius;
             if (road->center.x + road->radius > max_x) max_x = road->center.x + road->radius;
             if (road->center.y - road->radius < min_y) min_y = road->center.y - road->radius;
             if (road->center.y + road->radius > max_y) max_y = road->center.y + road->radius;
        }
    }
    
    // Add some padding
    double padding = 50.0; // meters
    minimap->min_coord = (Coordinates){min_x - padding, min_y - padding};
    minimap->max_coord = (Coordinates){max_x + padding, max_y + padding};

    double world_w = minimap->max_coord.x - minimap->min_coord.x;
    double world_h = minimap->max_coord.y - minimap->min_coord.y;

    // Maintain aspect ratio
    double scale_x = width / world_w;
    double scale_y = height / world_h;
    double scale = fmin(scale_x, scale_y);
    
    minimap->scale_x = scale;
    minimap->scale_y = scale;

    // Re-center
    double final_world_w = width / scale;
    double final_world_h = height / scale;
    double center_x = (minimap->min_coord.x + minimap->max_coord.x) / 2.0;
    double center_y = (minimap->min_coord.y + minimap->max_coord.y) / 2.0;
    
    minimap->min_coord.x = center_x - final_world_w / 2.0;
    minimap->min_coord.y = center_y - final_world_h / 2.0;
    minimap->max_coord.x = center_x + final_world_w / 2.0;
    minimap->max_coord.y = center_y + final_world_h / 2.0;

    minimap_render_static(minimap, map);

    return minimap;
}

void minimap_free(MiniMap* minimap) {
    if (minimap) {
        if (minimap->data) free(minimap->data);
        if (minimap->static_data) free(minimap->static_data);
        if (minimap->debug_planner) path_planner_free(minimap->debug_planner);
        free(minimap);
    }
}

#define MINIMAP_CONGESTION_SEGMENT_LENGTH 25.0

static void get_lane_point(Lane* lane, double progress, Coordinates* out_point) {
    if (!lane || !out_point) return;
    
    double t = progress / lane->length;
    if (t < 0) t = 0;
    if (t > 1) t = 1;

    if (lane->type == LINEAR_LANE) {
        out_point->x = lane->start_point.x + t * (lane->end_point.x - lane->start_point.x);
        out_point->y = lane->start_point.y + t * (lane->end_point.y - lane->start_point.y);
    } else if (lane->type == QUARTER_ARC_LANE) {
        double theta = lane->start_angle + t * (lane->end_angle - lane->start_angle);
        out_point->x = lane->center.x + lane->radius * cos(theta);
        out_point->y = lane->center.y + lane->radius * sin(theta);
    }
}



#ifdef RENDER_DEBUG_GRAPH
static void minimap_render_graph(MiniMap* minimap) {
    if (!minimap || !minimap->debug_planner || !minimap->debug_planner->map || !minimap->debug_planner->decision_graph) return;
    
    PathPlanner* planner = minimap->debug_planner;
    DirectedGraph* g = planner->decision_graph;
    
    // Draw Edges first (so nodes are on top)
    for (int u_id = 0; u_id < g->num_nodes; ++u_id) {
        MapNode* u_node = &planner->map_nodes[u_id];
        Lane* lane_u = map_get_lane(planner->map, u_node->lane_id);
        Coordinates p1;
        get_lane_point(lane_u, u_node->progress, &p1);
        
        int x1, y1;
        world_to_pixel(minimap, p1, &x1, &y1);

        DG_Edge* edge = g->adj[u_id];
        while (edge) {
            int v_id = edge->to;
            MapNode* v_node = &planner->map_nodes[v_id];
            Lane* lane_v = map_get_lane(planner->map, v_node->lane_id);
            
            Coordinates p2;
            get_lane_point(lane_v, v_node->progress, &p2);
            
            int x2, y2;
            world_to_pixel(minimap, p2, &x2, &y2);
            
            draw_line(minimap->data, minimap->width, minimap->height, x1, y1, x2, y2, COLOR_GRAPH_EDGE, 1);
            
            edge = edge->next;
        }
    }

    // Draw Nodes
    for (int i = 0; i < g->num_nodes; ++i) {
        MapNode* u = &planner->map_nodes[i];
        Lane* lane_u = map_get_lane(planner->map, u->lane_id);
        Coordinates p;
        get_lane_point(lane_u, u->progress, &p);
        
        int x, y;
        world_to_pixel(minimap, p, &x, &y);
        
        set_pixel(minimap->data, minimap->width, minimap->height, x, y, COLOR_GRAPH_NODE);
    }
}
#endif



void minimap_render(MiniMap* minimap, Simulation* sim) {
    if (!minimap || !minimap->data || !minimap->static_data) return;

    // Copy static background
    memcpy(minimap->data, minimap->static_data, 3 * minimap->width * minimap->height);
    
    // Render Graph
#ifdef RENDER_DEBUG_GRAPH
    minimap_render_graph(minimap);
#endif

    // Draw traffic congestion on straight roads
    for (int i = 0; i < sim->map.num_roads; i++) {
        Road* road = map_get_road(&sim->map, i);
        if (road->type != STRAIGHT) continue;

        int num_segments = (int)ceil(road->length / MINIMAP_CONGESTION_SEGMENT_LENGTH);
        if (num_segments == 0) continue;

        double* speed_sum = (double*)calloc(num_segments, sizeof(double));
        int* car_count = (int*)calloc(num_segments, sizeof(int));

        if (!speed_sum || !car_count) {
            if (speed_sum) free(speed_sum);
            if (car_count) free(car_count);
            continue;
        }

        for (int l = 0; l < road->num_lanes; l++) {
            Lane* lane = road_get_lane(road, &sim->map, l);
            if (!lane) continue;

            for (int c = 0; c < lane->num_cars; c++) {
                CarId cid = lane->cars_ids[c];
                Car* car = sim_get_car(sim, cid);
                if (!car) continue;

                // Calculate distance from start of road
                double dist = car->lane_progress_meters;
                
                // Clamp distance to road length just in case
                if (dist < 0) dist = 0;
                if (dist >= road->length) dist = road->length - 0.1;

                int seg_idx = (int)(dist / MINIMAP_CONGESTION_SEGMENT_LENGTH);
                if (seg_idx >= 0 && seg_idx < num_segments) {
                    speed_sum[seg_idx] += car->speed;
                    car_count[seg_idx]++;
                }
            }
        }

        for (int s = 0; s < num_segments; s++) {
            double avg_speed = (car_count[s] > 0) ? (speed_sum[s] / car_count[s]) : road->speed_limit;
            double flow = avg_speed / road->speed_limit;
            if (flow > 1.0) flow = 1.0;
            if (flow < 0.0) flow = 0.0;
            
            // Congestion = 1 - flow
            // Red (congestion=1) to Green (congestion=0)
            RGB color;
            color.r = (uint8_t)(255 * (1.0 - flow));
            color.g = (uint8_t)(255 * flow);
            color.b = 0;
            // mute them a little bit:
            color.r = (uint8_t)(color.r * 0.7);
            color.g = (uint8_t)(color.g * 0.7);

            double t1 = (double)s * MINIMAP_CONGESTION_SEGMENT_LENGTH / road->length;
            double t2 = (double)(s + 1) * MINIMAP_CONGESTION_SEGMENT_LENGTH / road->length;
            if (t2 > 1.0) t2 = 1.0;

            Coordinates p1 = {
                road->start_point.x + t1 * (road->end_point.x - road->start_point.x),
                road->start_point.y + t1 * (road->end_point.y - road->start_point.y)
            };
            Coordinates p2 = {
                road->start_point.x + t2 * (road->end_point.x - road->start_point.x),
                road->start_point.y + t2 * (road->end_point.y - road->start_point.y)
            };

            int x1, y1, x2, y2;
            world_to_pixel(minimap, p1, &x1, &y1);
            world_to_pixel(minimap, p2, &x2, &y2);
            
            int thickness = (int)(minimap->scale_x * 4.0); 
            if (thickness < 1) thickness = 1;

            draw_line(minimap->data, minimap->width, minimap->height, x1, y1, x2, y2, color, thickness);
        }

        free(speed_sum);
        free(car_count);
    }

    // Draw marked path
    if (minimap->marked_path && minimap->marked_path->path_exists) {
        RGB color_path = {0, 0, 255}; // Blue
        int thickness = 2;

        PathPlanner* planner = minimap->marked_path;

        for (int i = 0; i < planner->num_solution_actions; i++) {
            MapNode* u = &planner->solution_nodes[i];
            MapNode* v = &planner->solution_nodes[i+1];
            
            Lane* lane_u = map_get_lane(sim_get_map(sim), u->lane_id);
            Lane* lane_v = map_get_lane(sim_get_map(sim), v->lane_id);
            
            if (!lane_u || !lane_v) continue;

            if (u->lane_id == v->lane_id) {
                // Same lane segment
                double start_m = u->progress;
                double end_m = v->progress;
                
                // Draw segment on lane
                if (lane_u->type == LINEAR_LANE) {
                    double t1 = start_m / lane_u->length;
                    double t2 = end_m / lane_u->length;
                    
                    Coordinates p1 = {
                        lane_u->start_point.x + t1 * (lane_u->end_point.x - lane_u->start_point.x),
                        lane_u->start_point.y + t1 * (lane_u->end_point.y - lane_u->start_point.y)
                    };
                    Coordinates p2 = {
                        lane_u->start_point.x + t2 * (lane_u->end_point.x - lane_u->start_point.x),
                        lane_u->start_point.y + t2 * (lane_u->end_point.y - lane_u->start_point.y)
                    };
                    
                    int x0, y0, x1, y1;
                    world_to_pixel(minimap, p1, &x0, &y0);
                    world_to_pixel(minimap, p2, &x1, &y1);
                    draw_line(minimap->data, minimap->width, minimap->height, x0, y0, x1, y1, color_path, thickness);
                } else if (lane_u->type == QUARTER_ARC_LANE) {
                    double t_start = start_m / lane_u->length;
                    double t_end = end_m / lane_u->length;
                    
                    double angle_start = lane_u->start_angle + t_start * (lane_u->end_angle - lane_u->start_angle);
                    double angle_end = lane_u->start_angle + t_end * (lane_u->end_angle - lane_u->start_angle);

                    int segments = 10; 
                    double step = (angle_end - angle_start) / segments;

                    for (int j = 0; j < segments; j++) {
                        double theta1 = angle_start + j * step;
                        double theta2 = angle_start + (j + 1) * step;
                        
                        Coordinates p1_world = {
                            lane_u->center.x + lane_u->radius * cos(theta1),
                            lane_u->center.y + lane_u->radius * sin(theta1)
                        };
                        Coordinates p2_world = {
                            lane_u->center.x + lane_u->radius * cos(theta2),
                            lane_u->center.y + lane_u->radius * sin(theta2)
                        };
                        
                        int x1, y1, x2, y2;
                        world_to_pixel(minimap, p1_world, &x1, &y1);
                        world_to_pixel(minimap, p2_world, &x2, &y2);
                        
                        draw_line(minimap->data, minimap->width, minimap->height, x1, y1, x2, y2, color_path, thickness);
                    }
                }
            } else {
                // Different lanes (connection/change/merge)
                // Draw straight line between u and v
                Coordinates p1, p2;
                get_lane_point(lane_u, u->progress, &p1);
                get_lane_point(lane_v, v->progress, &p2);
                
                int x1, y1, x2, y2;
                world_to_pixel(minimap, p1, &x1, &y1);
                world_to_pixel(minimap, p2, &x2, &y2);
                
                draw_line(minimap->data, minimap->width, minimap->height, x1, y1, x2, y2, color_path, thickness);
            }
        }
    }

    // Draw landmarks
    for (int i = 0; i < 8; i++) {
        if (minimap->marked_landmark_colors[i].r != 0 || 
            minimap->marked_landmark_colors[i].g != 0 || 
            minimap->marked_landmark_colors[i].b != 0) {
            
            int px, py;
            world_to_pixel(minimap, minimap->marked_landmarks[i], &px, &py);

            // Calculate size based on resolution (e.g., 10 meters)
            int size = (int)(minimap->scale_x * 10.0);
            if (size < 6) size = 6; // Minimum size
            
            draw_filled_square(minimap->data, minimap->width, minimap->height, px, py, size, minimap->marked_landmark_colors[i]);
        }
    }

    // Draw marked car
    if (minimap->marked_car_id != ID_NULL) {
        Car* car = sim_get_car(sim, minimap->marked_car_id);
        if (car) {
            Coordinates car_pos = car->center; // Default to actual position

            // Try to center on road
            Lane* lane = car_get_lane(car, &sim->map);
            if (lane) {
                Road* road = lane_get_road(lane, &sim->map);
                if (road) {
                    if (road->type == STRAIGHT) {
                        // Project car position onto the road line segment
                        double dx = road->end_point.x - road->start_point.x;
                        double dy = road->end_point.y - road->start_point.y;
                        double length_sq = dx*dx + dy*dy;
                        
                        if (length_sq > 0) {
                            double t = ((car->center.x - road->start_point.x) * dx + 
                                        (car->center.y - road->start_point.y) * dy) / length_sq;
                            
                            // Clamp t to [0, 1] just in case
                            if (t < 0) t = 0;
                            if (t > 1) t = 1;
                            
                            car_pos.x = road->start_point.x + t * dx;
                            car_pos.y = road->start_point.y + t * dy;
                        }
                    } else if (road->type == TURN) {
                        // Project car position onto the circle arc
                        double dx = car->center.x - road->center.x;
                        double dy = car->center.y - road->center.y;
                        double angle = atan2(dy, dx);
                        
                        car_pos.x = road->center.x + road->radius * cos(angle);
                        car_pos.y = road->center.y + road->radius * sin(angle);
                    }
                }
            }

            int cx, cy;
            world_to_pixel(minimap, car_pos, &cx, &cy);
            
            // Triangle size based on resolution (e.g., 5 meters)
            double size = minimap->scale_x * 5.0;
            if (size < 8.0) size = 8.0; // Minimum size
            
            // Orientation (screen space angle = -world angle)
            double angle = -car->orientation;
            
            // Points relative to center
            // Tip
            int x1 = cx + (int)(size * cos(angle));
            int y1 = cy + (int)(size * sin(angle));
            
            // Back left
            int x2 = cx + (int)(size * 0.7 * cos(angle + 2.5));
            int y2 = cy + (int)(size * 0.7 * sin(angle + 2.5));
            
            // Back right
            int x3 = cx + (int)(size * 0.7 * cos(angle - 2.5));
            int y3 = cy + (int)(size * 0.7 * sin(angle - 2.5));
            
            draw_filled_triangle(minimap->data, minimap->width, minimap->height, x1, y1, x2, y2, x3, y3, COLOR_CAR);
        }
    }
}
