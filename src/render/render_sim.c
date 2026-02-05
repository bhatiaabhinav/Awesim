#include "utils.h"
#include "render.h"
#include "bad.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

static Lidar* highlighted_car_lidar = NULL;
static RGBCamera* highlighted_car_camera = NULL;
static InfosDisplay* highlighted_car_infos_display = NULL;
static MiniMap* highlighted_car_minimap = NULL;
static PathPlanner* highlighted_car_path_planner = NULL;
static Collisions* collisions = NULL;
static Car* highlighted_car = NULL;
static SituationalAwareness* situational_awareness_highlighted_car = NULL;

void render_sim(SDL_Renderer *renderer, Simulation *sim, bool draw_lanes, bool draw_cars,
                bool draw_track_lines, bool draw_traffic_lights, bool draw_car_ids, bool draw_car_speeds, bool draw_lane_ids, bool draw_road_names, bool draw_lidar, bool draw_camera, bool draw_minimap, bool draw_infos_display, int hud_font_size, bool benchmark)
{
    Map *map = sim_get_map(sim);

    // Clear screen with background color
    SDL_SetRenderDrawColor(renderer, BACKGROUND_COLOR.r, BACKGROUND_COLOR.g, BACKGROUND_COLOR.b, BACKGROUND_COLOR.a);
    SDL_RenderClear(renderer);

    Uint64 start, end;
    double elapsed_us;

    // Render road and intersection geometry
    if (draw_lanes) {
        // Intersections
        start = SDL_GetPerformanceCounter();
        for (int i = 0; i < map->num_intersections; i++) {
            render_intersection(renderer, map_get_intersection(map, i), map);
        }
        end = SDL_GetPerformanceCounter();
        elapsed_us = (double)(end - start) * 1e6 / SDL_GetPerformanceFrequency();
        if (benchmark) printf("Intersections: %.2f us\n", elapsed_us);

        // Straight roads
        start = SDL_GetPerformanceCounter();
        for (int i = 0; i < map->num_roads; i++) {
            Road *road = map_get_road(map, i);
            if (road->type == STRAIGHT) {
                for (int j = 0; j < road->num_lanes; j++) {
                    Lane* lane = road_get_lane(road, map, j);
                    render_lane(renderer, lane, map, true, true, draw_lane_ids);
                }
                if (draw_road_names) render_straight_road_name(renderer, road, map);
            }
        }
        end = SDL_GetPerformanceCounter();
        elapsed_us = (double)(end - start) * 1e6 / SDL_GetPerformanceFrequency();
        if (benchmark) printf("Straight Roads: %.2f us\n", elapsed_us);

        // Turn roads
        start = SDL_GetPerformanceCounter();
        for (int i = 0; i < map->num_roads; i++) {
            const Road *road = map_get_road(map, i);
            if (road->type == TURN) {
                for (int j = 0; j < road->num_lanes; j++) {
                    Lane *lane = road_get_lane(road, map, j);
                    render_lane(renderer, lane, map, true, true, draw_lane_ids);
                }
            }
        }
        end = SDL_GetPerformanceCounter();
        elapsed_us = (double)(end - start) * 1e6 / SDL_GetPerformanceFrequency();
        if (benchmark) printf("Turn Roads: %.2f us\n", elapsed_us);
    }

    // Render lane centerlines (used for debugging or visualizing traffic state)
    if (draw_track_lines) {
        start = SDL_GetPerformanceCounter();
        for (int i = 0; i < map->num_roads; i++) {
            Road *road = map_get_road(map, i);
            for (int j = 0; j < road->num_lanes; j++) {
                Lane* lane = road_get_lane(road, map, j);
                render_lane_center_line(renderer, lane, map, LANE_CENTER_LINE_COLOR, false);
            }
        }

        // Optional: draw center lines at intersections if traffic lights are not drawn
        if (!draw_traffic_lights) {
            for (int i = 0; i < map->num_intersections; i++) {
                Intersection *intersection = map_get_intersection(map, i);
                for (int j = 0; j < intersection->num_lanes; j++) {
                    Lane* lane = intersection_get_lane(intersection, map, j);
                    render_lane_center_line(renderer, lane, map, LANE_CENTER_LINE_COLOR, false);
                }
            }
        }

        end = SDL_GetPerformanceCounter();
        elapsed_us = (double)(end - start) * 1e6 / SDL_GetPerformanceFrequency();
        if (benchmark) printf("Track Lines: %.2f us\n", elapsed_us);
    }

    // Render traffic light logic on lanes at intersections
    if (draw_traffic_lights) {
        start = SDL_GetPerformanceCounter();

        for (int i = 0; i < map->num_intersections; i++) {
            Intersection *intersection = map_get_intersection(map, i);
            // if (draw_road_names) render_intersection_name(renderer, intersection, map);

            IntersectionState state = intersection->state;

            for (int j = 0; j < intersection->num_lanes; j++) {
                Lane *lane = intersection_get_lane(intersection, map, j);
                if (!lane) continue;

                Direction dir = lane->direction;
                Lane *next_lane = lane_get_connection_straight(lane, map);
                Direction next_dir = next_lane ? next_lane->direction : dir;

                bool is_vert = dir == DIRECTION_NORTH || dir == DIRECTION_SOUTH;
                bool is_turn = lane->type == QUARTER_ARC_LANE;
                bool next_is_hori = next_dir == DIRECTION_EAST || next_dir == DIRECTION_WEST;
                bool is_NS = is_vert || (is_turn && next_is_hori);
                bool free_right = true;

                SDL_Color light_color = RED_LIGHT_COLOR;
                bool yield = false;

                switch (state) {
                    case FOUR_WAY_STOP:
                        light_color = FOUR_WAY_STOP_COLOR;
                        break;
                    case T_JUNC_NS:
                        light_color = GREEN_LIGHT_COLOR;
                        if (is_NS) {
                            yield = dir == DIRECTION_CCW;                   // Left turns yield
                        } else {
                            yield = true;                                       // All vehicles must yield
                        }
                        break;
                    case T_JUNC_EW:
                        light_color = GREEN_LIGHT_COLOR;
                        if (is_NS) {
                            yield = true;                                   // All vehicles must yield
                        } else {
                            yield = dir == DIRECTION_CCW;                   // Left turns yield
                        }
                        break;
                    case NS_GREEN_EW_RED:
                        if (is_NS) {
                            light_color = GREEN_LIGHT_COLOR;
                            yield = dir == DIRECTION_CCW;                   // Left turns yield
                        } else {
                            light_color = RED_LIGHT_COLOR;
                            yield = free_right && dir == DIRECTION_CW;      // Right turns yield
                        }
                        break;
                    case NS_YELLOW_EW_RED:
                        if (is_NS) {
                            light_color = YELLOW_LIGHT_COLOR;
                            yield = dir == DIRECTION_CCW;                   // Left turns yield
                        } else {
                            light_color = RED_LIGHT_COLOR;
                            yield = free_right && dir == DIRECTION_CW;      // Right turns yield
                        }
                        break;
                    case ALL_RED_BEFORE_EW_GREEN:
                        light_color = RED_LIGHT_COLOR;
                        yield = free_right && dir == DIRECTION_CW;          // Right turns yield
                        break;
                    case NS_RED_EW_GREEN:
                        if (is_NS) {
                            light_color = RED_LIGHT_COLOR;
                            yield = free_right && dir == DIRECTION_CW;      // Right turns yield
                        } else {
                            light_color = GREEN_LIGHT_COLOR;
                            yield = dir == DIRECTION_CCW;                   // Left turns yield
                        }
                        break;
                    case EW_YELLOW_NS_RED:
                        if (is_NS) {
                            light_color = RED_LIGHT_COLOR;
                            yield = free_right && dir == DIRECTION_CW;      // Right turns yield
                        } else {
                            light_color = YELLOW_LIGHT_COLOR;
                            yield = dir == DIRECTION_CCW;                   // Left turns yield
                        }
                        break;
                    case ALL_RED_BEFORE_NS_GREEN:
                        light_color = RED_LIGHT_COLOR;
                        yield = free_right && dir == DIRECTION_CW;          // Right turns yield
                        break;
                    default:
                        light_color = RED_LIGHT_COLOR;
                        break;
                }

                render_lane_center_line(renderer, lane, map, light_color, yield);
            }
        }

        end = SDL_GetPerformanceCounter();
        elapsed_us = (double)(end - start) * 1e6 / SDL_GetPerformanceFrequency();
        if (benchmark) printf("Traffic Lights: %.2f us\n", elapsed_us);
    }

    if (HIGHLIGHTED_CARS[0] != ID_NULL) {
        highlighted_car = sim_get_car(sim, HIGHLIGHTED_CARS[0]);
        situational_awareness_highlighted_car = sim_get_situational_awareness(sim, highlighted_car->id);
        if (situational_awareness_highlighted_car) {
            situational_awareness_highlighted_car->is_valid = false; // situational awareness has lots of pointers inside, which cannot be copied across processes (remember rendering happens in a separate process than the simulation and the sim state is communicated over tcp), so we need to rebuild it.
            situational_awareness_build(sim, highlighted_car->id);
        }
    }
                

    // Render all cars in the simulation
    if (draw_cars) {
        start = SDL_GetPerformanceCounter();

        // for the first highlighted car, extract its nearby vehicles
        HIGHLIGHTED_NEARBY_VEHICLES.count = 0; // by default, no nearby vehicles
        HIGHLIGHTED_CAR_AEB_ENGAGED = false;
        if (highlighted_car) {
            SituationalAwareness *situation = situational_awareness_highlighted_car;
            nearby_vehicles_flatten(&situation->nearby_vehicles, &HIGHLIGHTED_NEARBY_VEHICLES);
            if (sim->is_agent_driving_assistant_enabled) {
                DrivingAssistant* das = sim_get_driving_assistant(sim, highlighted_car->id);
                if (das) {
                    HIGHLIGHTED_CAR_AEB_ENGAGED = das->aeb_in_progress;
                }
            }
        }

        for (int i = 0; i < sim->num_cars; i++) {
            Car* car = sim_get_car(sim, i);
            render_car(renderer, car, map, draw_car_ids, draw_car_speeds);
        }

        end = SDL_GetPerformanceCounter();
        elapsed_us = (double)(end - start) * 1e6 / SDL_GetPerformanceFrequency();
        if (benchmark) printf("Cars: %.2f us\n", elapsed_us);
    }

    // render lidar of the car that the camera is centered on
    if (draw_lidar && highlighted_car) {
        Car* highlighted_car = sim_get_car(sim, HIGHLIGHTED_CARS[0]);
        if (highlighted_car_lidar == NULL) {
            highlighted_car_lidar = lidar_malloc((Coordinates){0,0}, 0.0, 360, from_degrees(360), from_feet(255.0)); // 255 feet max depth
        }
        if (highlighted_car && highlighted_car_lidar) {
            highlighted_car_lidar->position = highlighted_car->center;
            highlighted_car_lidar->orientation = highlighted_car->orientation;
            void* exclude_objects[] = {(void*)highlighted_car, NULL};
            lidar_capture_exclude_objects(highlighted_car_lidar, sim, exclude_objects);
            render_lidar(renderer, highlighted_car_lidar);
        }
    }

    // here render the camera (attached to the highlighted car) bonut.
    if (draw_camera && highlighted_car) {
        Car* highlighted_car = sim_get_car(sim, HIGHLIGHTED_CARS[0]);
        if (highlighted_car_camera == NULL) {
            // Create a camera with 128x128 resolution, 90 degree fov, 500 meter max depth
            highlighted_car_camera = rgbcam_malloc((Coordinates){0,0}, 1.0, 0.0, 128, 128, from_degrees(90), meters(500.0));
            rgbcam_set_aa_level(highlighted_car_camera, 2);  // 2x anti-aliasing
        }
        if (highlighted_car && highlighted_car_camera) {
            rgbcam_attach_to_car(highlighted_car_camera, highlighted_car, HIGHLIGHTED_CAR_CAMERA_TYPE);
            rgbcam_capture(highlighted_car_camera, sim);
            render_camera(renderer, highlighted_car_camera);
        }
    }

        // Render infos display for highlighted car
    if (draw_infos_display && highlighted_car) {
        Car* highlighted_car = sim_get_car(sim, HIGHLIGHTED_CARS[0]);
        if (highlighted_car_infos_display == NULL) {
            highlighted_car_infos_display = infos_display_malloc(128, 128, 9);
        }
        
        if (highlighted_car && highlighted_car_infos_display) {
            infos_display_clear(highlighted_car_infos_display);
            
            int info_idx = 0;
            // Populate data

            // Fuel
            infos_display_set_info(highlighted_car_infos_display, info_idx++, highlighted_car->fuel_level / highlighted_car->fuel_tank_capacity);

            // Speed (normalized to ~120mph)
            double speed_mph = to_mph(highlighted_car->speed);
            // set blue color
            infos_display_set_info_color(highlighted_car_infos_display, info_idx, (RGB){0, 0, 255});
            infos_display_set_info(highlighted_car_infos_display, info_idx++, fmin(1.0, fmax(0.0, speed_mph / 120.0)));

            // Acceleration
            double accel = (highlighted_car->acceleration) / 12.0;
            // color is green for positive acceleration, red for negative acceleration
            if (highlighted_car->acceleration >= 0) {
                infos_display_set_info_color(highlighted_car_infos_display, info_idx, (RGB){0, 255, 0});
            } else {
                infos_display_set_info_color(highlighted_car_infos_display, info_idx, (RGB){255, 0, 0});
                accel = -accel;
            }
            infos_display_set_info(highlighted_car_infos_display, info_idx++, fmin(1.0, fmax(0.0, accel)));
            
            // Indicators
            // Turn
            // color is red
            infos_display_set_info_color(highlighted_car_infos_display, info_idx, (RGB){255, 0, 0});
            infos_display_set_info(highlighted_car_infos_display, info_idx++, (double)(highlighted_car->indicator_turn == INDICATOR_LEFT));
            info_idx++;  // leave middle empty
            infos_display_set_info_color(highlighted_car_infos_display, info_idx, (RGB){255, 0, 0});
            infos_display_set_info(highlighted_car_infos_display, info_idx++, (double)(highlighted_car->indicator_turn == INDICATOR_RIGHT));
            
            // Merge left
            // orange color
            infos_display_set_info_color(highlighted_car_infos_display, info_idx, (RGB){255, 165, 0});
            infos_display_set_info(highlighted_car_infos_display, info_idx++, (double)(highlighted_car->indicator_lane == INDICATOR_LEFT));

            // Stop sign FCFS
            // green color
            infos_display_set_info_color(highlighted_car_infos_display, info_idx, RGB_GREEN);
            infos_display_set_info(highlighted_car_infos_display, info_idx++, (double)(situational_awareness_highlighted_car->is_my_turn_at_stop_sign_fcfs));

            // Merge right
            // orange color
            infos_display_set_info_color(highlighted_car_infos_display, info_idx, (RGB){255, 165, 0});
            infos_display_set_info(highlighted_car_infos_display, info_idx++, (double)(highlighted_car->indicator_lane == INDICATOR_RIGHT));
            
            infos_display_render(highlighted_car_infos_display);
            render_infos_display(renderer, highlighted_car_infos_display);
        }
    }

    // Render minimap for highlighted car
    if (draw_minimap && highlighted_car) {
        Car* highlighted_car = sim_get_car(sim, HIGHLIGHTED_CARS[0]);
        if (highlighted_car_minimap == NULL) {
            highlighted_car_minimap = minimap_malloc(256, 256, map);
            highlighted_car_path_planner = path_planner_create(sim_get_map(sim), false);
        }
        
        if (highlighted_car && highlighted_car_minimap) {
            highlighted_car_minimap->marked_car_id = highlighted_car->id;
            
            Lane* lane_goal = map_get_lane(map, sim->agent_goal_lane_id);
            if (lane_goal) {
                highlighted_car_minimap->marked_landmarks[0] = lane_goal->center;
                highlighted_car_minimap->marked_landmark_colors[0] = (RGB){0, 0, 255};
                path_planner_compute_shortest_path(highlighted_car_path_planner, map_get_lane(map, highlighted_car->lane_id), car_get_lane_progress_meters(highlighted_car), lane_goal, 0.5 * lane_goal->length);
                highlighted_car_minimap->marked_path = highlighted_car_path_planner;
            }
            
            minimap_render(highlighted_car_minimap, sim);
            render_minimap(renderer, highlighted_car_minimap);
        }
    }

    // render collisions
    if (collisions == NULL) {
        collisions = collisions_malloc();
    }
    // for each collision, render a red line between the two colliding cars
    if (collisions) {
        collisions_detect_all(collisions, sim);
        render_collisions(renderer, sim, collisions);
    }

    // Render time stats
    char time_stats[40];
    ClockReading clock_reading = sim_get_clock_reading(sim);
    snprintf(time_stats, sizeof(time_stats), "%s %02d:%02d:%02d  (%s %.1fx)",
         day_of_week_strings[clock_reading.day_of_week],
         clock_reading.hours, clock_reading.minutes, (int)clock_reading.seconds,
         sim->is_paused ? "⏸" : (approxeq(sim->simulation_speedup, 1.0, 1e-6) ? "⏵" : "⏩"),
         sim->simulation_speedup);
    render_text(renderer, time_stats, WINDOW_SIZE_WIDTH - 10, 10, 255, 255, 255, 255,
                hud_font_size, ALIGN_TOP_RIGHT, false, NULL);

    // Render weather stats
    render_text(renderer, weather_strings[sim->weather], WINDOW_SIZE_WIDTH - 10, 20 + hud_font_size,
                255, 255, 255, 255, hud_font_size, ALIGN_TOP_RIGHT, false, NULL);

    // Render driving assistant status on top left
    Car* car0 = sim_get_car(sim, CAMERA_CENTERED_ON_CAR_ID);
    if (car0) situational_awareness_build(sim, CAMERA_CENTERED_ON_CAR_ID);
    SituationalAwareness* sit = car0 ? sim_get_situational_awareness(sim, CAMERA_CENTERED_ON_CAR_ID) : NULL;
    DrivingAssistant* das = car0 ? sim_get_driving_assistant(sim, car0->id) : NULL;

    
    // If smart das in enabled, print "Style $style" in cyan color
    if (das && das->smart_das) {
        char smart_das_str[64];
        const char* style_str = NULL;
        switch (das->smart_das_driving_style) {
            case SMART_DAS_DRIVING_STYLE_NO_RULES:
                style_str = "Lawless"; break;
            case SMART_DAS_DRIVING_STYLE_RECKLESS:
                style_str = "Reckless"; break;
            case SMART_DAS_DRIVING_STYLE_AGGRESSIVE:
                style_str = "Aggressive"; break;
            case SMART_DAS_DRIVING_STYLE_NORMAL:
                style_str = "Normal"; break;
            case SMART_DAS_DRIVING_STYLE_DEFENSIVE:
                style_str = "Defensive"; break;
            default:
                style_str = "Unknown"; break;
        }
        snprintf(smart_das_str, sizeof(smart_das_str), " Style: %s", style_str);
        render_text(renderer, smart_das_str, 10, 80 + 5 * hud_font_size, 0, 255, 255, 255,
                    hud_font_size, ALIGN_TOP_LEFT, false, NULL);
    }

    // print target speed in hot pink
    if (das) {
        char target_speed_stats[32];
        Uint8 r = 255, g = 105, b = 180; // hot pink
        snprintf(target_speed_stats, sizeof(target_speed_stats), " Speed:   %.1f mph", to_mph(das->speed_target));
        render_text(renderer, target_speed_stats, 10, 90 + 6 * hud_font_size, r, g, b, 255,
                    hud_font_size, ALIGN_TOP_LEFT, false, NULL);
    }

    // If FOLLOW is enabled and there is car ahead and speed target is higher than the lead car's speed, print that in green light color, else set alpha to low value.
    if (das) {
        char follow_str[32];
        snprintf(follow_str, sizeof(follow_str), " Follow:   %.1f s + %.1f m", das->thw, das->buffer);
        Uint8 r = GREEN_LIGHT_COLOR.r, g = GREEN_LIGHT_COLOR.g, b = GREEN_LIGHT_COLOR.b;
        Uint8 alpha = sit->nearby_vehicles.lead && das->speed_target > sit->nearby_vehicles.lead->speed ? 255 : 64;
        render_text(renderer, follow_str, 10, 100 + 7 * hud_font_size, r, g, b, alpha,
                    hud_font_size, ALIGN_TOP_LEFT, false, NULL);
    }

    // If STOP is enabled (and smart das not enabled), print that in light blue color, else set alpha to low value. If smart das is enabled, insted of STOP, write "APPROACH" when approaching intersection engaged.
    if (das) {
        char stop_str[32];
        snprintf(stop_str, sizeof(stop_str), das->smart_das ? " Approach" : " Stop");
        Uint8 r = 173, g = 216, b = 230; // light blue
        Uint8 alpha = (!das->smart_das && das->should_stop_at_intersection && (sit->is_an_intersection_upcoming || sit->is_approaching_dead_end)) || (das->smart_das && das->smart_das_intersection_approach_engaged) ? 255 : 64;
        render_text(renderer, stop_str, 10, 110 + 8 * hud_font_size, r, g, b, alpha,
                    hud_font_size, ALIGN_TOP_LEFT, false, NULL);
    }

    // If AEB is engaged, print that in bright red text, else set alpha to low value.
    if (das) {
        char aeb_str[8];
        snprintf(aeb_str, sizeof(aeb_str), " AEB");
        Uint8 alpha = das->aeb_in_progress ? 255 : 64;
        render_text(renderer, aeb_str, 10, 120 + 9 * hud_font_size, 255, 0, 0, alpha,
                    hud_font_size, ALIGN_TOP_LEFT, false, NULL);
    }
}
