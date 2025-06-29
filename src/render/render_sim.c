#include "utils.h"
#include "render.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

void render_sim(SDL_Renderer *renderer, Simulation *sim, const bool draw_lanes, const bool draw_cars,
                const bool draw_track_lines, const bool draw_traffic_lights, const bool draw_car_ids, const bool draw_lane_ids, const bool draw_road_names, int hud_font_size, const bool benchmark)
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

    // Render all cars in the simulation
    if (draw_cars) {
        start = SDL_GetPerformanceCounter();

        for (int i = 0; i < sim->num_cars; i++) {
            Car* car = sim_get_car(sim, i);
            render_car(renderer, car, map, draw_car_ids);
        }

        end = SDL_GetPerformanceCounter();
        elapsed_us = (double)(end - start) * 1e6 / SDL_GetPerformanceFrequency();
        if (benchmark) printf("Cars: %.2f us\n", elapsed_us);
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
}
