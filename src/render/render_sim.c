#include "utils.h"
#include "render.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

void render_sim(SDL_Renderer *renderer, const Simulation *sim, const bool draw_lanes, const bool draw_cars,
                const bool draw_track_lines, const bool draw_traffic_lights, const bool draw_car_ids, const bool draw_lane_ids, const bool benchmark)
{
    Map *map = sim->map;

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
            render_intersection(renderer, map->intersections[i]);
        }
        end = SDL_GetPerformanceCounter();
        elapsed_us = (double)(end - start) * 1e6 / SDL_GetPerformanceFrequency();
        if (benchmark) printf("Intersections: %.2f us\n", elapsed_us);

        // Straight roads
        start = SDL_GetPerformanceCounter();
        for (int i = 0; i < map->num_roads; i++) {
            const Road *road = map->roads[i];
            if (road->type == STRAIGHT) {
                for (int j = 0; j < road->num_lanes; j++) {
                    render_lane(renderer, road->lanes[j], true, true, draw_lane_ids);
                }
            }
        }
        end = SDL_GetPerformanceCounter();
        elapsed_us = (double)(end - start) * 1e6 / SDL_GetPerformanceFrequency();
        if (benchmark) printf("Straight Roads: %.2f us\n", elapsed_us);

        // Turn roads
        start = SDL_GetPerformanceCounter();
        for (int i = 0; i < map->num_roads; i++) {
            const Road *road = map->roads[i];
            if (road->type == TURN) {
                for (int j = 0; j < road->num_lanes; j++) {
                    render_lane(renderer, road->lanes[j], true, true, draw_lane_ids);
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
            const Road *road = map->roads[i];
            for (int j = 0; j < road->num_lanes; j++) {
                render_lane_center_line(renderer, road->lanes[j], LANE_CENTER_LINE_COLOR);
            }
        }

        // Optional: draw center lines at intersections if traffic lights are not drawn
        if (!draw_traffic_lights) {
            for (int i = 0; i < map->num_intersections; i++) {
                const Intersection *intersection = map->intersections[i];
                for (int j = 0; j < intersection->base.num_lanes; j++) {
                    render_lane_center_line(renderer, intersection->base.lanes[j], LANE_CENTER_LINE_COLOR);
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
            const Intersection *intersection = map->intersections[i];
            IntersectionState state = intersection->state;

            for (int j = 0; j < intersection->base.num_lanes; j++) {
                const Lane *lane = intersection->base.lanes[j];
                if (!lane) continue;

                Direction dir = lane->direction;
                const Lane *next_lane = lane->for_straight_connects_to;
                Direction next_dir = next_lane ? next_lane->direction : dir;

                bool is_vert = dir == DIRECTION_NORTH || dir == DIRECTION_SOUTH;
                bool is_turn = lane->type == QUARTER_ARC_LANE;
                bool next_is_hori = next_dir == DIRECTION_EAST || next_dir == DIRECTION_WEST;
                bool is_NS = is_vert || (is_turn && next_is_hori);

                SDL_Color light_color = RED_LIGHT_COLOR;

                switch (state) {
                    case NS_GREEN_EW_RED:
                        light_color = is_NS ? GREEN_LIGHT_COLOR : RED_LIGHT_COLOR;
                        break;
                    case NS_YELLOW_EW_RED:
                        light_color = is_NS ? YELLOW_LIGHT_COLOR : RED_LIGHT_COLOR;
                        break;
                    case NS_RED_EW_GREEN:
                        light_color = is_NS ? RED_LIGHT_COLOR : GREEN_LIGHT_COLOR;
                        break;
                    case EW_YELLOW_NS_RED:
                        light_color = is_NS ? RED_LIGHT_COLOR : YELLOW_LIGHT_COLOR;
                        break;
                    case FOUR_WAY_STOP:
                        printf("4-way!\n");
                        light_color = FOUR_WAY_STOP_COLOR;
                        break;
                    default:
                        light_color = RED_LIGHT_COLOR;
                        break;
                }

                // Special case: allow right turns on red
                if (dir == DIRECTION_CW) {
                    light_color = GREEN_LIGHT_COLOR;
                }

                render_lane_center_line(renderer, lane, light_color);
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
            render_car(renderer, sim->cars[i], draw_car_ids);
        }

        end = SDL_GetPerformanceCounter();
        elapsed_us = (double)(end - start) * 1e6 / SDL_GetPerformanceFrequency();
        if (benchmark) printf("Cars: %.2f us\n", elapsed_us);
    }
}
