#include "utils.h"
#include "render.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

void render_sim(SDL_Renderer *renderer, Simulation *sim, const bool draw_lanes, const bool draw_cars,
                const bool draw_track_lines, const bool draw_traffic_lights, const bool draw_car_ids, const bool draw_car_speeds, const bool draw_lane_ids, const bool draw_road_names, int hud_font_size, const bool benchmark)
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

        // for the first highlighted car, extract its nearby vehicles
        HIGHLIGHTED_NEARBY_VEHICLES.count = 0; // by default, no nearby vehicles
        HIGHLIGHTED_CAR_AEB_ENGAGED = false;
        if (HIGHLIGHTED_CARS[0] != -1) {
            Car *highlighted_car = sim_get_car(sim, HIGHLIGHTED_CARS[0]);
            if (highlighted_car) {
                SituationalAwareness *situation = sim_get_situational_awareness(sim, highlighted_car->id);
                if (situation) {
                    situation->is_valid = false; // situational awareness has lots of pointers inside, which cannot be copied across processes (remember rendering happens in a separate process than the simulation and the sim state is communicated over tcp), so we need to rebuild it.
                    situational_awareness_build(sim, highlighted_car->id);
                    nearby_vehicles_flatten(&situation->nearby_vehicles, &HIGHLIGHTED_NEARBY_VEHICLES, true);
                }
                if (sim->is_agent_driving_assistant_enabled) {
                    DrivingAssistant* das = sim_get_driving_assistant(sim, highlighted_car->id);
                    if (das) {
                        HIGHLIGHTED_CAR_AEB_ENGAGED = das->aeb_in_progress;
                    }
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


    if (HIGHLIGHTED_CARS[0] != ID_NULL) {
        // Render highlighted car 0 fuel level on top left (in gallon)

        Uint8 r = HIGHLIGHTED_CAR_COLOR.r, g = HIGHLIGHTED_CAR_COLOR.g, b = HIGHLIGHTED_CAR_COLOR.b;

        Car* car0 = sim_get_car(sim, HIGHLIGHTED_CARS[0]);
        if (car0) {
            char fuel_stats[32];
            snprintf(fuel_stats, sizeof(fuel_stats), "Fuel:       %.2f/%.1f gal", to_gallons(car0->fuel_level), to_gallons(car0->fuel_tank_capacity));
            render_text(renderer, fuel_stats, 10, 20 + hud_font_size, r, g, b, 255,
                        hud_font_size, ALIGN_TOP_LEFT, false, NULL);
        }

        // Render car 0 speed on top left
        if (car0) {
            char speed_stats[32];
            snprintf(speed_stats, sizeof(speed_stats), "Speed:   %.1f mph", to_mph(car0->speed));
            render_text(renderer, speed_stats, 10, 30 + 2 * hud_font_size, r, g, b, 255,
                        hud_font_size, ALIGN_TOP_LEFT, false, NULL);
        }

        // Render car 0 acceleration on top left
        if (car0) {
            char accel_stats[32];
            snprintf(accel_stats, sizeof(accel_stats), "Accel:     %.1f m/s²", car0->acceleration);
            render_text(renderer, accel_stats, 10, 40 + 3 * hud_font_size, r, g, b, 255,
                        hud_font_size, ALIGN_TOP_LEFT, false, NULL);
        }

        // Render indicator status of car 0 on top left. At a time, only one of merge or turn is active. So we pick one that is not INDICATOR_NONE. In parenthesis, we print whether it is turn or lane indicator.
        if (car0) {
            char indicator_stats[32];

            CarIndicator indicator = INDICATOR_NONE;
            bool is_turn_indicator = false;
            if (car0->indicator_turn != INDICATOR_NONE) {
                indicator = car0->indicator_turn;
                is_turn_indicator = true;
            } else if (car0->indicator_lane != INDICATOR_NONE) {
                indicator = car0->indicator_lane;
            }
            bool indicating_anything = indicator != INDICATOR_NONE;

            snprintf(indicator_stats, sizeof(indicator_stats), "%s (%s) %s", indicator == INDICATOR_LEFT ? "⬅" : "", indicating_anything ? (is_turn_indicator ? "Turn" : "Merge") : "⏺", indicator == INDICATOR_RIGHT ? "➡" : " ");
            render_text(renderer, indicator_stats, 10, 50 + 4 * hud_font_size, r, g, b, 255,
                        hud_font_size, ALIGN_TOP_LEFT, false, NULL);
        }
    }

    if (sim->is_agent_enabled && sim->is_agent_driving_assistant_enabled) {
        // Render driving assistant status on top left
        Car* car0 = sim_get_car(sim, 0);    // 0 is always the agent car
        situational_awareness_build(sim, 0); // rebuild situational awareness for car 0
        SituationalAwareness* sit = sim_get_situational_awareness(sim, 0);
        DrivingAssistant* das = car0 ? sim_get_driving_assistant(sim, car0->id) : NULL;


        // print "ADAS Targets:" header
        if (das) {
            Uint8 r = HIGHLIGHTED_CAR_COLOR.r, g = HIGHLIGHTED_CAR_COLOR.g, b = HIGHLIGHTED_CAR_COLOR.b;
            render_text(renderer, "ADAS Targets:", 10, 80 + 5 * hud_font_size, r, g, b, 255,
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
            snprintf(follow_str, sizeof(follow_str), " Follow:   %.1f s + %.1f ft", das->follow_mode_thw, to_feet(das->follow_mode_buffer));
            Uint8 r = GREEN_LIGHT_COLOR.r, g = GREEN_LIGHT_COLOR.g, b = GREEN_LIGHT_COLOR.b;
            Uint8 alpha = das->follow_mode && sit->lead_vehicle && das->speed_target > sit->lead_vehicle->speed ? 255 : 64;
            render_text(renderer, follow_str, 10, 100 + 7 * hud_font_size, r, g, b, alpha,
                        hud_font_size, ALIGN_TOP_LEFT, false, NULL);
        }

        // If STOP is enabled, print that in red light color, else set alpha to low value.
        if (das) {
            char stop_str[32];
            snprintf(stop_str, sizeof(stop_str), " Stop:      %.1f ft", to_feet(das->stopping_distance_target));
            Uint8 r = RED_LIGHT_COLOR.r, g = RED_LIGHT_COLOR.g, b = RED_LIGHT_COLOR.b;
            Uint8 alpha = das->stop_mode ? 255 : 64;
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
}
