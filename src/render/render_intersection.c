#include "render.h"

void render_intersection(SDL_Renderer* renderer, const Intersection* intersection, Map* map) {
    // Get physical world bounds
    Road* road_eastbound_from = map_get_road(map, intersection->road_eastbound_from_id);
    Road* road_eastbound_to = map_get_road(map, intersection->road_eastbound_to_id);
    Road* road_northbound_from = map_get_road(map, intersection->road_northbound_from_id);
    Road* road_northbound_to = map_get_road(map, intersection->road_northbound_to_id);
    Meters x_min = road_eastbound_from->end_point.x;
    Meters x_max = road_eastbound_to->start_point.x;
    Meters y_min = road_northbound_from->end_point.y;
    Meters y_max = road_northbound_to->start_point.y;

    // Convert corners to screen coordinates
    SDL_Point screen_left = to_screen_coords((Coordinates){x_min, 0}, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
    SDL_Point screen_right = to_screen_coords((Coordinates){x_max, 0}, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
    SDL_Point screen_top = to_screen_coords((Coordinates){0, y_max}, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
    SDL_Point screen_bottom = to_screen_coords((Coordinates){0, y_min}, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);

    int screen_x_left = screen_left.x;
    int screen_x_right = screen_right.x;
    int screen_y_top = screen_top.y;
    int screen_y_bottom = screen_bottom.y;

    int screen_width = screen_x_right - screen_x_left;
    int screen_height = screen_y_bottom - screen_y_top;

    Meters corner_radius = intersection->turn_radius - 0.5 * intersection->lane_width;
    int corner_radius_px = (int)(corner_radius * SCALE);

    // Draw gray intersection rectangle
    SDL_SetRenderDrawColor(renderer, 128, 128, 128, 255);
    drawFilledInwardRoundedRect(renderer, screen_x_left, screen_y_top, screen_width, screen_height, corner_radius_px);
}
