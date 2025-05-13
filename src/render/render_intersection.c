#include "render.h"

void render_intersection(SDL_Renderer* renderer, const Intersection* intersection) {
    // Get physical world bounds
    Meters x_min = intersection->road_eastbound_from->end_point.x;
    Meters x_max = intersection->road_eastbound_to->start_point.x;
    Meters y_min = intersection->road_northbound_from->end_point.y;
    Meters y_max = intersection->road_northbound_to->start_point.y;

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

    // Compute radius and pad with margin
    int radius_px = (int)(intersection->turn_radius * 1.45 * SCALE);
    int padded_x = screen_x_left - radius_px;
    int padded_y = screen_y_top - radius_px;
    int padded_width = screen_width + 2 * radius_px;
    int padded_height = screen_height + 2 * radius_px;
    int corner_radius = 2 * radius_px;

    // Draw gray intersection rectangle
    SDL_SetRenderDrawColor(renderer, 128, 128, 128, 255);
    drawFilledInwardRoundedRect(renderer, padded_x, padded_y, padded_width, padded_height, corner_radius);
}
