#include "utils.h"
#include "render.h"
#include "logging.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL2/SDL_ttf.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

SDL_Texture* road_name_texture_cache[MAX_NUM_ROADS][MAX_FONT_SIZE] = {{NULL}};
SDL_Texture* intersection_name_texture_cache[MAX_NUM_INTERSECTIONS][MAX_FONT_SIZE] = {{NULL}};


void render_straight_road_name(SDL_Renderer* renderer, const Road* road,  Map* map) {
    if (!road) {
        LOG_ERROR("Cannot render road name: road is NULL");
        return;
    }
    const char* name = road->name;
    Direction direction = road->direction;
    Lane* rightmost_lane = road_get_rightmost_lane(road, map);
    Coordinates rightmost_lane_center = rightmost_lane->center;
    Vec2D perp = direction_perpendicular_vector(direction_opposite(direction));
    if (road->id < 4) {
        perp = vec_scale(perp, -6); // todo: This this is a hack to render on opposite side for first 4 roads, which correspond to outer ring road. This is to avoid overlapping text.
    }
    Coordinates text_center = vec_add(rightmost_lane_center, vec_scale(perp, road->lane_width / 2 + meters(1)));
    SDL_Point text_center_screen = to_screen_coords(text_center, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
    int text_x = text_center_screen.x;
    int text_y = text_center_screen.y;
    int font_size = (int)(meters(3) * SCALE);

    TextAlign align;
    if (direction == DIRECTION_EAST) {
        align = ALIGN_TOP_CENTER;
    } else if (direction == DIRECTION_WEST) {
        align = ALIGN_BOTTOM_CENTER;
    } else if (direction == DIRECTION_NORTH) {
        align = ALIGN_CENTER_LEFT;
    } else if (direction == DIRECTION_SOUTH) {
        align = ALIGN_CENTER_RIGHT;
    } else {
        LOG_ERROR("Unknown direction %d for road ID %d", direction, road->id);
        return;
    }

    bool rotated = (direction == DIRECTION_NORTH || direction == DIRECTION_SOUTH);
    render_text(renderer, name, text_x, text_y, 255, 255, 255, 255, font_size, align, rotated, road_name_texture_cache[road->id]);
}


void render_intersection_name(SDL_Renderer* renderer, const Intersection* intersection, Map* map) {
    if (!intersection) {
        LOG_ERROR("Cannot render intersection name: intersection is NULL");
        return;
    }
    const char* name = intersection->name;
    Coordinates center = intersection->center;
    SDL_Point center_screen = to_screen_coords(center, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
    int text_x = center_screen.x;
    int text_y = center_screen.y;
    int font_size = (int)(meters(2) * SCALE);
    TextAlign align = ALIGN_CENTER;
    render_text(renderer, name, text_x, text_y, 255, 255, 255, 255, font_size, align, false, intersection_name_texture_cache[intersection->id]);
}