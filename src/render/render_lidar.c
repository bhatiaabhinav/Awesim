#include "utils.h"
#include "render.h"
#include "logging.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>


void render_lidar(SDL_Renderer* renderer, const Lidar* lidar) {
    for (int i = 0; i < lidar->num_points; i++) {
        double scan_progress = lidar->num_points == 1 ? 0.5 : (float) i / (lidar->num_points - 1); // when num_points is 1, just point in the center
        Radians ray_orientation = lidar->orientation + lidar->fov * (scan_progress - 0.5);
        Vec2D ray_unit_vector = angle_to_unit_vector(ray_orientation);
        Coordinates end_point = vec_add(lidar->position, vec_scale(ray_unit_vector, lidar->data[i]));
        SDL_Point screen_start = to_screen_coords(lidar->position, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
        SDL_Point screen_end = to_screen_coords(end_point, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
        thickLineRGBA_ignore_if_outside_screen(renderer, screen_start.x, screen_start.y, screen_end.x, screen_end.y, 1, 255, 0, 0, 64);
    }
}