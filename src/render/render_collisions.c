#include "render.h"

void render_collisions(SDL_Renderer* renderer, Simulation* sim, Collisions* collisions) {
    for (CarId car1 = 0; car1 < sim->num_cars; car1++) {
        int num_collisions = collisions_get_count_for_car(collisions, car1);
        for (int collision_idx = 0; collision_idx < num_collisions; collision_idx++) {
            CarId car2 = collisions_get_colliding_car(collisions, car1, collision_idx);
            if (car2 < car1) continue;  // Only draw each collision once
            Car* car1_obj = sim_get_car(sim, car1);
            Car* car2_obj = sim_get_car(sim, car2);
            if (car1_obj && car2_obj) {
                SDL_Point car1_center = to_screen_coords(car_get_center(car1_obj), WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
                SDL_Point car2_center = to_screen_coords(car_get_center(car2_obj), WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
                thickLineRGBA_ignore_if_outside_screen(renderer, car1_center.x, car1_center.y, car2_center.x, car2_center.y, 2, 255, 0, 0, 255);
                // printf("Car %d collided with Car %d\n", car1, car2);
            }
        }
    }
}