#pragma once

#include "sim.h"

#define COLLISIONS_MAX_PER_CAR 2

// Sparse collision storage: track up to two current collisions per car.
typedef struct Collisions {
    CarId colliding_cars[MAX_CARS_IN_SIMULATION][COLLISIONS_MAX_PER_CAR]; // Packed: indices [0, num_collisions_per_car) are valid IDs, rest are -1.
    int num_collisions_per_car[MAX_CARS_IN_SIMULATION];
    int total_collisions; // unique pairs currently tracked
} Collisions;

Collisions* collisions_malloc();
void collisions_free(Collisions* collisions);
void collisions_reset(Collisions* collisions);

int collisions_get_total(Collisions* collisions);
int collisions_get_count_for_car(Collisions* collisions, CarId car_id);
bool collisions_are_colliding(Collisions* collisions, CarId car1, CarId car2);
CarId collisions_get_colliding_car(Collisions* collisions, CarId car_id, int collision_index);

void collisions_detect_all(Collisions* collisions, Simulation* sim);
void collisions_detect_for_car(Collisions* collisions, Simulation* sim, CarId car_id);
void collisions_print(Collisions* collisions);