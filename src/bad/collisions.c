#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "sim.h"
#include "car.h"
#include "bad.h"
#include "logging.h"

static bool collisions_valid_index(CarId car_id) {
    return car_id >= 0 && car_id < MAX_CARS_IN_SIMULATION;
}

static bool collisions_has_pair(Collisions* collisions, CarId a, CarId b) {
    if (!collisions_valid_index(a)) return false;
    for (int i = 0; i < COLLISIONS_MAX_PER_CAR; i++) {
        if (collisions->colliding_cars[a][i] == b) return true;
    }
    return false;
}

// Pack a car's collision slots so valid IDs occupy [0, num_collisions_per_car)
// and trailing entries are set to -1. Returns the new count.
static int collisions_pack_car_slots(Collisions* collisions, CarId car_id) {
    if (!collisions || !collisions_valid_index(car_id)) return 0;

    int write_idx = 0;
    for (int read_idx = 0; read_idx < COLLISIONS_MAX_PER_CAR; read_idx++) {
        CarId other = collisions->colliding_cars[car_id][read_idx];
        if (other >= 0) {
            collisions->colliding_cars[car_id][write_idx++] = other;
        }
    }
    for (int i = write_idx; i < COLLISIONS_MAX_PER_CAR; i++) {
        collisions->colliding_cars[car_id][i] = -1;
    }

    collisions->num_collisions_per_car[car_id] = write_idx;
    return write_idx;
}

static bool collisions_add_pair(Collisions* collisions, CarId a, CarId b) {
    if (!collisions_valid_index(a) || !collisions_valid_index(b) || a == b) return false;
    if (collisions_has_pair(collisions, a, b)) return false;

    int slot_a = -1;
    int slot_b = -1;
    for (int i = 0; i < COLLISIONS_MAX_PER_CAR; i++) {
        if (collisions->colliding_cars[a][i] == -1) slot_a = slot_a == -1 ? i : slot_a;
        if (collisions->colliding_cars[b][i] == -1) slot_b = slot_b == -1 ? i : slot_b;
    }

    if (slot_a == -1 || slot_b == -1) {
        LOG_WARN("Collision capacity reached for car %d or %d; additional collisions ignored", a, b);
        return false;
    }

    collisions->colliding_cars[a][slot_a] = b;
    collisions->colliding_cars[b][slot_b] = a;
    collisions->num_collisions_per_car[a]++;
    collisions->num_collisions_per_car[b]++;
    collisions->total_collisions++;
    return true;
}

static bool collisions_remove_pair(Collisions* collisions, CarId a, CarId b) {
    if (!collisions_valid_index(a) || !collisions_valid_index(b) || a == b) return false;
    if (!collisions_has_pair(collisions, a, b)) return false;

    for (int i = 0; i < COLLISIONS_MAX_PER_CAR; i++) {
        if (collisions->colliding_cars[a][i] == b) collisions->colliding_cars[a][i] = -1;
        if (collisions->colliding_cars[b][i] == a) collisions->colliding_cars[b][i] = -1;
    }

    // Keep per-car collision arrays packed (no -1 gaps) so that indices
    // [0, num_collisions_per_car) are always valid collision partners.
    collisions_pack_car_slots(collisions, a);
    collisions_pack_car_slots(collisions, b);

    if (collisions->total_collisions > 0) collisions->total_collisions--;
    return true;
}

static void collisions_reset_car(Collisions* collisions, CarId car_id) {
    if (!collisions_valid_index(car_id)) return;
    for (int i = 0; i < COLLISIONS_MAX_PER_CAR; i++) {
        collisions->colliding_cars[car_id][i] = -1;
    }
    collisions->num_collisions_per_car[car_id] = 0;
}

Collisions* collisions_malloc() {
    Collisions* collisions = (Collisions*)malloc(sizeof(Collisions));
    if (!collisions) {
        LOG_ERROR("Failed to allocate Collisions");
        return NULL;
    }
    collisions_reset(collisions);
    return collisions;
}

void collisions_free(Collisions* collisions) {
    if (!collisions) {
        LOG_ERROR("Attempted to free a NULL Collisions pointer");
        return;
    }
    free(collisions);
}

void collisions_reset(Collisions* collisions) {
    if (!collisions) {
        LOG_ERROR("Attempted to reset a NULL Collisions pointer");
        return;
    }
    for (int i = 0; i < MAX_CARS_IN_SIMULATION; i++) {
        collisions_reset_car(collisions, i);
    }
    collisions->total_collisions = 0;
}

int collisions_get_total(Collisions* collisions) {
    return collisions ? collisions->total_collisions : 0;
}

int collisions_get_count_for_car(Collisions* collisions, CarId car_id) {
    if (!collisions || !collisions_valid_index(car_id)) return 0;
    return collisions->num_collisions_per_car[car_id];
}

bool collisions_are_colliding(Collisions* collisions, CarId car1, CarId car2) {
    if (!collisions) return false;
    return collisions_has_pair(collisions, car1, car2);
}

void collisions_detect_all(Collisions* collisions, Simulation* sim) {
    if (!collisions || !sim) {
        LOG_ERROR("Cannot detect collisions: collisions=%p, sim=%p", (void*)collisions, (void*)sim);
        return;
    }

    collisions_reset(collisions);

    int num_cars = sim_get_num_cars(sim);
    for (int i = 0; i < num_cars; i++) {
        Car* car_i = sim_get_car(sim, i);
        if (!car_i) continue;
        for (int j = i + 1; j < num_cars; j++) {
            Car* car_j = sim_get_car(sim, j);
            if (!car_j) continue;
            if (car_is_colliding(car_i, car_j)) {
                collisions_add_pair(collisions, car_i->id, car_j->id);
            }
        }
    }
}

void collisions_detect_for_car(Collisions* collisions, Simulation* sim, CarId car_id) {
    if (!collisions || !sim) {
        LOG_ERROR("Cannot detect collisions for car: collisions=%p, sim=%p", (void*)collisions, (void*)sim);
        return;
    }
    if (!collisions_valid_index(car_id)) {
        LOG_ERROR("Car id %d out of valid collisions bounds", car_id);
        return;
    }

    int num_cars = sim_get_num_cars(sim);
    if (car_id < 0 || car_id >= num_cars) {
        LOG_ERROR("Car id %d not present in simulation (num_cars=%d)", car_id, num_cars);
        return;
    }

    // Remove existing pairs involving this car.
    for (int j = 0; j < MAX_CARS_IN_SIMULATION; j++) {
        if (j == car_id) continue;
        collisions_remove_pair(collisions, car_id, j);
    }

    // Recompute collisions for this car.
    Car* car_i = sim_get_car(sim, car_id);
    if (!car_i) return;
    for (int j = 0; j < num_cars; j++) {
        if (j == car_id) continue;
        Car* car_j = sim_get_car(sim, j);
        if (!car_j) continue;
        if (car_is_colliding(car_i, car_j)) {
            collisions_add_pair(collisions, car_i->id, car_j->id);
        }
    }
}

CarId collisions_get_colliding_car(Collisions* collisions, CarId car_id, int collision_index) {
    if (!collisions || !collisions_valid_index(car_id) || collision_index < 0 || collision_index >= COLLISIONS_MAX_PER_CAR) {
        LOG_ERROR("Invalid input parameters for collisions_get_colliding_car");
        return -1;
    }

    return collisions->colliding_cars[car_id][collision_index];
}

void collisions_print(Collisions* collisions) {
    if (!collisions) {
        LOG_ERROR("Attempted to print a NULL Collisions pointer");
        return;
    }

    printf("Collision summary: total unique collisions = %d\n", collisions->total_collisions);
    printf("Per-car collisions:\n");
    for (int i = 0; i < MAX_CARS_IN_SIMULATION; i++) {
        if (collisions->num_collisions_per_car[i] > 0) {
            printf("  Car %d: %d\n", i, collisions->num_collisions_per_car[i]);
        }
    }

    printf("Collision pairs:\n");
    for (int i = 0; i < MAX_CARS_IN_SIMULATION; i++) {
        for (int k = 0; k < COLLISIONS_MAX_PER_CAR; k++) {
            CarId other = collisions->colliding_cars[i][k];
            if (other >= 0 && i < other) {
                printf("  %d <-> %d\n", i, other);
            }
        }
    }
}

