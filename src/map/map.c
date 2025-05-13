#include "map.h"
#include <stdio.h>
#include <stdlib.h>

Map* map_create() {
    Map* map = (Map*)malloc(sizeof(Map));
    if (!map) {
        fprintf(stderr, "Memory allocation failed for Map\n");
        exit(EXIT_FAILURE);
    }
    map->num_roads = 0;
    map->num_intersections = 0;
    return map;
}

void map_add_road(Map* self, const Road* road) {
    if (!self || !road) return;
    if (self->num_roads >= MAX_NUM_ROADS) {
        fprintf(stderr, "Map is full, cannot add more roads\n");
        return;
    }
    self->roads[self->num_roads++] = road;
}

void map_add_straight_road(Map* self, const StraightRoad* road) {
    map_add_road(self, (const Road*)road);
}

void map_add_turn(Map* self, const Turn* turn) {
    map_add_road(self, (const Road*)turn);
}

void map_add_intersection(Map* self, const Intersection* intersection) {
    if (!self || !intersection) return;
    if (self->num_intersections >= MAX_NUM_INTERSECTIONS) {
        fprintf(stderr, "Map is full, cannot add more intersections\n");
        return;
    }
    self->intersections[self->num_intersections++] = intersection;
}

// --- Getters ---

const Road** map_get_roads(Map* self) {
    return self ? self->roads : NULL;
}

int map_get_num_roads(const Map* self) {
    return self ? self->num_roads : 0;
}

const Intersection** map_get_intersections(Map* self) {
    return self ? self->intersections : NULL;
}

int map_get_num_intersections(const Map* self) {
    return self ? self->num_intersections : 0;
}

// --- Memory Management ---

void map_free(Map* self) {
    free(self);
}

// Frees all roads and intersections and the map itself
void map_deep_free(Map* self) {
    if (!self) return;
    for (int i = 0; i < self->num_roads; i++) {
        road_free((Road*)self->roads[i]);
    }
    for (int i = 0; i < self->num_intersections; i++) {
        intersection_free((Intersection*)self->intersections[i]);
    }
    self->num_roads = 0;
    self->num_intersections = 0;
    free(self);
}
