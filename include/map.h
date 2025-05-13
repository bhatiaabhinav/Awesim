#pragma once

#include "road.h"

#define MAX_NUM_ROADS 256
#define MAX_NUM_INTERSECTIONS 256

// Structure representing a simulation map containing roads and intersections
typedef struct Map {
    const Road* roads[MAX_NUM_ROADS];
    const Intersection* intersections[MAX_NUM_INTERSECTIONS];
    int num_roads;
    int num_intersections;
} Map;

// Creates and initializes a new Map instance
Map* map_create();

// Adds a generic Road to the map (StraightRoad, Turn, or Intersection)
void map_add_road(Map* self, const Road* road);

// Adds a StraightRoad to the map
void map_add_straight_road(Map* self, const StraightRoad* road);

// Adds a Turn to the map
void map_add_turn(Map* self, const Turn* turn);

// Adds an Intersection to the map
void map_add_intersection(Map* self, const Intersection* intersection);

// --- Getters ---

// Returns pointer to array of roads
const Road** map_get_roads(Map* self);

// Returns number of roads currently in the map
int map_get_num_roads(const Map* self);

// Returns pointer to array of intersections
const Intersection** map_get_intersections(Map* self);

// Returns number of intersections currently in the map
int map_get_num_intersections(const Map* self);

// Frees memory associated with map (does not free roads/intersections themselves)
void map_free(Map* self);


// Frees memory associated with map and all roads and intersections
void map_deep_free(Map* self);
