#include "map.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void map_init(Map* self) {
    self->num_lanes = 0;
    self->num_roads = 0;
    self->num_intersections = 0;
}

Lane* map_get_new_lane(Map* self) {
    if (self->num_lanes >= MAX_NUM_LANES) {
        LOG_ERROR("Map is full, cannot add more lanes");
        return NULL;
    }
    Lane* lane = &self->lanes[self->num_lanes++];
    lane->id = self->num_lanes - 1; // Assign a unique ID
    return lane;
}
Road* map_get_new_road(Map* self) {
    if (self->num_roads >= MAX_NUM_ROADS) {
        LOG_ERROR("Map is full, cannot add more roads");
        return NULL;
    }
    Road* road = &self->roads[self->num_roads++];
    road->id = self->num_roads - 1; // Assign a unique ID
    return road;
}
Intersection* map_get_new_intersection(Map* self) {
    if (self->num_intersections >= MAX_NUM_INTERSECTIONS) {
        LOG_ERROR("Map is full, cannot add more intersections");
        return NULL;
    }
    Intersection* intersection = &self->intersections[self->num_intersections++];
    intersection->id = self->num_intersections - 1; // Assign a unique ID
    return intersection;
}
Lane* map_get_lane(Map* self, LaneId id) {
    if (id < 0 || id >= self->num_lanes) {
        LOG_ERROR("Invalid lane ID: %d", id);
        return NULL;
    }
    return &self->lanes[id];
}
Road* map_get_road(Map* self, RoadId id) {
    if (id < 0 || id >= self->num_roads) {
        LOG_ERROR("Invalid road ID: %d", id);
        return NULL;
    }
    return &self->roads[id];
}
Road* map_get_road_by_lane_id(Map* self, LaneId id) {
    if (id < 0 || id >= self->num_lanes) {
        LOG_ERROR("Invalid lane ID: %d", id);
        return NULL;
    }
    Lane* lane = &self->lanes[id];
    if (lane->road_id < 0 || lane->road_id >= self->num_roads) {
        LOG_ERROR("Lane %d does not belong to a valid road", id);
        return NULL;
    }
    return &self->roads[lane->road_id];
}
Intersection* map_get_intersection(Map* self, IntersectionId id) {
    if (id < 0 || id >= self->num_intersections) {
        LOG_ERROR("Invalid intersection ID: %d", id);
        return NULL;
    }
    return &self->intersections[id];
}
Intersection* map_get_intersection_by_lane_id(Map* self, LaneId id) {
    if (id < 0 || id >= self->num_lanes) {
        LOG_ERROR("Invalid lane ID: %d", id);
        return NULL;
    }
    Lane* lane = &self->lanes[id];
    if (lane->intersection_id < 0 || lane->intersection_id >= self->num_intersections) {
        LOG_ERROR("Lane %d does not belong to a valid intersection", id);
        return NULL;
    }
    return &self->intersections[lane->intersection_id];
}
Lane* map_get_lane_by_name(Map* self, const char* name) {
    for (int i = 0; i < self->num_lanes; i++) {
        if (strcmp(self->lanes[i].name, name) == 0) {
            return &self->lanes[i];
        }
    }
    LOG_ERROR("Lane with name '%s' not found", name);
    return NULL;
}
Road* map_get_road_by_name(Map* self, const char* name) {
    for (int i = 0; i < self->num_roads; i++) {
        if (strcmp(self->roads[i].name, name) == 0) {
            return &self->roads[i];
        }
    }
    LOG_ERROR("Road with name '%s' not found", name);
    return NULL;
}
Intersection* map_get_intersection_by_name(Map* self, const char* name) {
    for (int i = 0; i < self->num_intersections; i++) {
        if (strcmp(self->intersections[i].name, name) == 0) {
            return &self->intersections[i];
        }
    }
    LOG_ERROR("Intersection with name '%s' not found", name);
    return NULL;
}

// --- Getters ---

Lane* map_get_lanes(Map* self) {
    return self->lanes;
}

int map_get_num_lanes(const Map* self) {
    return self->num_lanes;
}

Road* map_get_roads(Map* self) {
    return self->roads;
}

int map_get_num_roads(const Map* self) {
    return self->num_roads;
}

Intersection* map_get_intersections(Map* self) {
    return self->intersections;
}

int map_get_num_intersections(const Map* self) {
    return self->num_intersections;
}
