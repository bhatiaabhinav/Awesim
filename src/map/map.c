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

void map_print(Map* self, FILE* file) {
    // First print all lanes, their properties, their incoming & outgoing connections, what roads they belong to, and what intersections they are at.
    fprintf(file, "Map Information:\n\n");
    fprintf(file, "Total Lanes: %d\n", self->num_lanes);
    for (int i = 0; i < self->num_lanes; i++) {
        Lane* lane = map_get_lane(self, i);
        fprintf(file, "Lane %d: Name: %s, Type: %d, Direction: %d, Width: %.2f m, Speed Limit: %.2f m/s, Grip: %.2f, Length: %.2f m\n",
                lane->id,
                lane->name,
                lane->type,
                lane->direction,
                lane->width,
                lane->speed_limit,
                lane->grip,
                lane->length);
        fprintf(file, "  Start Point: (%.2f, %.2f), End Point: (%.2f, %.2f), Mid Point: (%.2f, %.2f), Center: (%.2f, %.2f)\n",
                lane->start_point.x, lane->start_point.y,
                lane->end_point.x, lane->end_point.y,
                lane->mid_point.x, lane->mid_point.y,
                lane->center.x, lane->center.y);
        fprintf(file, "  Radius: %.2f m, Start Angle: %.2f rad, End Angle: %.2f rad, Quadrant: %d\n",
                lane->radius,
                lane->start_angle,
                lane->end_angle,
                lane->quadrant);
        fprintf(file, "  Road ID: %d, Intersection ID: %d, Is at Intersection: %s\n",
                lane->road_id,
                lane->intersection_id,
                lane->is_at_intersection ? "Yes" : "No");
        fprintf(file, "  Connections: Incoming Left: %d, Incoming Straight: %d, Incoming Right: %d\n",
                lane->connections_incoming[0],
                lane->connections_incoming[1],
                lane->connections_incoming[2]);
        fprintf(file, "  Connections: Left: %d, Straight: %d, Right: %d\n",
                lane->connections[0],
                lane->connections[1],
                lane->connections[2]);
        fprintf(file, "  Adjacents: Left: %d, Right: %d\n",
                lane->adjacents[0],
                lane->adjacents[1]);
        fprintf(file, "  Merges Into: Lane ID: %d, Start: %.2f, End: %.2f\n",
                lane->merges_into_id,
                lane->merges_into_start,
                lane->merges_into_end);
        fprintf(file, "  Exit Lane: Lane ID: %d, Start: %.2f, End: %.2f\n",
                lane->exit_lane_id,
                lane->exit_lane_start,
                lane->exit_lane_end);
        fprintf(file, "  Cars in Lane: ");
        for (int j = 0; j < lane->num_cars; j++) {
            fprintf(file, "%d ", lane->cars_ids[j]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    // Now print all roads, their properties, and the lanes they contain.
    fprintf(file, "Total Roads: %d\n", self->num_roads);
    for (int i = 0; i < self->num_roads; i++) {
        Road* road = map_get_road(self, i);
        fprintf(file, "Road %d: Name: %s, Type: %d, Direction: %d, Speed Limit: %.2f m/s, Grip: %.2f, Lane Width: %.2f m, Length: %.2f m, Width: %.2f m\n",
                road->id,
                road->name,
                road->type,
                road->direction,
                road->speed_limit,
                road->grip,
                road->lane_width,
                road->length,
                road->width);
        fprintf(file, "  Start Point: (%.2f, %.2f), End Point: (%.2f, %.2f), Mid Point: (%.2f, %.2f), Center: (%.2f, %.2f)\n",
                road->start_point.x, road->start_point.y,
                road->end_point.x, road->end_point.y,
                road->mid_point.x, road->mid_point.y,
                road->center.x, road->center.y);
        fprintf(file, "  Radius: %.2f m, Start Angle: %.2f rad, End Angle: %.2f rad, Quadrant: %d\n",
                road->radius,
                road->start_angle,
                road->end_angle,
                road->quadrant);
        fprintf(file, "  Road From ID: %d, Road To ID: %d\n",
                road->road_from_id,
                road->road_to_id);
        fprintf(file, "  Lanes: ");
        for (int j = 0; j < road->num_lanes; j++) {
            fprintf(file, "%d ", road->lane_ids[j]);
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
    // Finally, print all intersections, their properties, lanes they contain, and the roads they connect.
    fprintf(file, "Total Intersections: %d\n", self->num_intersections);
    for (int i = 0; i < self->num_intersections; i++) {
        Intersection* intersection = map_get_intersection(self, i);
        fprintf(file, "Intersection %d: Name: %s, Center: (%.2f, %.2f), Dimensions: %.2f x %.2f, Lane Width: %.2f m, Grip: %.2f, Speed limit: %.2f m/s, Turn radius: %.2f m, Left lane turns left only: %s, Right lane turns right only: %s\n",
                intersection->id,
                intersection->name,
                intersection->center.x, intersection->center.y,
                intersection->dimensions.x, intersection->dimensions.y,
                intersection->lane_width,
                intersection->grip,
                intersection->speed_limit,
                intersection->turn_radius,
                intersection->left_lane_turns_left_only ? "Yes" : "No",
                intersection->right_lane_turns_right_only ? "Yes" : "No");
        fprintf(file, "  Lanes: ");
        for (int j = 0; j < intersection->num_lanes; j++) {
            fprintf(file, "%d ", intersection->lane_ids[j]);
        }
        // Incoming roads (road_eastbound_from_id, road_westbound_from_id, etc.) names:
        fprintf(file, "\n  Roads: Eastbound From: %d, Eastbound To: %d, Westbound From: %d, Westbound To: %d, Northbound From: %d, Northbound To: %d, Southbound From: %d, Southbound To: %d\n",
                intersection->road_eastbound_from_id,
                intersection->road_eastbound_to_id,
                intersection->road_westbound_from_id,
                intersection->road_westbound_to_id,
                intersection->road_northbound_from_id,
                intersection->road_northbound_to_id,
                intersection->road_southbound_from_id,
                intersection->road_southbound_to_id);
    }
    fprintf(file, "\n");
    fprintf(file, "Map Summary:\n");
    fprintf(file, "Total Roads: %d\n", self->num_roads);
    fprintf(file, "Total Intersections: %d\n", self->num_intersections);
}