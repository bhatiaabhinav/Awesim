#include "road.h"
#include <stdlib.h>
#include <stdio.h>

int road_id_counter = 0; // Global ID counter for roads

void road_free(Road* self) {
    switch (self->type) {
        case STRAIGHT:
            straight_road_free((StraightRoad*)self);
            break;
        case TURN:
            turn_free((Turn*)self);
            break;
        case INTERSECTION:
            intersection_free((Intersection*)self);
            break;
    }
}

const Lane* road_get_leftmost_lane(const Road* self) {
    if (self->num_lanes == 0) {
        return NULL;
    }
    return self->lanes[0];
}

const Lane* road_get_rightmost_lane(const Road* self) {
    if (self->num_lanes == 0) {
        return NULL;
    }
    return self->lanes[self->num_lanes - 1];
}

bool road_is_merge_available(const Road* self) {
    return self->type != INTERSECTION && lane_is_merge_available(road_get_leftmost_lane(self));
}

bool road_is_exit_road_available(const Road* self, double progress) {
    return self->type != INTERSECTION && lane_is_exit_lane_available(road_get_rightmost_lane(self), progress);
}

bool road_is_exit_road_eventually_available(const Road* self, double progress) {
    return self->type != INTERSECTION && lane_is_exit_lane_eventually_available(road_get_rightmost_lane(self), progress);
}

//
// Road Getters
//

int road_get_id(const Road* self) {
    return self->id;
}

RoadType road_get_type(const Road* self) {
    return self->type;
}

const char* road_get_name(const Road* self) {
    return self->name;
}

int road_get_num_lanes(const Road* self) {
    return self->num_lanes;
}

const Lane* road_get_lane(const Road* self, int index) {
    if (index < 0 || index >= self->num_lanes) {
        return NULL;
    }
    return self->lanes[index];
}

MetersPerSecond road_get_speed_limit(const Road* self) {
    return self->speed_limit;
}

double road_get_grip(const Road* self) {
    return self->grip;
}

Coordinates road_get_center(const Road* self) {
    return self->center;
}


//
// Road Setters
//

void road_set_name(Road* self, const char* name) {
    if (name) {
        snprintf(self->name, sizeof(self->name), "%s", name);
        // Name each lane with the road's name and lane index
        for (int i = 0; i < self->num_lanes; i++) {
            Lane* lane = (Lane*)self->lanes[i];
            snprintf(lane->name, sizeof(lane->name), "%s Lane %d", self->name, i);
        }
    } else {
        self->name[0] = '\0'; // Clear the name if NULL
    }
}




const Intersection* road_leads_to_intersection(const Road* road) {
    if (road->type != STRAIGHT && road->type != TURN) return NULL;

    const Lane* lane = road->lanes[0];
    const Lane* options[] = {
        lane->for_left_connects_to,
        lane->for_right_connects_to,
        lane->for_straight_connects_to
    };

    for (int i = 0; i < 3; i++) {
        if (options[i] && options[i]->road->type == INTERSECTION) {
            return (const Intersection*)options[i]->road;
        }
    }
    return NULL;
}
