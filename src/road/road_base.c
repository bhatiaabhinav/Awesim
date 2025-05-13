#include "road.h"
#include <stdlib.h>

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

const Lane* road_leftmost_lane(const Road* self) {
    if (self->num_lanes == 0) {
        return NULL;
    }
    return self->lanes[0];
}

const Lane* road_rightmost_lane(const Road* self) {
    if (self->num_lanes == 0) {
        return NULL;
    }
    return self->lanes[self->num_lanes - 1];
}

//
// Road Getters
//

RoadType road_get_type(const Road* self) {
    return self->type;
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
