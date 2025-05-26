#include "sim.h"
#include "logging.h"
#include <stdlib.h>



TrackIntersectionPoint track_intersection_check(const Lane* lane1, const Lane* lane2) {
    // bool intersect = false;
    // if (lane1->type == LINEAR_LANE && lane2->type == LINEAR_LANE) {
    //     // Check if the lanes are parallel and at the same position
    //     if (lane1->direction == lane2->direction)  {
    //         return (TrackIntersectionPoint){lane1, lane2, 0.0, 0.0, false}; // No intersection
    //     } else {
    //         // Calculate intersection point for linear lanes
    //         double start1 = lane1->start_point.x;
    //         double end1 = lane1->end_point.x;
    //         double start2 = lane2->start_point.x;
    //         double end2 = lane2->end_point.x;

    //         if ((start1 <= end2 && end1 >= start2) || (start2 <= end1 && end2 >= start1)) {
    //             intersect = true;
    //         }
    //     }
    // }
    return (TrackIntersectionPoint){NULL, NULL, 0, 0, false};
}






Hazards* hazards_create() {
    Hazards* points = malloc(sizeof(Hazards));
    if (!points) {
        LOG_ERROR("Failed to allocate memory for HazardPoints");
        exit(EXIT_FAILURE); // Handle memory allocation failure
        return NULL; // Memory allocation failed
    }
    points->num_intersection_points = 0;
    points->num_dead_ends = 0;
    return points;
}

void hazards_free(Hazards* self) {
    if (self) {
        free(self);
    }
}

void hazards_add_intersection_point(Hazards* self, const TrackIntersectionPoint* point) {
    if (self->num_intersection_points < MAX_NUM_HAZARDS_EACH_TYPE) {
        self->points[self->num_intersection_points++] = point;
    } else {
        LOG_ERROR("Maximum number of intersection points reached");
    }
}

void hazards_add_dead_end(Hazards* self, const DeadEnd* dead_end) {
    if (self->num_dead_ends < MAX_NUM_HAZARDS_EACH_TYPE) {
        self->dead_ends[self->num_dead_ends++] = dead_end;
    } else {
        LOG_ERROR("Maximum number of dead ends reached");
    }
}