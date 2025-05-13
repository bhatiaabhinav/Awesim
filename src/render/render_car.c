#include "utils.h"
#include "render.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

typedef struct {
    Coordinates position;
    double angle;
} CarPose;

CarPose get_car_pose(const Car* car) {
    const Lane* lane = car->lane;
    CarPose pose;

    if (lane->type == LINEAR_LANE) {
        Vec2D start = lane->start_point;
        Vec2D end = lane->end_point;
        Vec2D delta = vec_sub(end, start);
        pose.position = vec_add(start, vec_scale(delta, car->lane_progress));
        pose.angle = atan2(delta.y, delta.x);
    } else if (lane->type == QUARTER_ARC_LANE) {
        QuarterArcLane* arc_lane = (QuarterArcLane*)lane;
        double theta = arc_lane->start_angle + car->lane_progress * (arc_lane->end_angle - arc_lane->start_angle);
        pose.position.x = lane->center.x + arc_lane->radius * cos(theta);
        pose.position.y = lane->center.y + arc_lane->radius * sin(theta);
        if (lane->direction == DIRECTION_CCW) {
            pose.angle = theta + M_PI / 2;
        } else if (lane->direction == DIRECTION_CW) {
            pose.angle = theta - M_PI / 2;
        } else {
            fprintf(stderr, "Error: Invalid direction for arc lane!\n");
            pose.angle = 0;
        }
    } else {
        fprintf(stderr, "Error: Unknown lane type!\n");
        pose.position = coordinates_create(0, 0);
        pose.angle = 0;
    }
    return pose;
}

void render_car(SDL_Renderer* renderer, const Car* car) {
    CarPose pose = get_car_pose(car);
    double width = car->dimensions.x;
    double length = car->dimensions.y;

    Vec2D local_corners[4] = {
        {length / 2, width / 2},
        {length / 2, -width / 2},
        {-length / 2, -width / 2},
        {-length / 2, width / 2}
    };

    Coordinates world_corners[4];
    double cos_a = cos(pose.angle);
    double sin_a = sin(pose.angle);
    for (int i = 0; i < 4; i++) {
        double x_local = local_corners[i].x;
        double y_local = local_corners[i].y;
        double x_rot = x_local * cos_a - y_local * sin_a;
        double y_rot = x_local * sin_a + y_local * cos_a;
        world_corners[i].x = pose.position.x + x_rot;
        world_corners[i].y = pose.position.y + y_rot;
    }

    int screen_width = WINDOW_SIZE_WIDTH;
    int screen_height = WINDOW_SIZE_HEIGHT;
    SDL_Point screen_corners[4];
    for (int i = 0; i < 4; i++) {
        screen_corners[i] = to_screen_coords(world_corners[i], screen_width, screen_height);
    }

    Sint16 vx[4], vy[4];
    for (int i = 0; i < 4; i++) {
        vx[i] = (Sint16)screen_corners[i].x;
        vy[i] = (Sint16)screen_corners[i].y;
    }
    filledPolygonRGBA(renderer, vx, vy, 4, CAR_COLOR.r, CAR_COLOR.g, CAR_COLOR.b, CAR_COLOR.a);
    polygonRGBA(renderer, vx, vy, 4, CAR_COLOR.r, CAR_COLOR.g, CAR_COLOR.b, 255);
}