#include "utils.h"
#include "render.h"
#include "logging.h"
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
            LOG_ERROR("Error: Invalid direction for arc lane!");
            pose.angle = 0;
        }
    } else {
        fprintf(stderr, "Error: Unknown lane type!\n");
        pose.position = coordinates_create(0, 0);
        pose.angle = 0;
    }
    return pose;
}


void render_car(SDL_Renderer* renderer, const Car* car, const bool paint_id) {
    CarPose pose = get_car_pose(car);
    double width = car->dimensions.x;
    double length = car->dimensions.y;

    // Define car corners in local coordinates (front at +length/2, rear at -length/2)
    Vec2D local_corners[4] = {
        {length / 2, width / 2},   // Front-right (0)
        {length / 2, -width / 2},  // Front-left (1)
        {-length / 2, -width / 2}, // Rear-left (2)
        {-length / 2, width / 2}   // Rear-right (3)
    };

    // Transform corners to world coordinates
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

    // Convert to screen coordinates
    int screen_width = WINDOW_SIZE_WIDTH;
    int screen_height = WINDOW_SIZE_HEIGHT;
    SDL_Point screen_corners[4];
    for (int i = 0; i < 4; i++) {
        screen_corners[i] = to_screen_coords(world_corners[i], screen_width, screen_height);
    }

    // Draw car body
    Sint16 vx[4], vy[4];
    for (int i = 0; i < 4; i++) {
        vx[i] = (Sint16)screen_corners[i].x;
        vy[i] = (Sint16)screen_corners[i].y;
    }
    filledPolygonRGBA_ignore_if_outside_screen(renderer, vx, vy, 4, CAR_COLOR.r, CAR_COLOR.g, CAR_COLOR.b, CAR_COLOR.a);
    polygonRGBA_ignore_if_outside_screen(renderer, vx, vy, 4, CAR_COLOR.r, CAR_COLOR.g, CAR_COLOR.b, 255);

    int light_thickness = (int)(from_inches(6.0) * SCALE);
    light_thickness = fmax(light_thickness, 1); // Ensure light thickness is at least 1

    // Draw headlights (always on)
    {
        double headlight_length = width / 4 * 1.5; // 1.5x indicator length

        // Left headlight (front edge, left corner)
        Vec2D local_left_headlight_start = {length / 2, -width / 2};
        Vec2D local_left_headlight_end = {length / 2, -width / 2 + headlight_length};
        // Right headlight (front edge, right corner)
        Vec2D local_right_headlight_start = {length / 2, width / 2 - headlight_length};
        Vec2D local_right_headlight_end = {length / 2, width / 2};

        // Transform to world coordinates
        Coordinates world_left_headlight_start, world_left_headlight_end;
        Coordinates world_right_headlight_start, world_right_headlight_end;

        double x_local = local_left_headlight_start.x;
        double y_local = local_left_headlight_start.y;
        world_left_headlight_start.x = pose.position.x + x_local * cos_a - y_local * sin_a;
        world_left_headlight_start.y = pose.position.y + x_local * sin_a + y_local * cos_a;

        x_local = local_left_headlight_end.x;
        y_local = local_left_headlight_end.y;
        world_left_headlight_end.x = pose.position.x + x_local * cos_a - y_local * sin_a;
        world_left_headlight_end.y = pose.position.y + x_local * sin_a + y_local * cos_a;

        x_local = local_right_headlight_start.x;
        y_local = local_right_headlight_start.y;
        world_right_headlight_start.x = pose.position.x + x_local * cos_a - y_local * sin_a;
        world_right_headlight_start.y = pose.position.y + x_local * sin_a + y_local * cos_a;

        x_local = local_right_headlight_end.x;
        y_local = local_right_headlight_end.y;
        world_right_headlight_end.x = pose.position.x + x_local * cos_a - y_local * sin_a;
        world_right_headlight_end.y = pose.position.y + x_local * sin_a + y_local * cos_a;

        // Convert to screen coordinates
        SDL_Point screen_left_headlight_start = to_screen_coords(world_left_headlight_start, screen_width, screen_height);
        SDL_Point screen_left_headlight_end = to_screen_coords(world_left_headlight_end, screen_width, screen_height);
        SDL_Point screen_right_headlight_start = to_screen_coords(world_right_headlight_start, screen_width, screen_height);
        SDL_Point screen_right_headlight_end = to_screen_coords(world_right_headlight_end, screen_width, screen_height);

        // Draw headlights (yellowish, 5px thick)
        thickLineRGBA_ignore_if_outside_screen(renderer, screen_left_headlight_start.x, screen_left_headlight_start.y,
                                              screen_left_headlight_end.x, screen_left_headlight_end.y, light_thickness, 255, 255, 100, 255);
        thickLineRGBA_ignore_if_outside_screen(renderer, screen_right_headlight_start.x, screen_right_headlight_start.y,
                                              screen_right_headlight_end.x, screen_right_headlight_end.y, light_thickness, 255, 255, 100, 255);
    }

    // Draw taillights (always on, unless braking)
    bool is_braking = false;
    if (car->speed > 0 && car->acceleration < -0.1) {   // a little bit of tolerance since brake lights don't turn on with just a touch of brake
        is_braking = true; // Braking
    } else if (car->speed < 0) {
        is_braking = true; // reversing
    }

    {
        double taillight_length = width / 4; // Same as indicator length

        // Left taillight
        Vec2D local_left_taillight_start = {-length / 2, -width / 2};
        Vec2D local_left_taillight_end = {-length / 2, -width / 2 + taillight_length};
        // Right taillight
        Vec2D local_right_taillight_start = {-length / 2, width / 2 - taillight_length};
        Vec2D local_right_taillight_end = {-length / 2, width / 2};

        // Transform to world coordinates
        Coordinates world_left_taillight_start, world_left_taillight_end;
        Coordinates world_right_taillight_start, world_right_taillight_end;

        double x_local = local_left_taillight_start.x;
        double y_local = local_left_taillight_start.y;
        world_left_taillight_start.x = pose.position.x + x_local * cos_a - y_local * sin_a;
        world_left_taillight_start.y = pose.position.y + x_local * sin_a + y_local * cos_a;

        x_local = local_left_taillight_end.x;
        y_local = local_left_taillight_end.y;
        world_left_taillight_end.x = pose.position.x + x_local * cos_a - y_local * sin_a;
        world_left_taillight_end.y = pose.position.y + x_local * sin_a + y_local * cos_a;

        x_local = local_right_taillight_start.x;
        y_local = local_right_taillight_start.y;
        world_right_taillight_start.x = pose.position.x + x_local * cos_a - y_local * sin_a;
        world_right_taillight_start.y = pose.position.y + x_local * sin_a + y_local * cos_a;

        x_local = local_right_taillight_end.x;
        y_local = local_right_taillight_end.y;
        world_right_taillight_end.x = pose.position.x + x_local * cos_a - y_local * sin_a;
        world_right_taillight_end.y = pose.position.y + x_local * sin_a + y_local * cos_a;

        // Convert to screen coordinates
        SDL_Point screen_left_taillight_start = to_screen_coords(world_left_taillight_start, screen_width, screen_height);
        SDL_Point screen_left_taillight_end = to_screen_coords(world_left_taillight_end, screen_width, screen_height);
        SDL_Point screen_right_taillight_start = to_screen_coords(world_right_taillight_start, screen_width, screen_height);
        SDL_Point screen_right_taillight_end = to_screen_coords(world_right_taillight_end, screen_width, screen_height);

        // Draw taillights (faint red normally, full red when braking)
        Uint8 taillight_r = is_braking ? 255 : 100;
        Uint8 taillight_g = 0;
        Uint8 taillight_b = 0;
        Uint8 taillight_a = is_braking ? 255 : 150;
        thickLineRGBA_ignore_if_outside_screen(renderer, screen_left_taillight_start.x, screen_left_taillight_start.y,
                                              screen_left_taillight_end.x, screen_left_taillight_end.y, light_thickness, taillight_r, taillight_g, taillight_b, taillight_a);
        thickLineRGBA_ignore_if_outside_screen(renderer, screen_right_taillight_start.x, screen_right_taillight_start.y,
                                              screen_right_taillight_end.x, screen_right_taillight_end.y, light_thickness, taillight_r, taillight_g, taillight_b, taillight_a);
    }

    // Draw turn indicators (only if not braking)
    if (car->indicator_turn != INDICATOR_NONE) {
        // Blinking logic: 500ms on, 500ms off (1000ms period)
        Uint32 current_time = SDL_GetTicks();
        const Uint32 blink_period = 1000; // 1 second
        const Uint32 blink_on_duration = 500; // 0.5 seconds
        if ((current_time % blink_period) < blink_on_duration) {
            double indicator_length = width / 4; // Length of the indicator dash

            // Define local points for indicators
            Vec2D local_front_side_start, local_front_side_end; // Side indicator
            Vec2D local_back_start, local_back_end;             // Back indicator
            Vec2D local_front_edge_start, local_front_edge_end; // Original front edge indicator

            if (car->indicator_turn == INDICATOR_RIGHT) { // Left side indicators
                // Front side indicator: left edge, 25% from front
                double x_pos = length / 2 - 0.25 * length; // 25% from front
                local_front_side_start = (Vec2D){x_pos - indicator_length / 2, -width / 2};
                local_front_side_end = (Vec2D){x_pos + indicator_length / 2, -width / 2};
                // Back indicator: left side of back edge
                local_back_start = (Vec2D){-length / 2, -width / 2};
                local_back_end = (Vec2D){-length / 2, -width / 2 + indicator_length};
                // Original front edge indicator: left side of front edge
                local_front_edge_start = (Vec2D){length / 2, -width / 2};
                local_front_edge_end = (Vec2D){length / 2, -width / 2 + indicator_length};
            } else { // INDICATOR_LEFT: Right side indicators
                // Front side indicator: right edge, 25% from front
                double x_pos = length / 2 - 0.25 * length; // 25% from front
                local_front_side_start = (Vec2D){x_pos - indicator_length / 2, width / 2};
                local_front_side_end = (Vec2D){x_pos + indicator_length / 2, width / 2};
                // Back indicator: right side of back edge
                local_back_start = (Vec2D){-length / 2, width / 2 - indicator_length};
                local_back_end = (Vec2D){-length / 2, width / 2};
                // Original front edge indicator: right side of front edge
                local_front_edge_start = (Vec2D){length / 2, width / 2 - indicator_length};
                local_front_edge_end = (Vec2D){length / 2, width / 2};
            }

            // Transform to world coordinates
            Coordinates world_front_side_start, world_front_side_end;
            Coordinates world_back_start, world_back_end;
            Coordinates world_front_edge_start, world_front_edge_end;

            // Front side start
            double x_local = local_front_side_start.x;
            double y_local = local_front_side_start.y;
            world_front_side_start.x = pose.position.x + x_local * cos_a - y_local * sin_a;
            world_front_side_start.y = pose.position.y + x_local * sin_a + y_local * cos_a;

            // Front side end
            x_local = local_front_side_end.x;
            y_local = local_front_side_end.y;
            world_front_side_end.x = pose.position.x + x_local * cos_a - y_local * sin_a;
            world_front_side_end.y = pose.position.y + x_local * sin_a + y_local * cos_a;

            // Back start
            x_local = local_back_start.x;
            y_local = local_back_start.y;
            world_back_start.x = pose.position.x + x_local * cos_a - y_local * sin_a;
            world_back_start.y = pose.position.y + x_local * sin_a + y_local * cos_a;

            // Back end
            x_local = local_back_end.x;
            y_local = local_back_end.y;
            world_back_end.x = pose.position.x + x_local * cos_a - y_local * sin_a;
            world_back_end.y = pose.position.y + x_local * sin_a + y_local * cos_a;

            // Original front edge start
            x_local = local_front_edge_start.x;
            y_local = local_front_edge_start.y;
            world_front_edge_start.x = pose.position.x + x_local * cos_a - y_local * sin_a;
            world_front_edge_start.y = pose.position.y + x_local * sin_a + y_local * cos_a;

            // Original front edge end
            x_local = local_front_edge_end.x;
            y_local = local_front_edge_end.y;
            world_front_edge_end.x = pose.position.x + x_local * cos_a - y_local * sin_a;
            world_front_edge_end.y = pose.position.y + x_local * sin_a + y_local * cos_a;

            // Convert to screen coordinates
            SDL_Point screen_front_side_start = to_screen_coords(world_front_side_start, screen_width, screen_height);
            SDL_Point screen_front_side_end = to_screen_coords(world_front_side_end, screen_width, screen_height);
            SDL_Point screen_back_start = to_screen_coords(world_back_start, screen_width, screen_height);
            SDL_Point screen_back_end = to_screen_coords(world_back_end, screen_width, screen_height);
            SDL_Point screen_front_edge_start = to_screen_coords(world_front_edge_start, screen_width, screen_height);
            SDL_Point screen_front_edge_end = to_screen_coords(world_front_edge_end, screen_width, screen_height);

            // Draw thick lines for indicators
            // FRONT_SIDE_INDICATOR
            thickLineRGBA_ignore_if_outside_screen(renderer, screen_front_side_start.x, screen_front_side_start.y, 
                                                  screen_front_side_end.x, screen_front_side_end.y, light_thickness, 255, 0, 0, 255);
            // BACK_INDICATOR
            thickLineRGBA_ignore_if_outside_screen(renderer, screen_back_start.x, screen_back_start.y, 
                                                  screen_back_end.x, screen_back_end.y, light_thickness, 255, 0, 0, 255);
            // FRONT_EDGE_INDICATOR (comment out if not needed)
            thickLineRGBA_ignore_if_outside_screen(renderer, screen_front_edge_start.x, screen_front_edge_start.y, 
                                                  screen_front_edge_end.x, screen_front_edge_end.y, light_thickness, 255, 0, 0, 255);
        }
    }

    // Render car ID
    if (paint_id) {
        char id_str[10];
        snprintf(id_str, sizeof(id_str), "%d", car->id);
        int font_size = (int)(meters(1.0) * SCALE);     // font size = 1 meter
        SDL_Point car_center_screen = to_screen_coords(pose.position, screen_width, screen_height);
        int text_x = car_center_screen.x;
        int text_y = car_center_screen.y;
        render_text(renderer, id_str, text_x, text_y, 255, 255, 255, 255, font_size, ALIGN_CENTER, false);
    }
}