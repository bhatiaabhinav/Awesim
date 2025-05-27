#include "utils.h"
#include "render.h"
#include "logging.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

// Render a solid center line for the lane based on type.
void render_lane_center_line(SDL_Renderer* renderer, const Lane* lane, const SDL_Color color) {
    int width = WINDOW_SIZE_WIDTH;
    int height = WINDOW_SIZE_HEIGHT;
    int thickness = (int)(LANE_CENTER_LINE_THICKNESS * SCALE);
    thickness = fclamp(thickness, 1, 1);

    if (lane->type == LINEAR_LANE) {
        SDL_Point start = to_screen_coords(lane->start_point, width, height);
        SDL_Point end = to_screen_coords(lane->end_point, width, height);
        thickLineRGBA_ignore_if_outside_screen(renderer, start.x, start.y, end.x, end.y, thickness, color.r, color.g, color.b, color.a);

    } else if (lane->type == QUARTER_ARC_LANE) {
        QuarterArcLane* arc_lane = (QuarterArcLane*)lane;
        SDL_Point center = to_screen_coords(lane->center, width, height);

        double step = (arc_lane->end_angle - arc_lane->start_angle) / ARC_NUM_POINTS;
        for (int i = 0; i < ARC_NUM_POINTS; i++) {
            double theta1 = arc_lane->start_angle + i * step;
            double theta2 = arc_lane->start_angle + (i + 1) * step;

            SDL_Point p1 = {
                (int)(center.x + arc_lane->radius * SCALE * cos(theta1)),
                (int)(center.y - arc_lane->radius * SCALE * sin(theta1))
            };
            SDL_Point p2 = {
                (int)(center.x + arc_lane->radius * SCALE * cos(theta2)),
                (int)(center.y - arc_lane->radius * SCALE * sin(theta2))
            };

            thickLineRGBA_ignore_if_outside_screen(renderer, p1.x, p1.y, p2.x, p2.y, thickness, color.r, color.g, color.b, color.a);
        }
    }
}

// Renders a rectangular lane section with optional lane lines and arrows.
void render_lane_linear(SDL_Renderer* renderer, const LinearLane* lane, const bool paint_lines, const bool paint_arrows, const bool paint_id) {
    int width = WINDOW_SIZE_WIDTH;
    int height = WINDOW_SIZE_HEIGHT;

    // Determine bounding box for rendering the lane.
    double x_min, x_max, y_min, y_max;
    if (lane->base.direction == DIRECTION_EAST || lane->base.direction == DIRECTION_WEST) {
        x_min = fmin(lane->base.start_point.x, lane->base.end_point.x);
        x_max = fmax(lane->base.start_point.x, lane->base.end_point.x);
        y_min = lane->base.center.y - lane->base.width / 2;
        y_max = lane->base.center.y + lane->base.width / 2;
    } else {
        y_min = fmin(lane->base.start_point.y, lane->base.end_point.y);
        y_max = fmax(lane->base.start_point.y, lane->base.end_point.y);
        x_min = lane->base.center.x - lane->base.width / 2;
        x_max = lane->base.center.x + lane->base.width / 2;
    }

    // Convert bounds to screen coordinates
    Coordinates txmin = {x_min, 0}, txmax = {x_max, 0};
    Coordinates tymin = {0, y_min}, tymax = {0, y_max};

    int screen_x_left = to_screen_coords(txmin, width, height).x;
    int screen_x_right = to_screen_coords(txmax, width, height).x;
    int screen_y_top = to_screen_coords(tymax, width, height).y;
    int screen_y_bottom = to_screen_coords(tymin, width, height).y;

    SDL_Rect lane_rect = {
        screen_x_left,
        screen_y_top,
        screen_x_right - screen_x_left,
        screen_y_bottom - screen_y_top
    };

    // Fill lane with gray
    SDL_SetRenderDrawColor(renderer, 128, 128, 128, 255);
    SDL_RenderFillRect_ignore_if_outside_screen(renderer, &lane_rect);

    // Determine left/right edge line positions
    Coordinates left_start, left_end, right_start, right_end;
    if (lane->base.direction == DIRECTION_EAST || lane->base.direction == DIRECTION_WEST) {
        double left_y = (lane->base.direction == DIRECTION_EAST) ? y_max : y_min;
        double right_y = (lane->base.direction == DIRECTION_EAST) ? y_min : y_max;
        left_start = (Coordinates){x_min, left_y};
        left_end = (Coordinates){x_max, left_y};
        right_start = (Coordinates){x_min, right_y};
        right_end = (Coordinates){x_max, right_y};
    } else {
        double left_x = (lane->base.direction == DIRECTION_NORTH) ? x_min : x_max;
        double right_x = (lane->base.direction == DIRECTION_NORTH) ? x_max : x_min;
        left_start = (Coordinates){left_x, y_min};
        left_end = (Coordinates){left_x, y_max};
        right_start = (Coordinates){right_x, y_min};
        right_end = (Coordinates){right_x, y_max};
    }

    SDL_Point lstart = to_screen_coords(left_start, width, height);
    SDL_Point lend = to_screen_coords(left_end, width, height);
    SDL_Point rstart = to_screen_coords(right_start, width, height);
    SDL_Point rend = to_screen_coords(right_end, width, height);

    int ythick = (int)(YELLOW_LINE_THICKNESS * SCALE);
    ythick = fclamp(ythick, 1, 1);
    int wthick = (int)(WHITE_LINE_THICKNESS * SCALE);
    wthick = fclamp(wthick, 1, 1);
    SDL_Color white = {255, 255, 255, 255};
    SDL_Color yellow = {255, 255, 0, 255};

    if (paint_lines) {
        // Left edge line
        if (lane->base.adjacent_left && lane->base.adjacent_left->direction != lane->base.direction) {
            thickLineRGBA_ignore_if_outside_screen(renderer, lstart.x, lstart.y, lend.x, lend.y, ythick, yellow.r, yellow.g, yellow.b, yellow.a);
        } else if (!lane->base.adjacent_left) {
            thickLineRGBA_ignore_if_outside_screen(renderer, lstart.x, lstart.y, lend.x, lend.y, wthick, white.r, white.g, white.b, white.a);
        } else {
            draw_dotted_line(renderer, lstart, lend, white);
        }

        // Right edge line
        if (!lane->base.adjacent_right) {
            thickLineRGBA_ignore_if_outside_screen(renderer, rstart.x, rstart.y, rend.x, rend.y, wthick, white.r, white.g, white.b, white.a);
        } else if (lane->base.adjacent_right->direction == lane->base.direction) {
            draw_dotted_line(renderer, rstart, rend, white);
        } else {
            fprintf(stderr, "Error: Right edge cannot be opposite direction!\n");
        }
    }

    if (paint_arrows) {
        // Calculate three arrow positions along the lane
        Coordinates mid = vec_div(vec_add(vec_scale(lane->base.start_point, 45), vec_scale(lane->base.end_point, 55)), 100);
        Coordinates a1 = vec_div(vec_add(vec_scale(lane->base.start_point, 4), lane->base.end_point), 5);
        Coordinates a2 = vec_div(vec_add(lane->base.start_point, vec_scale(lane->base.end_point, 3)), 4);

        SDL_Point arrow_locs[] = {
            to_screen_coords(mid, width, height),
            to_screen_coords(a1, width, height),
            to_screen_coords(a2, width, height)
        };

        int size = (int)(ARROW_SIZE * SCALE);
        SDL_Color color = {255, 255, 255, 255};

        for (int i = 0; i < 3; i++) {
            SDL_Point loc = arrow_locs[i];
            SDL_Point pts[3];

            switch (lane->base.direction) {
                case DIRECTION_EAST:
                    pts[0] = (SDL_Point){loc.x + size / 2, loc.y};
                    pts[1] = (SDL_Point){loc.x - size / 2, loc.y - size / 2};
                    pts[2] = (SDL_Point){loc.x - size / 2, loc.y + size / 2};
                    break;
                case DIRECTION_WEST:
                    pts[0] = (SDL_Point){loc.x - size / 2, loc.y};
                    pts[1] = (SDL_Point){loc.x + size / 2, loc.y - size / 2};
                    pts[2] = (SDL_Point){loc.x + size / 2, loc.y + size / 2};
                    break;
                case DIRECTION_NORTH:
                    pts[0] = (SDL_Point){loc.x, loc.y - size / 2};
                    pts[1] = (SDL_Point){loc.x - size / 2, loc.y + size / 2};
                    pts[2] = (SDL_Point){loc.x + size / 2, loc.y + size / 2};
                    break;
                case DIRECTION_SOUTH:
                default:
                    pts[0] = (SDL_Point){loc.x, loc.y + size / 2};
                    pts[1] = (SDL_Point){loc.x - size / 2, loc.y - size / 2};
                    pts[2] = (SDL_Point){loc.x + size / 2, loc.y - size / 2};
                    break;
            }

            Sint16 vx[3] = {pts[0].x, pts[1].x, pts[2].x};
            Sint16 vy[3] = {pts[0].y, pts[1].y, pts[2].y};
            filledTrigonRGBA_ignore_if_outside_screen(renderer, vx[0], vy[0], vx[1], vy[1], vx[2], vy[2], color.r, color.g, color.b, color.a);
            trigonRGBA_ignore_if_outside_screen(renderer, vx[0], vy[0], vx[1], vy[1], vx[2], vy[2], color.r, color.g, color.b, 255);
        }
    }

    // Render lane id:
    if (paint_id) {
        char id_str[10];
        snprintf(id_str, sizeof(id_str), "%d", lane->base.id);
        int font_size = (int)(meters(2.0) * SCALE); // font size = 2 meter
        SDL_Color text_color = {255, 255, 255, 255}; // white text color
        SDL_Point lane_center_screen = to_screen_coords(lane->base.center, width, height);
        int text_x = lane_center_screen.x;
        int text_y = lane_center_screen.y;
        render_text(renderer, id_str, text_x, text_y, text_color.r, text_color.g, text_color.b, text_color.a, font_size, ALIGN_CENTER);
    }
}


// Renders a quarter-arc lane, optionally painting boundary lines and directional arrows.
void render_lane_quarterarc(SDL_Renderer* renderer, const QuarterArcLane* lane, const bool paint_lines, const bool paint_arrows, const bool paint_id) {
    int width = WINDOW_SIZE_WIDTH;
    int height = WINDOW_SIZE_HEIGHT;

    SDL_Point center_screen = to_screen_coords(lane->base.center, width, height);

    // Define inner and outer radii of the arc-shaped lane
    double r_inner = lane->radius - lane->base.width / 2;
    double r_outer = lane->radius + lane->base.width / 2;

    // Prepare vertex arrays to define the polygon shape of the lane
    Sint16 vx[2 * (ARC_NUM_POINTS + 1)];
    Sint16 vy[2 * (ARC_NUM_POINTS + 1)];
    double angle_step = (lane->end_angle - lane->start_angle) / ARC_NUM_POINTS;
    int idx = 0;

    // Fill inner arc points (counter-clockwise)
    for (int i = 0; i <= ARC_NUM_POINTS; i++) {
        double angle = lane->start_angle + i * angle_step;
        Coordinates pt = {
            lane->base.center.x + r_inner * cos(angle),
            lane->base.center.y + r_inner * sin(angle)
        };
        SDL_Point screen_pt = to_screen_coords(pt, width, height);
        vx[idx] = screen_pt.x;
        vy[idx] = screen_pt.y;
        idx++;
    }

    // Fill outer arc points (clockwise)
    for (int i = ARC_NUM_POINTS; i >= 0; i--) {
        double angle = lane->start_angle + i * angle_step;
        Coordinates pt = {
            lane->base.center.x + r_outer * cos(angle),
            lane->base.center.y + r_outer * sin(angle)
        };
        SDL_Point screen_pt = to_screen_coords(pt, width, height);
        vx[idx] = screen_pt.x;
        vy[idx] = screen_pt.y;
        idx++;
    }

    // Draw the lane surface (gray)
    filledPolygonRGBA_ignore_if_outside_screen(renderer, vx, vy, 2 * (ARC_NUM_POINTS + 1), 128, 128, 128, 255);

    if (paint_lines) {
        // Determine appropriate color and thickness for the left boundary
        SDL_Color left_line_color = {255, 255, 255, 255};
        int left_thickness = (int)(WHITE_LINE_THICKNESS * SCALE);
        left_thickness = fclamp(left_thickness, 1, 1);

        if (lane->base.adjacent_left && lane->base.adjacent_left->direction != lane->base.direction) {
            left_line_color = (SDL_Color){255, 255, 0, 255}; // Yellow for opposing direction
            left_thickness = (int)(YELLOW_LINE_THICKNESS * SCALE);
            left_thickness = fclamp(left_thickness, 1, 1);
        } else if (lane->base.adjacent_left) {
            left_thickness = (int)(DOTTED_LINE_THICKNESS * SCALE);
            left_thickness = fclamp(left_thickness, 1, 1);
        }

        // Fixed white line for right side (no dotted support for arcs)
        SDL_Color right_line_color = {255, 255, 255, 255};
        int right_thickness = (int)(WHITE_LINE_THICKNESS * SCALE);
        right_thickness = fclamp(right_thickness, 1, 1);

        // Assign correct radius based on turning direction
        double r_left = (lane->base.direction == DIRECTION_CCW) ? r_inner : r_outer;
        double r_right = (lane->base.direction == DIRECTION_CCW) ? r_outer : r_inner;

        // Draw arc segments for left and right lines
        for (int i = 0; i < ARC_NUM_POINTS; i++) {
            double theta1 = lane->start_angle + i * angle_step;
            double theta2 = lane->start_angle + (i + 1) * angle_step;

            // Left boundary line
            SDL_Point p1_left = {
                (int)(center_screen.x + r_left * SCALE * cos(theta1)),
                (int)(center_screen.y - r_left * SCALE * sin(theta1))
            };
            SDL_Point p2_left = {
                (int)(center_screen.x + r_left * SCALE * cos(theta2)),
                (int)(center_screen.y - r_left * SCALE * sin(theta2))
            };
            thickLineRGBA_ignore_if_outside_screen(renderer, p1_left.x, p1_left.y, p2_left.x, p2_left.y,
                          left_thickness, left_line_color.r, left_line_color.g, left_line_color.b, left_line_color.a);

            // Right boundary line
            SDL_Point p1_right = {
                (int)(center_screen.x + r_right * SCALE * cos(theta1)),
                (int)(center_screen.y - r_right * SCALE * sin(theta1))
            };
            SDL_Point p2_right = {
                (int)(center_screen.x + r_right * SCALE * cos(theta2)),
                (int)(center_screen.y - r_right * SCALE * sin(theta2))
            };
            thickLineRGBA_ignore_if_outside_screen(renderer, p1_right.x, p1_right.y, p2_right.x, p2_right.y,
                          right_thickness, right_line_color.r, right_line_color.g, right_line_color.b, right_line_color.a);
        }
    }

    if (paint_arrows) {
        // Render a single directional arrow at the center of the arc
        int arrow_size = (int)(ARROW_SIZE * SCALE);
        SDL_Color arrow_color = {255, 255, 255, 255};

        double middle_radius = (r_inner + r_outer) / 2;
        double angle = lane->start_angle + (55 * lane->end_angle - 45 * lane->start_angle) / 100.0; // Middle angle of the arc plus slight offset

        // Arrow center on the arc path
        SDL_Point arrow_center = {
            center_screen.x + (int)(middle_radius * SCALE * cos(angle)),
            center_screen.y - (int)(middle_radius * SCALE * sin(angle))
        };

        // Tangent direction of arrow (along motion)
        double tangent_angle = angle + M_PI_2; // Perpendicular to radius
        if (lane->base.direction == DIRECTION_CW)
            tangent_angle += M_PI; // Flip for clockwise lanes

        // Construct arrow triangle points
        SDL_Point arrow[3];
        arrow[0] = (SDL_Point){
            arrow_center.x + (int)(arrow_size / 2 * cos(tangent_angle)),
            arrow_center.y - (int)(arrow_size / 2 * sin(tangent_angle))
        };
        double base_angle1 = tangent_angle + M_PI - M_PI / 4;
        double base_angle2 = tangent_angle + M_PI + M_PI / 4;

        arrow[1] = (SDL_Point){
            arrow_center.x + (int)(arrow_size / 2 * cos(base_angle1)),
            arrow_center.y - (int)(arrow_size / 2 * sin(base_angle1))
        };
        arrow[2] = (SDL_Point){
            arrow_center.x + (int)(arrow_size / 2 * cos(base_angle2)),
            arrow_center.y - (int)(arrow_size / 2 * sin(base_angle2))
        };

        // Draw filled and outlined arrow triangle
        Sint16 vx[3] = {arrow[0].x, arrow[1].x, arrow[2].x};
        Sint16 vy[3] = {arrow[0].y, arrow[1].y, arrow[2].y};

        filledTrigonRGBA_ignore_if_outside_screen(renderer, vx[0], vy[0], vx[1], vy[1], vx[2], vy[2],
                         arrow_color.r, arrow_color.g, arrow_color.b, arrow_color.a);
        trigonRGBA_ignore_if_outside_screen(renderer, vx[0], vy[0], vx[1], vy[1], vx[2], vy[2],
                   arrow_color.r, arrow_color.g, arrow_color.b, 255);
    }

    // Render lane id:
    if (paint_id) {
        // Render lane ID at the center of the arc
        char id_str[10];
        snprintf(id_str, sizeof(id_str), "%d", lane->base.id);
        int font_size = (int)(meters(2.0) * SCALE); // font size = 2 meter
        SDL_Color text_color = {255, 255, 255, 255}; // white text color
        double middle_radius = (r_inner + r_outer) / 2;
        double middle_angle = lane->start_angle + (lane->end_angle - lane->start_angle) / 2.0; // Middle angle of the arc
        int text_x = center_screen.x + (int)(middle_radius * SCALE * cos(middle_angle));
        int text_y = center_screen.y - (int)(middle_radius * SCALE * sin(middle_angle));
        render_text(renderer, id_str, text_x, text_y, text_color.r, text_color.g, text_color.b, text_color.a, font_size, ALIGN_CENTER);
    }
}


void render_lane(SDL_Renderer* renderer, const Lane* lane, const bool paint_lines, const bool paint_arrows, const bool paint_id) {
    if (lane->type == LINEAR_LANE) {
        render_lane_linear(renderer, (LinearLane*)lane, paint_lines, paint_arrows, paint_id);
    } else if (lane->type == QUARTER_ARC_LANE) {
        render_lane_quarterarc(renderer, (QuarterArcLane*)lane, paint_lines, paint_arrows, paint_id);
    } else {
        LOG_ERROR("Error: Unknown lane type!");
    }
}
