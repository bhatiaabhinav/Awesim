#include "utils.h"
#include "render.h"
#include "logging.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

SDL_Texture* lane_id_texture_cache[MAX_NUM_LANES][MAX_FONT_SIZE] = {{NULL}};

static bool should_highlight_lane(const Lane* lane) {
    // Check if the lane ID is in the highlighted lanes array
    for (int i = 0; HIGHLIGHTED_LANES[i] != ID_NULL; i++) {
        if (HIGHLIGHTED_LANES[i] == lane->id) {
            return true;
        }
    }
    return false;
}

// Render a solid center line for the lane based on type.
void render_lane_center_line(SDL_Renderer* renderer, const Lane* lane, Map* map, const SDL_Color color, bool dotted) {
    int width = WINDOW_SIZE_WIDTH;
    int height = WINDOW_SIZE_HEIGHT;
    int thickness = (int)(LANE_CENTER_LINE_THICKNESS * SCALE);
    thickness = fclamp(thickness, 1, 1);
    if (should_highlight_lane(lane)) {
        thickness += 1; // Increase thickness for highlighted lanes
    }

    if (lane->type == LINEAR_LANE) {
        SDL_Point start = to_screen_coords(lane->start_point, width, height);
        SDL_Point end = to_screen_coords(lane->end_point, width, height);
        thickLineRGBA_ignore_if_outside_screen(renderer, start.x, start.y, end.x, end.y, thickness, color.r, color.g, color.b, color.a);

    } else if (lane->type == QUARTER_ARC_LANE) {
        // Calculate offsets for smooth interpolation (fix for asymmetric intersections)
        Vec2D theoretical_start = {
            lane->center.x + lane->radius * cos(lane->start_angle),
            lane->center.y + lane->radius * sin(lane->start_angle)
        };
        Vec2D theoretical_end = {
            lane->center.x + lane->radius * cos(lane->end_angle),
            lane->center.y + lane->radius * sin(lane->end_angle)
        };
        
        Lane* incoming = lane_get_connection_incoming_straight(lane, map);
        Lane* outgoing = lane_get_connection_straight(lane, map);
        
        Vec2D actual_start = incoming ? incoming->end_point : lane->start_point;
        Vec2D actual_end = outgoing ? outgoing->start_point : lane->end_point;
        
        Vec2D offset_start = vec_sub(actual_start, theoretical_start);
        Vec2D offset_end = vec_sub(actual_end, theoretical_end);

        int arc_num_points = dotted ? ARC_NUM_POINTS * 2 : ARC_NUM_POINTS;  // Double points for dotted lines since we skip every second point

        double step = (lane->end_angle - lane->start_angle) / arc_num_points;
        for (int i = 0; i < arc_num_points; i++) {
            if (dotted && i % 2 == 0) continue; // Skip every second point for dotted lines
            
            double p1_ratio = (double)i / arc_num_points;
            double p2_ratio = (double)(i + 1) / arc_num_points;

            double theta1 = lane->start_angle + i * step;
            double theta2 = lane->start_angle + (i + 1) * step;

            Vec2D off1 = vec_add(vec_scale(offset_start, 1.0 - p1_ratio), vec_scale(offset_end, p1_ratio));
            Vec2D off2 = vec_add(vec_scale(offset_start, 1.0 - p2_ratio), vec_scale(offset_end, p2_ratio));

            Coordinates w1 = {
                lane->center.x + lane->radius * cos(theta1) + off1.x,
                lane->center.y + lane->radius * sin(theta1) + off1.y
            };
            Coordinates w2 = {
                lane->center.x + lane->radius * cos(theta2) + off2.x,
                lane->center.y + lane->radius * sin(theta2) + off2.y
            };

            SDL_Point p1 = to_screen_coords(w1, width, height);
            SDL_Point p2 = to_screen_coords(w2, width, height);

            thickLineRGBA_ignore_if_outside_screen(renderer, p1.x, p1.y, p2.x, p2.y, thickness, color.r, color.g, color.b, color.a);
        }
    }
}

// Renders a rectangular lane section with optional lane lines and arrows.
void render_lane_linear(SDL_Renderer* renderer, const Lane* lane, Map* map, bool paint_lines, bool paint_arrows, bool paint_id) {
    int width = WINDOW_SIZE_WIDTH;
    int height = WINDOW_SIZE_HEIGHT;

    // if it's an extremely short lane, skip painting
    if (lane->length < LANE_LENGTH_EPSILON) {
        paint_lines = false;
        paint_arrows = false;
        paint_id = false;
    }

    // Determine bounding box for rendering the lane.
    double x_min, x_max, y_min, y_max;
    if (lane->direction == DIRECTION_EAST || lane->direction == DIRECTION_WEST) {
        x_min = fmin(lane->start_point.x, lane->end_point.x);
        x_max = fmax(lane->start_point.x, lane->end_point.x);
        y_min = lane->center.y - lane->width / 2;
        y_max = lane->center.y + lane->width / 2;
    } else {
        y_min = fmin(lane->start_point.y, lane->end_point.y);
        y_max = fmax(lane->start_point.y, lane->end_point.y);
        x_min = lane->center.x - lane->width / 2;
        x_max = lane->center.x + lane->width / 2;
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
    SDL_Color lane_color = should_highlight_lane(lane) ? HIGHLIGHTED_LANE_COLOR : ROAD_COLOR;
    SDL_SetRenderDrawColor(renderer, lane_color.r, lane_color.g, lane_color.b, lane_color.a);
    SDL_RenderFillRect_ignore_if_outside_screen(renderer, &lane_rect);

    // Determine left/right edge line positions
    Coordinates left_start, left_end, right_start, right_end;
    if (lane->direction == DIRECTION_EAST || lane->direction == DIRECTION_WEST) {
        double left_y = (lane->direction == DIRECTION_EAST) ? y_max : y_min;
        double right_y = (lane->direction == DIRECTION_EAST) ? y_min : y_max;
        left_start = (Coordinates){x_min, left_y};
        left_end = (Coordinates){x_max, left_y};
        right_start = (Coordinates){x_min, right_y};
        right_end = (Coordinates){x_max, right_y};
    } else {
        double left_x = (lane->direction == DIRECTION_NORTH) ? x_min : x_max;
        double right_x = (lane->direction == DIRECTION_NORTH) ? x_max : x_min;
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
        Lane* adjacent_left = lane_get_adjacent_left(lane, map);
        if (adjacent_left && adjacent_left->direction != lane->direction) {
            thickLineRGBA_ignore_if_outside_screen(renderer, lstart.x, lstart.y, lend.x, lend.y, ythick, yellow.r, yellow.g, yellow.b, yellow.a);
        } else if (!adjacent_left) {
            thickLineRGBA_ignore_if_outside_screen(renderer, lstart.x, lstart.y, lend.x, lend.y, wthick, white.r, white.g, white.b, white.a);
        } else {
            draw_dotted_line(renderer, lstart, lend, white);
        }

        // Right edge line
        Lane* adjacent_right = lane_get_adjacent_right(lane, map);
        if (!adjacent_right) {
            thickLineRGBA_ignore_if_outside_screen(renderer, rstart.x, rstart.y, rend.x, rend.y, wthick, white.r, white.g, white.b, white.a);
        } else if (adjacent_right->direction == lane->direction) {
            draw_dotted_line(renderer, rstart, rend, white);
        } else {
            LOG_ERROR("Right edge cannot be opposite direction!");
            exit(EXIT_FAILURE);
        }
    }

    if (paint_arrows) {
        // Calculate three arrow positions along the lane
        Coordinates mid = vec_div(vec_add(vec_scale(lane->start_point, 45), vec_scale(lane->end_point, 55)), 100);
        Coordinates a1 = vec_div(vec_add(vec_scale(lane->start_point, 4), lane->end_point), 5);
        Coordinates a2 = vec_div(vec_add(lane->start_point, vec_scale(lane->end_point, 3)), 4);

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

            switch (lane->direction) {
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
        snprintf(id_str, sizeof(id_str), "%d", lane->id);
        int font_size = (int)(meters(2.0) * SCALE); // font size = 2 meter
        SDL_Color text_color = {255, 255, 255, 255}; // white text color
        SDL_Point lane_center_screen = to_screen_coords(lane->center, width, height);
        int text_x = lane_center_screen.x;
        int text_y = lane_center_screen.y;
        bool rotated = (lane->direction == DIRECTION_NORTH || lane->direction == DIRECTION_SOUTH);
        render_text(renderer, id_str, text_x, text_y, text_color.r, text_color.g, text_color.b, text_color.a, font_size, ALIGN_CENTER, rotated, lane_id_texture_cache[lane->id]);
    }
}


// Renders a quarter-arc lane, optionally painting boundary lines and directional arrows.
void render_lane_quarterarc(SDL_Renderer* renderer, const Lane* lane, Map* map,  bool paint_lines, bool paint_arrows, bool paint_id) {
    int width = WINDOW_SIZE_WIDTH;
    int height = WINDOW_SIZE_HEIGHT;

    SDL_Point center_screen = to_screen_coords(lane->center, width, height);

    // Define inner and outer radii of the arc-shaped lane
    double r_inner = lane->radius - lane->width / 2;
    double r_outer = lane->radius + lane->width / 2;

    // Calculate offsets for smooth interpolation (fix for asymmetric intersections)
    Vec2D theoretical_start = {
        lane->center.x + lane->radius * cos(lane->start_angle),
        lane->center.y + lane->radius * sin(lane->start_angle)
    };
    Vec2D theoretical_end = {
        lane->center.x + lane->radius * cos(lane->end_angle),
        lane->center.y + lane->radius * sin(lane->end_angle)
    };
    
    Lane* incoming = lane_get_connection_incoming_straight(lane, map);
    Lane* outgoing = lane_get_connection_straight(lane, map);
    
    Vec2D actual_start = incoming ? incoming->end_point : lane->start_point;
    Vec2D actual_end = outgoing ? outgoing->start_point : lane->end_point;
    
    Vec2D offset_start = vec_sub(actual_start, theoretical_start);
    Vec2D offset_end = vec_sub(actual_end, theoretical_end);

    // Prepare vertex arrays to define the polygon shape of the lane
    Sint16 vx[2 * (ARC_NUM_POINTS + 1)];
    Sint16 vy[2 * (ARC_NUM_POINTS + 1)];
    double angle_step = (lane->end_angle - lane->start_angle) / ARC_NUM_POINTS;
    int idx = 0;

    // Fill inner arc points (counter-clockwise)
    for (int i = 0; i <= ARC_NUM_POINTS; i++) {
        double p = (double)i / ARC_NUM_POINTS;
        double angle = lane->start_angle + i * angle_step;
        
        Vec2D offset = vec_add(vec_scale(offset_start, 1.0 - p), vec_scale(offset_end, p));
        Coordinates pt = {
            lane->center.x + r_inner * cos(angle) + offset.x,
            lane->center.y + r_inner * sin(angle) + offset.y
        };
        SDL_Point screen_pt = to_screen_coords(pt, width, height);
        vx[idx] = screen_pt.x;
        vy[idx] = screen_pt.y;
        idx++;
    }

    // Fill outer arc points (clockwise)
    for (int i = ARC_NUM_POINTS; i >= 0; i--) {
        double p = (double)i / ARC_NUM_POINTS;
        double angle = lane->start_angle + i * angle_step;

        Vec2D offset = vec_add(vec_scale(offset_start, 1.0 - p), vec_scale(offset_end, p));
        Coordinates pt = {
            lane->center.x + r_outer * cos(angle) + offset.x,
            lane->center.y + r_outer * sin(angle) + offset.y
        };
        SDL_Point screen_pt = to_screen_coords(pt, width, height);
        vx[idx] = screen_pt.x;
        vy[idx] = screen_pt.y;
        idx++;
    }

    // Draw the lane surface (gray)
    SDL_Color lane_color = should_highlight_lane(lane) ? HIGHLIGHTED_LANE_COLOR : ROAD_COLOR;
    filledPolygonRGBA_ignore_if_outside_screen(renderer, vx, vy, 2 * (ARC_NUM_POINTS + 1), lane_color.r, lane_color.g, lane_color.b, lane_color.a);

    if (paint_lines) {
        // Determine appropriate color and thickness for the left boundary
        SDL_Color left_line_color = {255, 255, 255, 255};
        int left_thickness;
        bool is_left_dotted = false;
        Lane* adjacent_left = lane_get_adjacent_left(lane, map);
        if (adjacent_left && adjacent_left->direction != lane->direction) {
            left_line_color = (SDL_Color){255, 255, 0, 255}; // Yellow for opposing direction
            left_thickness = (int)(YELLOW_LINE_THICKNESS * SCALE);
        } else if (!adjacent_left) {
            left_thickness = (int)(WHITE_LINE_THICKNESS * SCALE);
        } else {
            left_thickness = (int)(DOTTED_LINE_THICKNESS * SCALE);
            is_left_dotted = true; // Use dotted line for same direction adjacent lanes
        }
        left_thickness = fclamp(left_thickness, 1, 1);

        // Determine appropriate color and thickness for the right boundary
        SDL_Color right_line_color = {255, 255, 255, 255};
        int right_thickness;
        bool is_right_dotted = false;
        Lane* adjacent_right = lane_get_adjacent_right(lane, map);
        if (!adjacent_right) {
            // Fixed white line for right side
            right_thickness = (int)(WHITE_LINE_THICKNESS * SCALE);
        } else if (adjacent_right->direction == lane->direction) {
            right_thickness = (int)(DOTTED_LINE_THICKNESS * SCALE);
            is_right_dotted = true; // Use dotted line for same direction adjacent lanes
        } else {
            LOG_ERROR("Right edge cannot be opposite direction!");
            exit(EXIT_FAILURE);
        }
        right_thickness = fclamp(right_thickness, 1, 1);

        // Assign correct radius based on turning direction
        double r_left = (lane->direction == DIRECTION_CCW) ? r_inner : r_outer;
        double r_right = (lane->direction == DIRECTION_CCW) ? r_outer : r_inner;



        // Draw arc segments for right line
        int arc_num_points = is_right_dotted ? ARC_NUM_POINTS * 2 : ARC_NUM_POINTS;  // Double points for dotted lines since we skip every second point
        angle_step = (lane->end_angle - lane->start_angle) / arc_num_points;
        for (int i = 0; i < arc_num_points; i++) {
            double p1_ratio = (double)i / arc_num_points;
            double p2_ratio = (double)(i + 1) / arc_num_points;
            
            double theta1 = lane->start_angle + i * angle_step;
            double theta2 = lane->start_angle + (i + 1) * angle_step;
            
            Vec2D off1 = vec_add(vec_scale(offset_start, 1.0 - p1_ratio), vec_scale(offset_end, p1_ratio));
            Vec2D off2 = vec_add(vec_scale(offset_start, 1.0 - p2_ratio), vec_scale(offset_end, p2_ratio));

            // Right boundary line
           if (is_right_dotted && i % 2 == 0) continue; // Skip every second point for dotted lines
            SDL_Point p1_right = {
                (int)(center_screen.x + r_right * SCALE * cos(theta1) + off1.x * SCALE),
                (int)(center_screen.y - (r_right * sin(theta1) + off1.y) * SCALE) // Correct Y axis handled by transform? NO.
            };
            
            // Wait, to_screen_coords handles flipping Y.
            // Let's use world coords first.
            Coordinates w1 = {
                lane->center.x + r_right * cos(theta1) + off1.x,
                lane->center.y + r_right * sin(theta1) + off1.y
            };
            Coordinates w2 = {
                lane->center.x + r_right * cos(theta2) + off2.x,
                lane->center.y + r_right * sin(theta2) + off2.y
            };
            
            p1_right = to_screen_coords(w1, width, height);
            SDL_Point p2_right = to_screen_coords(w2, width, height);
            
            thickLineRGBA_ignore_if_outside_screen(renderer, p1_right.x, p1_right.y, p2_right.x, p2_right.y,
                        right_thickness, right_line_color.r, right_line_color.g, right_line_color.b, right_line_color.a);
        }

        // Draw arc segments for left line
        arc_num_points = is_left_dotted ? ARC_NUM_POINTS * 2 : ARC_NUM_POINTS;
        angle_step = (lane->end_angle - lane->start_angle) / arc_num_points;
        for (int i = 0; i < arc_num_points; i++) {
            double p1_ratio = (double)i / arc_num_points;
            double p2_ratio = (double)(i + 1) / arc_num_points;

            double theta1 = lane->start_angle + i * angle_step;
            double theta2 = lane->start_angle + (i + 1) * angle_step;
            
            Vec2D off1 = vec_add(vec_scale(offset_start, 1.0 - p1_ratio), vec_scale(offset_end, p1_ratio));
            Vec2D off2 = vec_add(vec_scale(offset_start, 1.0 - p2_ratio), vec_scale(offset_end, p2_ratio));

            // Left boundary line
            if (is_left_dotted && i % 2 == 0) continue; // Skip every second point for dotted lines
            
            Coordinates w1 = {
                lane->center.x + r_left * cos(theta1) + off1.x,
                lane->center.y + r_left * sin(theta1) + off1.y
            };
            Coordinates w2 = {
                lane->center.x + r_left * cos(theta2) + off2.x,
                lane->center.y + r_left * sin(theta2) + off2.y
            };
            
            SDL_Point p1_left = to_screen_coords(w1, width, height);
            SDL_Point p2_left = to_screen_coords(w2, width, height);

            thickLineRGBA_ignore_if_outside_screen(renderer, p1_left.x, p1_left.y, p2_left.x, p2_left.y,
                        left_thickness, left_line_color.r, left_line_color.g, left_line_color.b, left_line_color.a);
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
        if (lane->direction == DIRECTION_CW)
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
        snprintf(id_str, sizeof(id_str), "%d", lane->id);
        int font_size = (int)(meters(2.0) * SCALE); // font size = 2 meter
        SDL_Color text_color = {255, 255, 255, 255}; // white text color
        double middle_radius = (r_inner + r_outer) / 2;
        double middle_angle = lane->start_angle + (lane->end_angle - lane->start_angle) / 2.0; // Middle angle of the arc
        int text_x = center_screen.x + (int)(middle_radius * SCALE * cos(middle_angle));
        int text_y = center_screen.y - (int)(middle_radius * SCALE * sin(middle_angle));
        render_text(renderer, id_str, text_x, text_y, text_color.r, text_color.g, text_color.b, text_color.a, font_size, ALIGN_CENTER, false, lane_id_texture_cache[lane->id]);
    }
}


void render_lane(SDL_Renderer* renderer, const Lane* lane, Map* map, bool paint_lines, bool paint_arrows, bool paint_id) {
    if (lane->type == LINEAR_LANE) {
        render_lane_linear(renderer, lane, map, paint_lines, paint_arrows, paint_id);
    } else if (lane->type == QUARTER_ARC_LANE) {
        render_lane_quarterarc(renderer, lane, map, paint_lines, paint_arrows, paint_id);
    } else {
        LOG_ERROR("Error: Unknown lane type!");
    }
}
