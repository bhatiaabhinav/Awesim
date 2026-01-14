#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//
// Vec2D functions
//

Vec2D vec_create(double x, double y) {
    return (Vec2D){x, y};
}

Vec2D vec_add(Vec2D v1, Vec2D v2) {
    return (Vec2D){v1.x + v2.x, v1.y + v2.y};
}

Vec2D vec_sub(Vec2D v1, Vec2D v2) {
    return (Vec2D){v1.x - v2.x, v1.y - v2.y};
}

Vec2D vec_scale(Vec2D v, double scalar) {
    return (Vec2D){v.x * scalar, v.y * scalar};
}

Vec2D vec_div(Vec2D v, double scalar) {
    return (Vec2D){v.x / scalar, v.y / scalar};
}

double vec_dot(Vec2D v1, Vec2D v2) {
    return v1.x * v2.x + v1.y * v2.y;
}

double vec_magnitude(Vec2D v) {
    return sqrt(v.x * v.x + v.y * v.y);
}

double vec_distance(Vec2D v1, Vec2D v2) {
    return vec_magnitude(vec_sub(v1, v2));
}

Vec2D vec_normalize(Vec2D v) {
    double mag = vec_magnitude(v);
    if (mag == 0.0)
        return (Vec2D){0, 0};
    return vec_div(v, mag);
}

Vec2D vec_rotate(Vec2D v, double angle) {
    double cos_a = cos(angle);
    double sin_a = sin(angle);
    return (Vec2D){
        v.x * cos_a - v.y * sin_a,
        v.x * sin_a + v.y * cos_a
    };
}

Vec2D vec_midpoint(Vec2D v1, Vec2D v2) {
    return (Vec2D){(v1.x + v2.x) / 2.0, (v1.y + v2.y) / 2.0};
}

bool vec_approx_equal(Vec2D a, Vec2D b, double epsilon) {
    return vec_distance(a, b) < epsilon;
}

//
// Coordinates and Dimensions
//

Coordinates coordinates_create(double x, double y) {
    return vec_create(x, y);
}

Dimensions dimensions_create(double width, double length) {
    return vec_create(width, length);
}

//
// Line Segments
//

LineSegment line_segment_create(Coordinates start, Coordinates end) {
    return (LineSegment){start, end};
}

LineSegment line_segment_from_center(Vec2D center, Vec2D direction, double length) {
    Vec2D half = vec_scale(vec_normalize(direction), length / 2.0);
    Coordinates start = vec_sub(center, half);
    Coordinates end = vec_add(center, half);
    return line_segment_create(start, end);
}

Vec2D line_segment_unit_vector(LineSegment line) {
    return vec_normalize(vec_sub(line.end, line.start));
}

// Returns the intersection point of two line segments, if they intersect. Returns true if they intersect, false otherwise.
bool line_segments_intersect(LineSegment line1, LineSegment line2, Coordinates* intersection_point) {
    double x1 = line1.start.x;
    double y1 = line1.start.y;
    double x2 = line1.end.x;
    double y2 = line1.end.y;

    double x3 = line2.start.x;
    double y3 = line2.start.y;
    double x4 = line2.end.x;
    double y4 = line2.end.y;

    double denom = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3);
    double numer_t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3));
    double numer_u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1));

    if (denom == 0) {
        if (numer_u == 0 && numer_t == 0) {
            // Lines are collinear
            double dx = x2 - x1;
            double dy = y2 - y1;
            double len_sq = dx * dx + dy * dy;

            if (len_sq == 0) {
                // Line 1 is a point. Check if it lies on Line 2.
                double min_x = (x3 < x4) ? x3 : x4;
                double max_x = (x3 > x4) ? x3 : x4;
                double min_y = (y3 < y4) ? y3 : y4;
                double max_y = (y3 > y4) ? y3 : y4;
                
                if (x1 >= min_x && x1 <= max_x && y1 >= min_y && y1 <= max_y) {
                     if (intersection_point) {
                        intersection_point->x = x1;
                        intersection_point->y = y1;
                     }
                     return true;
                }
                return false;
            }

            double t3 = ((x3 - x1) * dx + (y3 - y1) * dy) / len_sq;
            double t4 = ((x4 - x1) * dx + (y4 - y1) * dy) / len_sq;

            double t_min = (t3 < t4) ? t3 : t4;
            double t_max = (t3 > t4) ? t3 : t4;

            double t_start = (t_min > 0) ? t_min : 0;
            double t_end = (t_max < 1) ? t_max : 1;

            if (t_start <= t_end) {
                if (intersection_point) {
                    intersection_point->x = x1 + t_start * dx;
                    intersection_point->y = y1 + t_start * dy;
                }
                return true;
            }
            return false;
        }
        return false;
    }

    double t = numer_t / denom;
    double u = numer_u / denom;

    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        if (intersection_point != NULL) {
            intersection_point->x = x1 + t * (x2 - x1);
            intersection_point->y = y1 + t * (y2 - y1);
        }
        return true;
    }

    return false;
}

//
// Angle functions
//

Radians angles_add(Radians theta1, Radians theta2) {
    Radians result = theta1 + theta2;
    return angle_normalize(result);
}


Radians angles_sub(Radians theta1, Radians theta2) {
    theta1 = angle_normalize(theta1);
    theta2 = angle_normalize(theta2);
    Radians result = theta1 - theta2;
    return angle_normalize(result);
}

Radians angle_normalize(Radians theta) {
    while (theta < 0) {
        theta += 2 * M_PI;
    }
    while (theta >= 2 * M_PI) {
        theta -= 2 * M_PI;
    }
    return theta;
}

Quadrant angle_get_quadrant(Radians theta) {
    theta = angle_normalize(theta);
    if (theta >= 0 && theta < M_PI / 2) {
        return QUADRANT_TOP_RIGHT;
    } else if (theta >= M_PI / 2 && theta < M_PI) {
        return QUADRANT_TOP_LEFT;
    } else if (theta >= M_PI && theta < 3 * M_PI / 2) {
        return QUADRANT_BOTTOM_LEFT;
    } else {
        return QUADRANT_BOTTOM_RIGHT;
    }
}

Vec2D angle_to_unit_vector(Radians theta) {
    return vec_create(cos(theta), sin(theta));
}

Radians from_degrees(Degrees theta) {
    return theta * (M_PI / 180.0);
}

Degrees to_degrees(Radians theta) {
    return theta * (180.0 / M_PI);
}

//
// Direction vectors
//

Vec2D direction_to_vector(Direction direction) {
    switch (direction) {
        case DIRECTION_EAST:  return vec_create(1.0, 0.0);
        case DIRECTION_WEST:  return vec_create(-1.0, 0.0);
        case DIRECTION_NORTH: return vec_create(0.0, 1.0);
        case DIRECTION_SOUTH: return vec_create(0.0, -1.0);
        default:
            fprintf(stderr, "Unsupported direction: %d\n", direction);
            exit(EXIT_FAILURE);
    }
}

Vec2D direction_perpendicular_vector(Direction direction) {
    switch (direction) {
        case DIRECTION_EAST:  return vec_create(0.0, 1.0);
        case DIRECTION_WEST:  return vec_create(0.0, -1.0);
        case DIRECTION_NORTH: return vec_create(-1.0, 0.0);
        case DIRECTION_SOUTH: return vec_create(1.0, 0.0);
        default:
            fprintf(stderr, "Unsupported direction: %d\n", direction);
            exit(EXIT_FAILURE);
    }
}

Direction direction_opposite(Direction direction) {
    switch (direction) {
        case DIRECTION_EAST:  return DIRECTION_WEST;
        case DIRECTION_WEST:  return DIRECTION_EAST;
        case DIRECTION_NORTH: return DIRECTION_SOUTH;
        case DIRECTION_SOUTH: return DIRECTION_NORTH;
        case DIRECTION_CW:    return DIRECTION_CCW;
        case DIRECTION_CCW:   return DIRECTION_CW;
        default:
            fprintf(stderr, "Unsupported direction: %d\n", direction);
            exit(EXIT_FAILURE);
    }
}

Direction direction_perp_cw(Direction direction) {
    switch (direction) {
        case DIRECTION_EAST:  return DIRECTION_SOUTH;
        case DIRECTION_WEST:  return DIRECTION_NORTH;
        case DIRECTION_NORTH: return DIRECTION_EAST;
        case DIRECTION_SOUTH: return DIRECTION_WEST;
        default:
            fprintf(stderr, "Unsupported direction for perp_cw: %d\n", direction);
            exit(EXIT_FAILURE);
    }
}

Direction direction_perp_ccw(Direction direction) {
    switch (direction) {
        case DIRECTION_EAST:  return DIRECTION_NORTH;
        case DIRECTION_WEST:  return DIRECTION_SOUTH;
        case DIRECTION_NORTH: return DIRECTION_WEST;
        case DIRECTION_SOUTH: return DIRECTION_EAST;
        default:
            fprintf(stderr, "Unsupported direction for perp_ccw: %d\n", direction);
            exit(EXIT_FAILURE);
    }
}

//
// Rotated Rectangle Intersection
//

static void get_rect_axes(RotatedRect2D rect, Vec2D axes[2]) {
    // Optimize: compute cos/sin once
    double c = cos(rect.rotation);
    double s = sin(rect.rotation);
    
    // Axis 1: along the width (local x-axis) rotated (1, 0) -> (c, s)
    axes[0] = (Vec2D){c, s};
    // Axis 2: along the length (local y-axis) rotated (0, 1) -> (-s, c)
    axes[1] = (Vec2D){-s, c};
}

static void get_rect_corners(RotatedRect2D rect, Vec2D axes[2], Vec2D corners[4]) {
    double hw = rect.dimensions.x / 2.0;
    double hl = rect.dimensions.y / 2.0;
    
    Vec2D dx = vec_scale(axes[0], hw);
    Vec2D dy = vec_scale(axes[1], hl);
    
    // TR: center + dx + dy
    corners[0] = vec_add(rect.center, vec_add(dx, dy));
    // TL: center - dx + dy
    corners[1] = vec_add(rect.center, vec_sub(dy, dx));
    // BL: center - dx - dy
    corners[2] = vec_sub(rect.center, vec_add(dx, dy));
    // BR: center + dx - dy
    corners[3] = vec_sub(rect.center, vec_sub(dy, dx));
}

static void project_rect(Vec2D axis, Vec2D corners[4], double* min, double* max) {
    double dot = vec_dot(axis, corners[0]);
    *min = dot;
    *max = dot;
    for (int i = 1; i < 4; i++) {
        dot = vec_dot(axis, corners[i]);
        if (dot < *min) *min = dot;
        if (dot > *max) *max = dot;
    }
}

bool rotated_rects_intersect(RotatedRect2D rect1, RotatedRect2D rect2) {

    // Preliminary AABB Check

    Vec2D axes1[2];
    Vec2D axes2[2];
    get_rect_axes(rect1, axes1);
    get_rect_axes(rect2, axes2);
    
    // Calculate half-extents for AABB
    double w1_half = rect1.dimensions.x / 2.0;
    double h1_half = rect1.dimensions.y / 2.0;
    double ext1_x = w1_half * fabs(axes1[0].x) + h1_half * fabs(axes1[1].x);
    double ext1_y = w1_half * fabs(axes1[0].y) + h1_half * fabs(axes1[1].y);

    double w2_half = rect2.dimensions.x / 2.0;
    double h2_half = rect2.dimensions.y / 2.0;
    double ext2_x = w2_half * fabs(axes2[0].x) + h2_half * fabs(axes2[1].x);
    double ext2_y = w2_half * fabs(axes2[0].y) + h2_half * fabs(axes2[1].y);

    // Check AABB overlap
    if (fabs(rect1.center.x - rect2.center.x) > (ext1_x + ext2_x) ||
        fabs(rect1.center.y - rect2.center.y) > (ext1_y + ext2_y)) {
        return false;
    }

    // SAT Check
    Vec2D corners1[4];
    Vec2D corners2[4];
    get_rect_corners(rect1, axes1, corners1);
    get_rect_corners(rect2, axes2, corners2);
    
    Vec2D axes[4];
    axes[0] = axes1[0];
    axes[1] = axes1[1];
    axes[2] = axes2[0];
    axes[3] = axes2[1];

    
    // Check for overlap on all 4 axes
    for (int i = 0; i < 4; i++) {
        double min1, max1, min2, max2;
        project_rect(axes[i], corners1, &min1, &max1);
        project_rect(axes[i], corners2, &min2, &max2);
        
        if (max1 < min2 || max2 < min1) {
            return false; // Separating axis found
        }
    }
    
    return true;
}

