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
