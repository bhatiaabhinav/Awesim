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

Dimensions dimensions_create(double width, double height) {
    return vec_create(width, height);
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
