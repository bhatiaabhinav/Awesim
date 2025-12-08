#include "utils.h"
#include <math.h>
#include <stdbool.h>

//
// Vec 3D
//

Vec3D vec3d_create(double x, double y, double z) {
    Vec3D v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

Vec3D vec3d_add(Vec3D v1, Vec3D v2) {
    return vec3d_create(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

Vec3D vec3d_sub(Vec3D v1, Vec3D v2) {
    return vec3d_create(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

Vec3D vec3d_scale(Vec3D v, double scalar) {
    return vec3d_create(v.x * scalar, v.y * scalar, v.z * scalar);
}

Vec3D vec3d_div(Vec3D v, double scalar) {
    if (scalar == 0.0) {
        return vec3d_create(0, 0, 0); 
    }
    return vec3d_create(v.x / scalar, v.y / scalar, v.z / scalar);
}

double vec3d_dot(Vec3D v1, Vec3D v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

Vec3D vec3d_cross(Vec3D v1, Vec3D v2) {
    return vec3d_create(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    );
}

double vec3d_magnitude(Vec3D v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3D vec3d_normalize(Vec3D v) {
    double mag = vec3d_magnitude(v);
    if (mag == 0.0) {
        return vec3d_create(0, 0, 0);
    }
    return vec3d_div(v, mag);
}

double vec3d_distance(Vec3D v1, Vec3D v2) {
    return vec3d_magnitude(vec3d_sub(v1, v2));
}

Vec3D vec3d_midpoint(Vec3D v1, Vec3D v2) {
    return vec3d_scale(vec3d_add(v1, v2), 0.5);
}

bool vec3d_approx_equal(Vec3D a, Vec3D b, double epsilon) {
    return approxeq(a.x, b.x, epsilon) && 
           approxeq(a.y, b.y, epsilon) && 
           approxeq(a.z, b.z, epsilon);
}

//
// Coordinates3D
//

Coordinates3D coordinates3d_create(double x, double y, double z) {
    return vec3d_create(x, y, z);
}

//
// Dimensions3D
//

Dimensions3D dimensions3d_create(double width, double length, double altitude) {
    return vec3d_create(width, length, altitude);
}

//
// LineSegment3D
//

LineSegment3D line_segment3d_create(Coordinates3D start, Coordinates3D end) {
    LineSegment3D l;
    l.start = start;
    l.end = end;
    return l;
}

LineSegment3D line_segment3d_from_center(Coordinates3D center, Vec3D direction, double length) {
    Vec3D half_vec = vec3d_scale(vec3d_normalize(direction), length / 2.0);
    return line_segment3d_create(vec3d_sub(center, half_vec), vec3d_add(center, half_vec));
}

Vec3D line_segment3d_unit_vector(LineSegment3D line) {
    return vec3d_normalize(vec3d_sub(line.end, line.start));
}

// Helper: Check if point p is inside the quad defined by corners.
// Assumes p is on the plane of the quad.
static bool is_point_in_quad(Vec3D p, Quad3D quad, Vec3D normal) {
    for (int i = 0; i < 4; i++) {
        Vec3D edge = vec3d_sub(quad.corners[(i + 1) % 4], quad.corners[i]);
        Vec3D to_point = vec3d_sub(p, quad.corners[i]);
        Vec3D cross = vec3d_cross(edge, to_point);
        if (vec3d_dot(cross, normal) < -1e-9) return false; // Allow small epsilon
    }
    return true;
}

// Helper: Check if two coplanar 3D segments intersect
static bool segments_intersect_coplanar(Vec3D p1, Vec3D p2, Vec3D p3, Vec3D p4, Vec3D* intersection) {
    Vec3D da = vec3d_sub(p2, p1);
    Vec3D db = vec3d_sub(p4, p3);
    Vec3D dc = vec3d_sub(p3, p1);
    
    Vec3D cross_da_db = vec3d_cross(da, db);
    double denom = vec3d_dot(cross_da_db, cross_da_db);
    
    if (denom < 1e-18) {
        // Parallel or Collinear
        Vec3D cross_dc_db = vec3d_cross(dc, db);
        if (vec3d_dot(cross_dc_db, cross_dc_db) > 1e-18) return false; // Parallel, not collinear

        // Collinear - project to scalar on line p3-p4
        double len2 = vec3d_dot(db, db);
        if (len2 < 1e-18) return false; // Segment 2 is a point

        double t_p1 = vec3d_dot(vec3d_sub(p1, p3), db) / len2;
        double t_p2 = vec3d_dot(vec3d_sub(p2, p3), db) / len2;

        double t_min = (t_p1 < t_p2) ? t_p1 : t_p2;
        double t_max = (t_p1 > t_p2) ? t_p1 : t_p2;

        // Intersect [t_min, t_max] with [0, 1]
        double i_min = (t_min > 0.0) ? t_min : 0.0;
        double i_max = (t_max < 1.0) ? t_max : 1.0;

        if (i_min <= i_max) {
            if (intersection) *intersection = vec3d_add(p3, vec3d_scale(db, i_min));
            return true;
        }
        return false;
    }
    
    Vec3D num_t_vec = vec3d_cross(dc, db);
    double t = vec3d_dot(num_t_vec, cross_da_db) / denom;
    
    Vec3D num_u_vec = vec3d_cross(dc, da);
    double u = vec3d_dot(num_u_vec, cross_da_db) / denom;
    
    if (t >= -1e-9 && t <= 1.0 + 1e-9 && u >= -1e-9 && u <= 1.0 + 1e-9) {
        if (intersection) {
            // Clamp t to [0, 1] to avoid precision issues returning points slightly outside
            if (t < 0) t = 0;
            if (t > 1) t = 1;
            *intersection = vec3d_add(p1, vec3d_scale(da, t));
        }
        return true;
    }
    return false;
}

bool line_segment3d_intersect_quad(LineSegment3D line, Quad3D quad, Coordinates3D* intersection_point) {
    // 1. Calculate Plane Normal
    Vec3D v01 = vec3d_sub(quad.corners[1], quad.corners[0]);
    Vec3D v12 = vec3d_sub(quad.corners[2], quad.corners[1]);
    Vec3D normal = vec3d_cross(v01, v12);
    
    if (vec3d_magnitude(normal) < 1e-9) return false; // Degenerate quad
    normal = vec3d_normalize(normal);
    
    Vec3D dir = vec3d_sub(line.end, line.start);
    double denom = vec3d_dot(normal, dir);
    
    // 2. Check Parallel / Coplanar
    if (fabs(denom) < 1e-9) {
        // Check if coplanar
        Vec3D to_start = vec3d_sub(line.start, quad.corners[0]);
        if (fabs(vec3d_dot(to_start, normal)) > 1e-5) return false; // Parallel, distinct planes
        
        // Coplanar: Check if start is inside
        if (is_point_in_quad(line.start, quad, normal)) {
            if (intersection_point) *intersection_point = line.start;
            return true;
        }
        
        // Check intersection with edges
        double min_dist_sq = 1e30;
        bool found = false;
        Vec3D best_point = {0,0,0};
        
        for (int i = 0; i < 4; i++) {
            Vec3D pA = quad.corners[i];
            Vec3D pB = quad.corners[(i+1)%4];
            Vec3D pt;
            if (segments_intersect_coplanar(line.start, line.end, pA, pB, &pt)) {
                double d2 = vec3d_distance(line.start, pt); // Use distance as metric for "first" intersection
                // Actually squared distance is fine for comparison
                d2 = d2 * d2; 
                if (d2 < min_dist_sq) {
                    min_dist_sq = d2;
                    best_point = pt;
                    found = true;
                }
            }
        }
        
        if (found) {
            if (intersection_point) *intersection_point = best_point;
            return true;
        }
        return false;
    }
    
    // 3. Plane Intersection
    double t = vec3d_dot(vec3d_sub(quad.corners[0], line.start), normal) / denom;
    
    if (t < -1e-9 || t > 1.0 + 1e-9) return false;
    
    Vec3D p = vec3d_add(line.start, vec3d_scale(dir, t));
    
    // 4. Check if point is inside quad
    if (is_point_in_quad(p, quad, normal)) {
        if (intersection_point) *intersection_point = p;
        return true;
    }
    
    return false;
}

