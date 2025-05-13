#include "render.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>

SDL_Point to_screen_coords(const Coordinates point, const int width, const int height) {
    double x_scaled = point.x * SCALE;
    double y_scaled = point.y * SCALE;
    int x_screen = (int)(width / 2 + x_scaled);
    int y_screen = (int)(height / 2 - y_scaled);
    return (SDL_Point){x_screen, y_screen};
}

void draw_dotted_line(SDL_Renderer* renderer, const SDL_Point start, const SDL_Point end, const SDL_Color color) {
    int dash_px = (int)(DOTTED_LINE_LENGTH * SCALE);
    int gap_px = (int)(DOTTED_LINE_LENGTH_GAP * SCALE);
    int thickness = (int)(DOTTED_LINE_THICKNESS * SCALE);

    int dx = end.x - start.x;
    int dy = end.y - start.y;
    double total_len = sqrt(dx * dx + dy * dy);
    if (total_len <= 0.0) return;

    int segment_len = dash_px + gap_px;
    int num_segments = fmax(1, (int)(total_len / segment_len));
    double dash_len = (total_len - (num_segments * gap_px)) / num_segments;

    for (int j = 0; j < num_segments; j++) {
        double t_start = (gap_px / 2.0 + j * (dash_len + gap_px)) / total_len;
        double t_end = t_start + dash_len / total_len;

        SDL_Point p1 = { (int)(start.x + dx * t_start), (int)(start.y + dy * t_start) };
        SDL_Point p2 = { (int)(start.x + dx * t_end),   (int)(start.y + dy * t_end) };

        thickLineRGBA(renderer, p1.x, p1.y, p2.x, p2.y, thickness, color.r, color.g, color.b, color.a);
    }
}

void drawQuarterCircleOutline(SDL_Renderer* renderer, int cx, int cy, int radius, int quadrant, int thickness) {
    if (radius < 1 || thickness < 1) return;

    int start_deg = 0, end_deg = 90;
    switch (quadrant) {
        case 1: start_deg = 0; end_deg = 90; break;       // Bottom-right
        case 2: start_deg = 90; end_deg = 180; break;     // Bottom-left
        case 3: start_deg = 270; end_deg = 360; break;    // Top-right
        case 4: start_deg = 180; end_deg = 270; break;    // Top-left
        default: return;
    }

    double start_rad = start_deg * M_PI / 180.0;
    double end_rad = end_deg * M_PI / 180.0;
    int segments = ARC_NUM_POINTS;
    double dtheta = (end_rad - start_rad) / segments;

    for (int r = radius - thickness / 2; r <= radius + thickness / 2; r++) {
        if (r < 1) continue;
        for (int i = 0; i < segments; i++) {
            double t1 = start_rad + i * dtheta;
            double t2 = t1 + dtheta;
            SDL_Point p1 = { (int)(cx + r * cos(t1)), (int)(cy + r * sin(t1)) };
            SDL_Point p2 = { (int)(cx + r * cos(t2)), (int)(cy + r * sin(t2)) };
            thickLineRGBA(renderer, p1.x, p1.y, p2.x, p2.y, 1, 255, 255, 255, 255);
        }
    }
}


void drawFilledInwardRoundedRect(SDL_Renderer* renderer, int x, int y, int width, int height, int radius) {
    if (width <= 0 || height <= 0) return;

    radius = fmin(radius, fmin(width / 2, height / 2));

    // Save original draw color
    Uint8 r, g, b, a;
    SDL_GetRenderDrawColor(renderer, &r, &g, &b, &a);

    // Top region (rounded top)
    int y_top_end = y + radius;
    for (int cy = y; cy < y_top_end; cy++) {
        double dy = cy - y;
        double offset = sqrt(radius * radius - dy * dy);
        int x1 = x + (int)offset;
        int x2 = x + width - (int)offset;
        SDL_RenderDrawLine(renderer, x1, cy, x2, cy);
    }

    // Fill center rectangle
    if (y + radius <= y + height - radius) {
        SDL_Rect mid = { x, y + radius, width, height - 2 * radius };
        SDL_RenderFillRect(renderer, &mid);
    }

    // Bottom region (rounded bottom)
    int y_bot_start = y + height - radius;
    for (int cy = y_bot_start; cy <= y + height; cy++) {
        double dy = (y + height) - cy;
        double offset = sqrt(radius * radius - dy * dy);
        int x1 = x + (int)offset;
        int x2 = x + width - (int)offset;
        SDL_RenderDrawLine(renderer, x1, cy, x2, cy);
    }

    // White rounded arc outlines (thick border)
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    int thickness = (int)(WHITE_LINE_THICKNESS * SCALE);
    drawQuarterCircleOutline(renderer, x, y, radius, 1, thickness);
    drawQuarterCircleOutline(renderer, x + width, y, radius, 2, thickness);
    drawQuarterCircleOutline(renderer, x, y + height, radius, 3, thickness);
    drawQuarterCircleOutline(renderer, x + width, y + height, radius, 4, thickness);

    SDL_SetRenderDrawColor(renderer, r, g, b, a); // Restore color
}

