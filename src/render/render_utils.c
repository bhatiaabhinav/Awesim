#include "render.h"
#include "logging.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL2/SDL_ttf.h>
#include <math.h>

int thickLineRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Uint8 width, Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
    // Check if the line is outside the screen bounds
    if ((x1 < 0 && x2 < 0) || (x1 >= WINDOW_SIZE_WIDTH && x2 >= WINDOW_SIZE_WIDTH) ||
        (y1 < 0 && y2 < 0) || (y1 >= WINDOW_SIZE_HEIGHT && y2 >= WINDOW_SIZE_HEIGHT)) {
        return 0; // Line is completely outside the screen, do not draw
    }
    // Draw the line using thickLineRGBA
    return thickLineRGBA(renderer, x1, y1, x2, y2, width, r, g, b, a);
}

int SDL_RenderDrawLine_ignore_if_outside_screen(SDL_Renderer * renderer, int x1, int y1, int x2, int y2) {
    // Check if the line is outside the screen bounds
    if ((x1 < 0 && x2 < 0) || (x1 >= WINDOW_SIZE_WIDTH && x2 >= WINDOW_SIZE_WIDTH) ||
        (y1 < 0 && y2 < 0) || (y1 >= WINDOW_SIZE_HEIGHT && y2 >= WINDOW_SIZE_HEIGHT)) {
        return 0; // Line is completely outside the screen, do not draw
    }
    // Draw the line using SDL_RenderDrawLine
    return SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
}

int SDL_RenderFillRect_ignore_if_outside_screen(SDL_Renderer * renderer, const SDL_Rect * rect) {
    // Check if the rectangle is outside the screen bounds
    if ((rect->x + rect->w < 0) || (rect->x > WINDOW_SIZE_WIDTH) ||
        (rect->y + rect->h < 0) || (rect->y > WINDOW_SIZE_HEIGHT)) {
        return 0; // Rectangle is completely outside the screen, do not draw
    }
    // Draw the rectangle using SDL_RenderFillRect
    return SDL_RenderFillRect(renderer, rect);
}

int filledTrigonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
    // Check if the triangle is outside the screen bounds
    if ((x1 < 0 && x2 < 0 && x3 < 0) || (x1 >= WINDOW_SIZE_WIDTH && x2 >= WINDOW_SIZE_WIDTH && x3 >= WINDOW_SIZE_WIDTH) ||
        (y1 < 0 && y2 < 0 && y3 < 0) || (y1 >= WINDOW_SIZE_HEIGHT && y2 >= WINDOW_SIZE_HEIGHT && y3 >= WINDOW_SIZE_HEIGHT)) {
        return 0; // Triangle is completely outside the screen, do not draw
    }
    // Draw the filled triangle using filledTrigonRGBA
    return filledTrigonRGBA(renderer, x1, y1, x2, y2, x3, y3, r, g, b, a);
}

int trigonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
    // Check if the triangle is outside the screen bounds
    if ((x1 < 0 && x2 < 0 && x3 < 0) || (x1 >= WINDOW_SIZE_WIDTH && x2 >= WINDOW_SIZE_WIDTH && x3 >= WINDOW_SIZE_WIDTH) ||
        (y1 < 0 && y2 < 0 && y3 < 0) || (y1 >= WINDOW_SIZE_HEIGHT && y2 >= WINDOW_SIZE_HEIGHT && y3 >= WINDOW_SIZE_HEIGHT)) {
        return 0; // Triangle is completely outside the screen, do not draw
    }
    // Draw the triangle using trigonRGBA
    return trigonRGBA(renderer, x1, y1, x2, y2, x3, y3, r, g, b, a);
}

int filledPolygonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, const Sint16 * vx, const Sint16 * vy, int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
    if (n < 3) {
        return 0; // Invalid polygon
    }
    Sint16 min_x = vx[0], max_x = vx[0];
    Sint16 min_y = vy[0], max_y = vy[0];
    for (int i = 1; i < n; i++) {
        if (vx[i] < min_x) min_x = vx[i];
        if (vx[i] > max_x) max_x = vx[i];
        if (vy[i] < min_y) min_y = vy[i];
        if (vy[i] > max_y) max_y = vy[i];
    }
    if (max_x < 0 || min_x >= WINDOW_SIZE_WIDTH || max_y < 0 || min_y >= WINDOW_SIZE_HEIGHT) {
        return 0; // Polygon is completely outside
    }
    return filledPolygonRGBA(renderer, vx, vy, n, r, g, b, a);
}

int polygonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, const Sint16 * vx, const Sint16 * vy, int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
    if (n < 3) {
        return 0; // Invalid polygon
    }
    Sint16 min_x = vx[0], max_x = vx[0];
    Sint16 min_y = vy[0], max_y = vy[0];
    for (int i = 1; i < n; i++) {
        if (vx[i] < min_x) min_x = vx[i];
        if (vx[i] > max_x) max_x = vx[i];
        if (vy[i] < min_y) min_y = vy[i];
        if (vy[i] > max_y) max_y = vy[i];
    }
    if (max_x < 0 || min_x >= WINDOW_SIZE_WIDTH || max_y < 0 || min_y >= WINDOW_SIZE_HEIGHT) {
        return 0; // Polygon is completely outside
    }
    return polygonRGBA(renderer, vx, vy, n, r, g, b, a);
}


void draw_dotted_line(SDL_Renderer* renderer, const SDL_Point start, const SDL_Point end, const SDL_Color color) {
    int dash_px = (int)(DOTTED_LINE_LENGTH * SCALE);
    int gap_px = (int)(DOTTED_LINE_LENGTH_GAP * SCALE);
    int thickness = (int)(DOTTED_LINE_THICKNESS * SCALE);
    thickness = fclamp(thickness, 1, 1);

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

        thickLineRGBA_ignore_if_outside_screen(renderer, p1.x, p1.y, p2.x, p2.y, thickness, color.r, color.g, color.b, color.a);
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
            thickLineRGBA_ignore_if_outside_screen(renderer, p1.x, p1.y, p2.x, p2.y, 1, 255, 255, 255, 255);
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
        SDL_RenderDrawLine_ignore_if_outside_screen(renderer, x1, cy, x2, cy);
    }

    // Fill center rectangle
    if (y + radius <= y + height - radius) {
        SDL_Rect mid = { x, y + radius, width, height - 2 * radius };
        SDL_RenderFillRect_ignore_if_outside_screen(renderer, &mid);
    }

    // Bottom region (rounded bottom)
    int y_bot_start = y + height - radius;
    for (int cy = y_bot_start; cy <= y + height; cy++) {
        double dy = (y + height) - cy;
        double offset = sqrt(radius * radius - dy * dy);
        int x1 = x + (int)offset;
        int x2 = x + width - (int)offset;
        SDL_RenderDrawLine_ignore_if_outside_screen(renderer, x1, cy, x2, cy);
    }

    // White rounded arc outlines (thick border)
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    int thickness = (int)(WHITE_LINE_THICKNESS * SCALE);
    thickness = fclamp(thickness, 1, 1);
    drawQuarterCircleOutline(renderer, x, y, radius, 1, thickness);
    drawQuarterCircleOutline(renderer, x + width, y, radius, 2, thickness);
    drawQuarterCircleOutline(renderer, x, y + height, radius, 3, thickness);
    drawQuarterCircleOutline(renderer, x + width, y + height, radius, 4, thickness);

    SDL_SetRenderDrawColor(renderer, r, g, b, a); // Restore color
}


void render_text(SDL_Renderer* renderer, const char* text, int x, int y, Uint8 r, Uint8 g, Uint8 b, Uint8 a, int font_size, TextAlign align, bool rotated, SDL_Texture** cache) {
    if (!fonts_initialized) {
        LOG_ERROR("Fonts not initialized", font_size);
        return;
    }

    if (font_size < 1 || font_size > MAX_FONT_SIZE) {
        LOG_WARN("Invalid font size %d", font_size);
        font_size = fclamp(font_size, 1, MAX_FONT_SIZE);
    }

    // cache = NULL;
    SDL_Texture* texture = cache ? cache[font_size - 1] : NULL;
    // SDL_Texture* texture = NULL;
    int text_w;
    int text_h;
    if (!texture) {
        // Get font from cache (size 1 maps to index 0, size 32 to index 31)
        TTF_Font* font = font_cache[font_size - 1];
        if (!font) {
            LOG_ERROR("Font for size %d not available", font_size);
            return;
        }

        // Create text surface
        SDL_Color color = {r, g, b, a};
        SDL_Surface* surface = TTF_RenderText_Solid(font, text, color);
        if (!surface) {
            LOG_ERROR("TTF_RenderText_Solid failed: %s", TTF_GetError());
            return;
        }

        // Create texture from surface
        texture = SDL_CreateTextureFromSurface(renderer, surface);
        SDL_FreeSurface(surface);
        if (!texture) {
            LOG_ERROR("SDL_CreateTextureFromSurface failed: %s", SDL_GetError());
            return;
        }
        if (cache) cache[font_size - 1] = texture; // Cache the texture for future use
    }

    SDL_QueryTexture(texture, NULL, NULL, &text_w, &text_h);
    if (cache && cache[font_size - 1]) {
        LOG_TRACE("Using cached texture for font size %d. Obtained dimensions: %dx%d", font_size, text_w, text_h);
    }

    // Adjust dimensions for rotation (90 degrees anticlockwise swaps width and height)
    int render_w = rotated ? text_h : text_w;
    int render_h = rotated ? text_w : text_h;

    // Adjust position based on alignment using rotated dimensions
    int render_x = x;
    int render_y = y;
    switch (align) {
        case ALIGN_TOP_LEFT:
            break;
        case ALIGN_TOP_CENTER:
            render_x = x - render_w / 2;    // Center horizontally
            break;
        case ALIGN_TOP_RIGHT:
            render_x = x - render_w;        // Right edge at x
            break;
        case ALIGN_CENTER_LEFT:
            render_y = y - render_h / 2;    // Center vertically
            break;
        case ALIGN_CENTER:
            render_x = x - render_w / 2;    // Center horizontally
            render_y = y - render_h / 2;    // Center vertically
            break;
        case ALIGN_CENTER_RIGHT:
            render_x = x - render_w;        // Right edge at x        
            render_y = y - render_h / 2;    // Center vertically    
            break;
        case ALIGN_BOTTOM_LEFT:
            render_y = y - render_h;        // Bottom edge at y
            break;
        case ALIGN_BOTTOM_CENTER:
            render_x = x - render_w / 2;    // Center horizontally
            render_y = y - render_h;        
            break;
        case ALIGN_BOTTOM_RIGHT:
            render_x = x - render_w;        // Right edge at x        
            render_y = y - render_h;        // Bottom edge at y
            break;
    }

    // Adjust for rotation offset (90 degrees anticlockwise around top-left)
    if (rotated) {
        render_y += text_w; // Adjust y to account for the height swap
    }

    // Check if text is outside screen bounds
    if (render_x < 0 || render_x + render_w >= WINDOW_SIZE_WIDTH || render_y < 0 || render_y + render_h >= WINDOW_SIZE_HEIGHT) {
        if (!cache) SDL_DestroyTexture(texture);
        return; // Text is completely outside the screen
    }

    // Render text
    SDL_Rect dst = {render_x, render_y, text_w, text_h};
    if (rotated) {
        // Rotate 90 degrees anticlockwise around top-left
        SDL_Point center = {0, 0};
        SDL_RenderCopyEx(renderer, texture, NULL, &dst, -90.0, &center, SDL_FLIP_NONE);
    } else {
        SDL_RenderCopy(renderer, texture, NULL, &dst);
    }

    // Clean up
    if (!cache) SDL_DestroyTexture(texture);
}