#pragma once

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "road.h"
#include "car.h"
#include "sim.h"

// Rendering constants
#define WINDOW_SIZE_WIDTH 1000
#define WINDOW_SIZE_HEIGHT 1000

extern double SCALE;

extern int PAN_X;
extern int PAN_Y;

// Lane visualization constants
#define LANE_CENTER_LINE_THICKNESS from_inches(16)
#define DOTTED_LINE_LENGTH from_feet(10)
#define DOTTED_LINE_LENGTH_GAP from_feet(30)
#define DOTTED_LINE_THICKNESS from_inches(16)
#define WHITE_LINE_THICKNESS from_inches(16)
#define YELLOW_LINE_THICKNESS from_inches(16)
#define ARROW_SIZE from_feet(4.5)
#define CAR_COLOR (SDL_Color){74, 103, 65, 255}         // Olive greenish color
#define BACKGROUND_COLOR (SDL_Color){0, 0, 0, 255}  // black
#define LANE_CENTER_LINE_COLOR (SDL_Color){100, 100, 255, 255}  // blue
#define RED_LIGHT_COLOR (SDL_Color){200, 30, 30, 255}       // Soft red
#define GREEN_LIGHT_COLOR (SDL_Color){50, 200, 50, 255}     // Soft green
#define YELLOW_LIGHT_COLOR (SDL_Color){220, 180, 20, 255}   // Warm, soft yellow
#define FOUR_WAY_STOP_COLOR (SDL_Color){200, 30, 30, 64}    // Soft red with transparency
#define ARC_NUM_POINTS 10   // Number of points to approximate quarter arcs
#define MAX_FONT_SIZE 128

extern SDL_Texture* road_name_texture_cache[MAX_NUM_ROADS][MAX_FONT_SIZE];
extern SDL_Texture* lane_id_texture_cache[MAX_NUM_ROADS * MAX_NUM_LANES][MAX_FONT_SIZE];
extern SDL_Texture* car_id_texture_cache[MAX_CARS_IN_SIMULATION][MAX_FONT_SIZE];

// Converts world coordinates to screen coordinates relative to screen center.
SDL_Point to_screen_coords(const Coordinates point, const int width, const int height);

// Draws a lane's center line based on its geometry.
void render_lane_center_line(SDL_Renderer* renderer, const Lane* lane, const SDL_Color color, bool dotted);

// Renders a linear lane, optionally painting lane lines and arrows.
void render_lane_linear(SDL_Renderer* renderer, const LinearLane* lane, const bool paint_lines, const bool paint_arrows, const bool paint_id);

// Renders a quarter arc lane, optionally painting lane lines and arrows.
void render_lane_quarterarc(SDL_Renderer* renderer, const QuarterArcLane* lane, const bool paint_lines, const bool paint_arrows, const bool paint_id);

// Renders an intersection area and its traffic lights.
void render_intersection(SDL_Renderer* renderer, const Intersection* intersection);

// Renders a generic lane by delegating to the appropriate lane type renderer.
void render_lane(SDL_Renderer* renderer, const Lane* lane, const bool paint_lines, const bool paint_arrows, const bool paint_id);

// Draws a dotted line between two screen points using the specified color.
void draw_dotted_line(SDL_Renderer* renderer, const SDL_Point start, const SDL_Point end, const SDL_Color color);

// Draws a filled inward-rounded rectangle with arc outlines.
void drawFilledInwardRoundedRect(SDL_Renderer *renderer, const int x, const int y, const int width, const int height, const int radius);

// Draws a quarter-circle arc outline for decorative corners or turns.
void drawQuarterCircleOutline(SDL_Renderer *renderer, const int center_x, const int center_y, const int radius, const int quadrant, const int thickness);

// Renders a car based on its current lane and position.
void render_car(SDL_Renderer* renderer, const Car* car, const bool paint_id);

// Renders the entire simulation state: roads, cars, lanes, traffic.
void render_sim(SDL_Renderer* renderer, const Simulation* sim, const bool draw_lanes, const bool draw_cars, const bool draw_track_lines, const bool draw_traffic_lights, const bool draw_car_ids, const bool draw_lane_ids, const bool draw_road_names, const bool benchmark);

// thickLineRGBA(renderer, p1.x, p1.y, p2.x, p2.y, thickness, color.r, color.g, color.b, color.a);

int thickLineRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Uint8 width, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int SDL_RenderDrawLine_ignore_if_outside_screen(SDL_Renderer * renderer,
                                               int x1, int y1, int x2, int y2);

int SDL_RenderFillRect_ignore_if_outside_screen(SDL_Renderer * renderer, const SDL_Rect * rect);

int filledTrigonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int trigonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int filledPolygonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, const Sint16 * vx, const Sint16 * vy, int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int polygonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, const Sint16 * vx, const Sint16 * vy, int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int init_text_rendering(const char* font_path);
void cleanup_text_rendering();

// Alignment options
// Alignment options
typedef enum {
    ALIGN_TOP_LEFT,      // Text starts at (x, y)
    ALIGN_TOP_CENTER,    // Top edge centered at x, y at top
    ALIGN_TOP_RIGHT,     // Top-right corner at (x, y)
    ALIGN_CENTER_LEFT,   // Left edge at x, centered vertically
    ALIGN_CENTER,        // Centered at (x, y)
    ALIGN_CENTER_RIGHT,  // Right edge at x, centered vertically
    ALIGN_BOTTOM_LEFT,   // Bottom-left corner at (x, y)
    ALIGN_BOTTOM_CENTER, // Bottom edge centered at x, y at bottom
    ALIGN_BOTTOM_RIGHT   // Bottom-right corner at (x, y)
} TextAlign;


void render_text(SDL_Renderer* renderer, const char* text, int x, int y, Uint8 r, Uint8 g, Uint8 b, Uint8 a, int font_size, TextAlign align, bool rotated, SDL_Texture** cache);

void render_straight_road_name(SDL_Renderer* renderer, const StraightRoad* road);
void render_intersection_name(SDL_Renderer* renderer, const Intersection* intersection);