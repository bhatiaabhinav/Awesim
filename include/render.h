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
#define RED_LIGHT_COLOR (SDL_Color){255, 0, 0, 255}  // red
#define GREEN_LIGHT_COLOR (SDL_Color){0, 255, 0, 255}  // green
#define YELLOW_LIGHT_COLOR (SDL_Color){255, 255, 0, 255}  // yellow
#define FOUR_WAY_STOP_COLOR (SDL_Color){255, 0, 0, 100}  // maroon
#define ARC_NUM_POINTS 10   // Number of points to approximate quarter arcs

// Converts world coordinates to screen coordinates relative to screen center.
SDL_Point to_screen_coords(const Coordinates point, const int width, const int height);

// Draws a lane's center line based on its geometry.
void render_lane_center_line(SDL_Renderer* renderer, const Lane* lane, const SDL_Color color);

// Renders a linear lane, optionally painting lane lines and arrows.
void render_lane_linear(SDL_Renderer* renderer, const LinearLane* lane, const bool paint_lines, const bool paint_arrows);

// Renders a quarter arc lane, optionally painting lane lines and arrows.
void render_lane_quarterarc(SDL_Renderer* renderer, const QuarterArcLane* lane, const bool paint_lines, const bool paint_arrows);

// Renders an intersection area and its traffic lights.
void render_intersection(SDL_Renderer* renderer, const Intersection* intersection);

// Renders a generic lane by delegating to the appropriate lane type renderer.
void render_lane(SDL_Renderer* renderer, const Lane* lane, const bool paint_lines, const bool paint_arrows);

// Draws a dotted line between two screen points using the specified color.
void draw_dotted_line(SDL_Renderer* renderer, const SDL_Point start, const SDL_Point end, const SDL_Color color);

// Draws a filled inward-rounded rectangle with arc outlines.
void drawFilledInwardRoundedRect(SDL_Renderer *renderer, const int x, const int y, const int width, const int height, const int radius);

// Draws a quarter-circle arc outline for decorative corners or turns.
void drawQuarterCircleOutline(SDL_Renderer *renderer, const int center_x, const int center_y, const int radius, const int quadrant, const int thickness);

// Renders a car based on its current lane and position.
void render_car(SDL_Renderer* renderer, const Car* car);

// Renders the entire simulation state: roads, cars, lanes, traffic.
void render_sim(SDL_Renderer* renderer, const Simulation* sim, const bool draw_lanes, const bool draw_cars, const bool draw_track_lines, const bool draw_traffic_lights, const bool benchmark);

// thickLineRGBA(renderer, p1.x, p1.y, p2.x, p2.y, thickness, color.r, color.g, color.b, color.a);

int thickLineRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Uint8 width, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int SDL_RenderDrawLine_ignore_if_outside_screen(SDL_Renderer * renderer,
                                               int x1, int y1, int x2, int y2);

int SDL_RenderFillRect_ignore_if_outside_screen(SDL_Renderer * renderer, const SDL_Rect * rect);

int filledTrigonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int trigonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int filledPolygonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, const Sint16 * vx, const Sint16 * vy, int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int polygonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, const Sint16 * vx, const Sint16 * vy, int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a);