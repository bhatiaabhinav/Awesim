#pragma once

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "map.h"
#include "car.h"
#include "sim.h"

// Settings
extern int WINDOW_SIZE_WIDTH;
extern int WINDOW_SIZE_HEIGHT;
extern bool ENABLE_DOUBLE_CLICK_TO_TOGGLE_FULLSCREEN;
extern bool VSYNC_ENABLED;
extern bool HW_RENDERING_ENABLED;
extern bool DRAW_LANES;
extern bool DRAW_CARS;
extern bool DRAW_TRACK_LINES;
extern bool DRAW_TRAFFIC_LIGHTS;
extern bool DRAW_CAR_IDS;
extern bool DRAW_CAR_SPEEDS;
extern bool DRAW_LANE_IDS;
extern bool DRAW_ROAD_NAMES;
extern int HUD_FONT_SIZE;
extern CarId CAMERA_CENTERED_ON_CAR_ID;
extern bool CAMERA_CENTERED_ON_CAR_ENABLED;
extern const CarId HIGHLIGHTED_CARS[];
extern const LaneId HIGHLIGHTED_LANES[];
extern SDL_Color HIGHLIGHTED_CAR_COLOR; // Color for highlighted cars
extern SDL_Color HIGHLIGHTED_NEARBY_VEHICLES_COLOR; // Color for nearby vehicles of the first highlighted car
extern SDL_Color HIGHLIGHTED_FORWARD_VEHICLE_COLOR_AEB_ENGAGED; // Color for forward vehicle when AEB is engaged
extern SDL_Color HIGHLIGHTED_LANE_COLOR; // Color for highlighted lanes
extern const char* FONT_PATH;

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
#define ROAD_COLOR (SDL_Color){128, 128, 128, 255} // gray
#define LANE_CENTER_LINE_COLOR (SDL_Color){100, 100, 255, 255}  // blue
#define RED_LIGHT_COLOR (SDL_Color){200, 30, 30, 255}       // Soft red
#define GREEN_LIGHT_COLOR (SDL_Color){50, 200, 50, 255}     // Soft green
#define YELLOW_LIGHT_COLOR (SDL_Color){220, 180, 20, 255}   // Warm, soft yellow
#define FOUR_WAY_STOP_COLOR (SDL_Color){200, 30, 30, 64}    // Soft red with transparency
#define ARC_NUM_POINTS 8   // Number of points to approximate quarter arcs
#define MAX_FONT_SIZE 128

extern SDL_Renderer* renderer;
extern SDL_Window* window;
extern double SCALE;
extern int PAN_X;
extern int PAN_Y;
extern int fonts_initialized;
extern TTF_Font* font_cache[MAX_FONT_SIZE];
extern SDL_Texture* road_name_texture_cache[MAX_NUM_ROADS][MAX_FONT_SIZE];
extern SDL_Texture* intersection_name_texture_cache[MAX_NUM_INTERSECTIONS][MAX_FONT_SIZE];
extern SDL_Texture* lane_id_texture_cache[MAX_NUM_LANES][MAX_FONT_SIZE];
extern SDL_Texture* car_id_texture_cache[MAX_CARS_IN_SIMULATION][MAX_FONT_SIZE];
extern SDL_Texture* car_speed_texture_cache[300][MAX_FONT_SIZE];
extern NearbyVehiclesFlattened HIGHLIGHTED_NEARBY_VEHICLES;
extern bool HIGHLIGHTED_CAR_AEB_ENGAGED;

bool init_sdl();
void cleanup_sdl();
SimCommand handle_sdl_events();
void render(Simulation* sim);

// Converts world coordinates to screen coordinates relative to screen center.
SDL_Point to_screen_coords(const Coordinates point, const int width, const int height);

// Draws a lane's center line based on its geometry.
void render_lane_center_line(SDL_Renderer* renderer, const Lane* lane, Map* map, const SDL_Color color, bool dotted);

// Renders a linear lane, optionally painting lane lines and arrows.
void render_lane_linear(SDL_Renderer* renderer, const Lane* lane, Map* map, const bool paint_lines, const bool paint_arrows, const bool paint_id);

// Renders a quarter arc lane, optionally painting lane lines and arrows.
void render_lane_quarterarc(SDL_Renderer* renderer, const Lane* lane, Map* map, const bool paint_lines, const bool paint_arrows, const bool paint_id);

// Renders an intersection area and its traffic lights.
void render_intersection(SDL_Renderer* renderer, const Intersection* intersection, Map* map);

// Renders a generic lane by delegating to the appropriate lane type renderer.
void render_lane(SDL_Renderer* renderer, const Lane* lane, Map* map, const bool paint_lines, const bool paint_arrows, const bool paint_id);

// Draws a dotted line between two screen points using the specified color.
void draw_dotted_line(SDL_Renderer* renderer, const SDL_Point start, const SDL_Point end, const SDL_Color color);

// Draws a filled inward-rounded rectangle with arc outlines.
void drawFilledInwardRoundedRect(SDL_Renderer *renderer, const int x, const int y, const int width, const int height, const int radius);

// Draws a quarter-circle arc outline for decorative corners or turns.
void drawQuarterCircleOutline(SDL_Renderer *renderer, const int center_x, const int center_y, const int radius, const int quadrant, const int thickness);

// Renders a car based on its current lane and position.
void render_car(SDL_Renderer* renderer, const Car* car, Map* map, const bool paint_id, const bool paint_speed);

// Renders a lidar
void render_lidar(SDL_Renderer* renderer, const Lidar* lidar);

// Renders a camera
void render_camera(SDL_Renderer* renderer, const RGBCamera* camera);

// Renders the entire simulation state: roads, cars, lanes, traffic.
void render_sim(SDL_Renderer* renderer, Simulation* sim, const bool draw_lanes, const bool draw_cars, const bool draw_track_lines, const bool draw_traffic_lights, const bool draw_car_ids, const bool draw_car_speeds, const bool draw_lane_ids, const bool draw_road_names, int hud_font_size, const bool benchmark);

// thickLineRGBA(renderer, p1.x, p1.y, p2.x, p2.y, thickness, color.r, color.g, color.b, color.a);

int thickLineRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Uint8 width, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int SDL_RenderDrawLine_ignore_if_outside_screen(SDL_Renderer * renderer,
                                               int x1, int y1, int x2, int y2);

int SDL_RenderFillRect_ignore_if_outside_screen(SDL_Renderer * renderer, const SDL_Rect * rect);

int filledTrigonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int trigonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, Sint16 x1, Sint16 y1, Sint16 x2, Sint16 y2, Sint16 x3, Sint16 y3, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int filledPolygonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, const Sint16 * vx, const Sint16 * vy, int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

int polygonRGBA_ignore_if_outside_screen(SDL_Renderer * renderer, const Sint16 * vx, const Sint16 * vy, int n, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

// Text alignment options
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

void render_straight_road_name(SDL_Renderer* renderer, const Road* road, Map* map);
void render_intersection_name(SDL_Renderer* renderer, const Intersection* intersection, Map* map);