#include "render.h"

int WINDOW_SIZE_WIDTH = 1000;
int WINDOW_SIZE_HEIGHT = 1000;
bool ENABLE_DOUBLE_CLICK_TO_TOGGLE_FULLSCREEN = true;

bool DRAW_LANES = true;                // Draw lanes
bool DRAW_CARS = true;                 // Draw cars
bool DRAW_TRACK_LINES = false;         // Draw track lines (center lines on lanes)
bool DRAW_TRAFFIC_LIGHTS = true;       // Draw traffic lights
bool DRAW_CAR_IDS = true;              // Draw car IDs
bool DRAW_LANE_IDS = true;             // Draw lane IDs
bool DRAW_ROAD_NAMES = true;           // Draw road names
int HUD_FONT_SIZE = 20;                // Font size for HUD text

// Keep the screen centered on the following car ID:
int CAMERA_CENTERED_ON_CAR_ID = 0;              // Car 0 is "agent" car
bool CAMERA_CENTERED_ON_CAR_ENABLED = false;    // can be toggled with 'f' key

const CarId HIGHLIGHTED_CARS[] = {
    0,      // Highlight the agent car by default
    ID_NULL // The last element MUST be ID_NULL to mark end of array.
};

const LaneId HIGHLIGHTED_LANES[] = {
    ID_NULL // The last element MUST be ID_NULL to mark end of array.
};

SDL_Color HIGHLIGHTED_CAR_COLOR = {0, 0, 255, 255}; // Blue color for highlighted cars
SDL_Color HIGHLIGHTED_LANE_COLOR = {192, 192, 192, 255}; // Light gray color for highlighted lanes

const char* FONT_PATH = "assets/fonts/FreeSans.ttf"; // Default font path