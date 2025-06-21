#include "render.h"
#include "logging.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#endif

SDL_Renderer* renderer = NULL;
SDL_Window* window = NULL;
int PAN_X = 0;
int PAN_Y = 0;
double SCALE = 4.0;
int fonts_initialized = 0;
TTF_Font* font_cache[MAX_FONT_SIZE] = {NULL};


SDL_Point to_screen_coords(const Coordinates point, const int width, const int height) {
    double x_scaled = point.x * SCALE;
    double y_scaled = point.y * SCALE;
    int x_screen = (int)(width / 2 + x_scaled) - PAN_X;
    int y_screen = (int)(height / 2 - y_scaled) - PAN_Y;
    return (SDL_Point){x_screen, y_screen};
}


// Initialize SDL_ttf and load fonts (call once at program startup)
static int init_text_rendering(const char* font_path) {
    if (TTF_Init() == -1) {
        LOG_ERROR("TTF_Init failed: %s", TTF_GetError());
        return 0;
    }

    // Load fonts for sizes 1 to 32
    for (int i = 0; i < MAX_FONT_SIZE; i++) {
        font_cache[i] = TTF_OpenFont(font_path, i + 1);
        if (!font_cache[i]) {
            LOG_ERROR("TTF_OpenFont failed for size %d: %s", i + 1, TTF_GetError());
            // Clean up any loaded fonts
            for (int j = 0; j < i; j++) {
                TTF_CloseFont(font_cache[j]);
                font_cache[j] = NULL;
            }
            TTF_Quit();
            return 0;
        }
    }
    fonts_initialized = 1;
    return 1;
}

// Clean up SDL_ttf and fonts (call at program shutdown)
static void cleanup_text_rendering() {
    for (int i = 0; i < MAX_FONT_SIZE; i++) {
        if (font_cache[i]) {
            TTF_CloseFont(font_cache[i]);
            font_cache[i] = NULL;
        }
    }
    fonts_initialized = 0;
    LOG_DEBUG("Fonts cleaned up successfully.");
    for (int i = 0; i < MAX_NUM_ROADS; i++) {
        for (int j = 0; j < MAX_FONT_SIZE; j++) {
            if (road_name_texture_cache[i][j]) {
                SDL_DestroyTexture(road_name_texture_cache[i][j]);
                road_name_texture_cache[i][j] = NULL;
            }
        }
    }
    LOG_DEBUG("Road name textures cleaned up successfully.");
    for (int i = 0; i < MAX_NUM_INTERSECTIONS; i++) {
        for (int j = 0; j < MAX_FONT_SIZE; j++) {
            if (intersection_name_texture_cache[i][j]) {
                SDL_DestroyTexture(intersection_name_texture_cache[i][j]);
                intersection_name_texture_cache[i][j] = NULL;
            }
        }
    }
    LOG_DEBUG("Intersection name textures cleaned up successfully.");
    for (int i = 0; i < MAX_NUM_LANES; i++) {
        for (int j = 0; j < MAX_FONT_SIZE; j++) {
            if (lane_id_texture_cache[i][j]) {
                SDL_DestroyTexture(lane_id_texture_cache[i][j]);
                lane_id_texture_cache[i][j] = NULL;
            }
        }
    }
    LOG_DEBUG("Lane ID textures cleaned up successfully.");
    for (int i = 0; i < MAX_CARS_IN_SIMULATION; i++) {
        for (int j = 0; j < MAX_FONT_SIZE; j++) {
            if (car_id_texture_cache[i][j]) {
                SDL_DestroyTexture(car_id_texture_cache[i][j]);
                car_id_texture_cache[i][j] = NULL;
            }
        }
    }
    LOG_DEBUG("Car ID textures cleaned up successfully.");
    TTF_Quit();
}

// Initialize SDL, window, renderer, and text rendering
bool init_sdl() {
    #ifdef _WIN32
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    #endif

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        LOG_ERROR("SDL could not initialize! SDL_Error: %s", SDL_GetError());
        return false;
    }
    LOG_DEBUG("SDL initialized successfully");

    window = SDL_CreateWindow("Awesim", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                              WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        LOG_ERROR("Window could not be created! SDL_Error: %s", SDL_GetError());
        SDL_Quit();
        return false;
    }
    LOG_DEBUG("Window created successfully with size %dx%d", WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);

    SDL_Surface* icon = IMG_Load("assets/icons/icon.png");
    if (icon) {
        SDL_SetWindowIcon(window, icon);
        SDL_FreeSurface(icon);
        LOG_DEBUG("Icon set successfully");
    } else {
        LOG_ERROR("Unable to load icon: %s", SDL_GetError());
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        LOG_ERROR("Renderer could not be created! SDL_Error: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return false;
    }
    LOG_DEBUG("Renderer created successfully");

    if (!init_text_rendering("assets/fonts/Roboto.ttf")) {
        LOG_ERROR("Text rendering initialization failed");
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return false;
    }
    LOG_DEBUG("Text rendering initialized successfully");

    return true;
}

// Cleanup resources
void cleanup_sdl() {
    LOG_DEBUG("Cleaning up SDL");
    cleanup_text_rendering();
    LOG_DEBUG("Text rendering cleaned up");
    SDL_DestroyRenderer(renderer);
    LOG_DEBUG("Renderer destroyed");
    SDL_DestroyWindow(window);
    LOG_DEBUG("Window destroyed");
    SDL_Quit();
    LOG_DEBUG("SDL cleaned up");
}

static bool dragging = false;
static int mouse_start_x = 0, mouse_start_y = 0;

// Handle SDL events
SimCommand handle_sdl_events() {
    SimCommand command = COMMAND_NONE;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            command = COMMAND_QUIT; // Quit command
            LOG_DEBUG("Quit event received");
        } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_RIGHT) {
            command = COMMAND_SIM_INCREASE_SPEED; // Speed up command
            LOG_DEBUG("Right arrow key pressed, increasing speed");
        } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_LEFT) {
            command = COMMAND_SIM_DECREASE_SPEED; // Slow down command
            LOG_DEBUG("Left arrow key pressed, decreasing speed");
        } else if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
            mouse_start_x = event.button.x;
            mouse_start_y = event.button.y;
            dragging = true;
        } else if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
            dragging = false;
        } else if (event.type == SDL_MOUSEMOTION && dragging) {
            int dx = event.motion.x - mouse_start_x;
            int dy = event.motion.y - mouse_start_y;
            PAN_X -= dx;
            PAN_Y -= dy;
            mouse_start_x = event.motion.x;
            mouse_start_y = event.motion.y;
            LOG_TRACE("Mouse dragged. New PAN_X: %d, PAN_Y: %d", PAN_X, PAN_Y);
        } else if (event.type == SDL_MOUSEWHEEL && event.wheel.y != 0) {
            double width = WINDOW_SIZE_WIDTH;
            double height = WINDOW_SIZE_HEIGHT;
            int mx, my;
            SDL_GetMouseState(&mx, &my);
            double x = (mx + PAN_X - width / 2) / SCALE;
            double y = (my + PAN_Y - height / 2) / SCALE;
            if (event.wheel.y > 0) SCALE *= 1.1f;
            else SCALE /= 1.1f;
            SCALE = fclamp(SCALE, 1.0f, 32.0f);
            LOG_TRACE("Mouse wheel scrolled. New scale: %f", SCALE);
            PAN_X = (int)(x * SCALE + width / 2 - mx);
            PAN_Y = (int)(y * SCALE + height / 2 - my);
        } else if (event.type == SDL_MULTIGESTURE) {
            SDL_TouchID touch_id = event.mgesture.touchId;
            int num_fingers = SDL_GetNumTouchFingers(touch_id);
            if (num_fingers >= 2) {
                double sum_x = 0.0, sum_y = 0.0;
                double width = WINDOW_SIZE_WIDTH;
                double height = WINDOW_SIZE_HEIGHT;
                for (int i = 0; i < num_fingers; i++) {
                    SDL_Finger* finger = SDL_GetTouchFinger(touch_id, i);
                    if (finger) {
                        sum_x += finger->x * width;
                        sum_y += finger->y * height;
                    }
                }
                double gesture_x = sum_x / num_fingers;
                double gesture_y = sum_y / num_fingers;
                double world_x = (gesture_x + PAN_X - width / 2) / SCALE;
                double world_y = (gesture_y + PAN_Y - height / 2) / SCALE;
                double sensitivity = 2.0;
                double scale_factor = 1.0 + event.mgesture.dDist * sensitivity;
                SCALE *= scale_factor;
                SCALE = fclamp(SCALE, 1.0, 32.0);
                LOG_TRACE("Pinch gesture detected. New scale: %f", SCALE);
                PAN_X = (int)(world_x * SCALE + width / 2 - gesture_x);
                PAN_Y = (int)(world_y * SCALE + height / 2 - gesture_y);
            }
        }
    }
    return command;
}
