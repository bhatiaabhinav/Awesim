#include "awesim.h"
#include "render.h"
#include "logging.h"
#include <stdio.h>
#include <sys/time.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#include <pthread.h>
#ifdef _WIN32
#include <windows.h>
#endif

// RenderState to hold shared resources
typedef struct RenderState {
    Simulation* sim;
    SDL_Window* window;
    SDL_Renderer* renderer;
    bool quit;
} RenderState;

static RenderState render_state = {0};
static pthread_t render_thread;

// Rendering thread function
static void* render_thread_func(void* arg) {
    RenderState* state = (RenderState*)arg;
    Simulation* sim = state->sim;

    #ifdef _WIN32
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    #endif

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        LOG_ERROR("SDL could not initialize! SDL_Error: %s", SDL_GetError());
        sim->is_rendering_window_open = false;
        return NULL;
    }
    LOG_DEBUG("SDL initialized successfully.");

    // Create window
    SDL_Window* window = SDL_CreateWindow("Awesim", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        LOG_ERROR("Window could not be created! SDL_Error: %s", SDL_GetError());
        SDL_Quit();
        sim->is_rendering_window_open = false;
        return NULL;
    }
    state->window = window;
    LOG_DEBUG("Window created successfully with size %dx%d.", WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);

    // Set window icon
    SDL_Surface* icon = IMG_Load("assets/icons/icon.png");
    if (icon) {
        SDL_SetWindowIcon(window, icon);
        SDL_FreeSurface(icon);
        LOG_DEBUG("Icon set successfully.");
    } else {
        LOG_ERROR("Unable to load icon: %s", SDL_GetError());
    }

    // Create renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        LOG_ERROR("Renderer could not be created! SDL_Error: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        sim->is_rendering_window_open = false;
        return NULL;
    }
    state->renderer = renderer;
    LOG_DEBUG("Renderer created successfully.");

    if (!init_text_rendering("assets/fonts/Roboto.ttf")) {
        LOG_ERROR("Text rendering initialization failed");
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        sim->is_rendering_window_open = false;
        return NULL;
    }
    LOG_DEBUG("Text rendering initialized successfully.");

    LOG_DEBUG("SDL initialized in render thread.");

    SDL_Event event;
    int mouse_start_x, mouse_start_y;
    bool dragging = false;
    double render_fps = 0.0; // For FPS measurement
    int text_font_size = 20;                // font size for stats rendering

    while (!state->quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                state->quit = true;
                LOG_DEBUG("Quit event received. Closing render thread.");
            } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_RIGHT) {
                sim->simulation_speedup += sim->simulation_speedup < 0.99 ? 0.1 : 1.0;
                sim->simulation_speedup = fmin(sim->simulation_speedup, 100.0);
                LOG_TRACE("Simulation speedup increased to %f", sim->simulation_speedup);
            } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_LEFT) {
                sim->simulation_speedup -= sim->simulation_speedup < 1.01 ? 0.1 : 1.0;
                sim->simulation_speedup = fmax(sim->simulation_speedup, 0.0);
                LOG_TRACE("Simulation speedup decreased to %f", sim->simulation_speedup);
            }
            else if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT) {
                mouse_start_x = event.button.x;
                mouse_start_y = event.button.y;
                dragging = true;
            }
            else if (event.type == SDL_MOUSEBUTTONUP && event.button.button == SDL_BUTTON_LEFT) {
                dragging = false;
            }
            else if (event.type == SDL_MOUSEMOTION && dragging) {
                int dx = event.motion.x - mouse_start_x;
                int dy = event.motion.y - mouse_start_y;
                PAN_X -= dx;
                PAN_Y -= dy;
                mouse_start_x = event.motion.x;
                mouse_start_y = event.motion.y;
                LOG_TRACE("Mouse dragged. New PAN_X: %d, PAN_Y: %d", PAN_X, PAN_Y);
            }
            else if (event.type == SDL_MOUSEWHEEL && event.wheel.y != 0) {
                double width = WINDOW_SIZE_WIDTH;
                double height = WINDOW_SIZE_HEIGHT;
                int mx, my;
                SDL_GetMouseState(&mx, &my);
                // find x under mouse in world coordinates
                double x = (mx + PAN_X - width / 2) / SCALE;
                double y = (my + PAN_Y - height / 2) / SCALE;
                // find new scale
                if (event.wheel.y > 0) SCALE *= 1.1f;
                else SCALE /= 1.1f;
                SCALE = fclamp(SCALE, 1.0f, 32.0f);
                LOG_TRACE("Mouse wheel scrolled. New scale: %f", SCALE);
                // Adjust pan to keep the mouse under the same world coordinates
                PAN_X = (int)(x * SCALE + width / 2 - mx);
                PAN_Y = (int)(y * SCALE + height / 2 - my);
            }
            else if (event.type == SDL_MULTIGESTURE) {
                SDL_TouchID touch_id = event.mgesture.touchId;
                int num_fingers = SDL_GetNumTouchFingers(touch_id);

                if (num_fingers >= 2) {  // Ensure at least two fingers for pinch-to-zoom
                    double sum_x = 0.0, sum_y = 0.0;
                    double width = WINDOW_SIZE_WIDTH;
                    double height = WINDOW_SIZE_HEIGHT;
                    // Sum the positions of all fingers
                    for (int i = 0; i < num_fingers; i++) {
                        SDL_Finger* finger = SDL_GetTouchFinger(touch_id, i);
                        if (finger) {
                            sum_x += finger->x * width;  // Convert normalized to pixel coordinates
                            sum_y += finger->y * height;
                        }
                    }
                    // Calculate the mean gesture center in pixel coordinates
                    double gesture_x = sum_x / num_fingers;
                    double gesture_y = sum_y / num_fingers;
                    // Convert gesture center to world coordinates before scaling
                    double world_x = (gesture_x + PAN_X - width / 2) / SCALE;
                    double world_y = (gesture_y + PAN_Y - height / 2) / SCALE;
                    // Apply zoom based on distance change
                    double sensitivity = 2.0;  // Adjusted lower for finesse
                    double scale_factor = 1.0 + event.mgesture.dDist * sensitivity;
                    SCALE *= scale_factor;
                    SCALE = fclamp(SCALE, 1.0, 32.0);  // Clamp scale between 1x and 32x
                    LOG_TRACE("Pinch gesture detected. New scale: %f", SCALE);
                    // Update panning to keep the gesture center stable
                    PAN_X = (int)(world_x * SCALE + width / 2 - gesture_x);
                    PAN_Y = (int)(world_y * SCALE + height / 2 - gesture_y);
                }
            }
        }

        // Measure render start time
        double _t = get_sys_time_seconds();

        // Render simulation
        bool draw_lanes = true; // Draw lanes
        bool draw_cars = true; // Draw cars
        bool draw_track_lines = false; // Draw track lines
        bool draw_traffic_lights = true; // Draw traffic lights
        bool draw_car_ids = true; // Draw car IDs
        bool draw_lane_ids = true; // Draw lane IDs
        bool draw_road_names = true; // Draw road names
        bool benchmark = false; // Benchmark mode
        render_sim(renderer, sim, draw_lanes, draw_cars, draw_track_lines, draw_traffic_lights, draw_car_ids, draw_lane_ids, draw_road_names, benchmark);

        // Render time stats
        char time_stats[40];
        ClockReading clock_reading = sim_get_clock_reading(sim); 
        snprintf(time_stats, sizeof(time_stats), "%s %02d:%02d:%02d  (>> %.1fx)", 
                 day_of_week_strings[clock_reading.day_of_week], 
                 clock_reading.hours, 
                 clock_reading.minutes, 
                 (int)clock_reading.seconds, 
                 sim->simulation_speedup);
        render_text(renderer, time_stats, WINDOW_SIZE_WIDTH - 10, 10, 255, 255, 255, 255, text_font_size, ALIGN_TOP_RIGHT, false, NULL); // Render time stats at the  top right corner
        
        
        // Render weather stats just below time stats
        render_text(renderer, weather_strings[sim->weather], WINDOW_SIZE_WIDTH - 10, 20 + text_font_size, 255, 255, 255, 255, text_font_size, ALIGN_TOP_RIGHT, false, NULL); // Render weather stats at the top right corner below time stats

        // Render FPS stats        
        char fps_stats[24];
        snprintf(fps_stats, sizeof(fps_stats), "FPS: %d   (VSync On)", (int)render_fps);
        render_text(renderer, fps_stats, 10, 10, 255, 255, 255, 255, text_font_size, ALIGN_TOP_LEFT, false, NULL); // Render FPS at the top left corner
        SDL_RenderPresent(renderer);

        // Update FPS calculation
        double render_time = get_sys_time_seconds() - _t; // time taken to render the simulation
        render_fps = 0.9 * render_fps + 0.1 / render_time; // exponential smoothing
    }

    // Cleanup
    LOG_DEBUG("Cleaning up render thread resources.");
    cleanup_text_rendering();
    LOG_DEBUG("Text rendering cleaned up.");
    SDL_DestroyRenderer(renderer);
    LOG_DEBUG("Renderer destroyed.");
    SDL_DestroyWindow(window);
    LOG_DEBUG("Window destroyed.");
    SDL_Quit();
    LOG_DEBUG("SDL cleaned up.");
    state->sim->is_rendering_window_open = false; // Mark the rendering window as closed
    state->sim = NULL; // Clear the simulation pointer
    LOG_INFO("Render thread closed.");
    return NULL;
}

// Start rendering thread
int sim_open_rendering_window(Simulation* sim) {
    if (render_state.sim != NULL) {
        LOG_ERROR("Rendering already started");
        return -1;
    }
    render_state.sim = sim;
    render_state.quit = false;
    if (pthread_create(&render_thread, NULL, render_thread_func, &render_state) != 0) {
        LOG_ERROR("Failed to create render thread");
        return -1;
    }
    sim->is_rendering_window_open = true; // Mark the rendering window as open
    LOG_INFO("Render thread started.");
    return 0;
}

// Stop rendering thread
void sim_close_rendering_window() {
    if (render_state.sim == NULL) {
        return;
    }
    render_state.quit = true;
    pthread_join(render_thread, NULL);
}
