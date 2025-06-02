#include "awesim.h"
#include "render.h"
#include "logging.h"
#include <stdio.h>
#include <sys/time.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>
#ifdef _WIN32
#include <windows.h>
#endif


double get_sys_time_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)((long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec) / 1000000.0; // Convert milliseconds to seconds
}

static int mouse_start_x, mouse_start_y;
static bool dragging = false;

int main(int argc, char* argv[]) {

    Meters city_width = argc >= 2 ? meters(atof(argv[1])) : meters(1000); // width of the city in meters. Prefer to read from command line argument, else use default.
    if (city_width <= 0) {
        LOG_ERROR("Invalid input. City width must be greater than 0. Exiting.");
        exit(EXIT_FAILURE);
    }
    int num_cars = argc >= 3 ? atoi(argv[2]) : 128;  // number of cars to simulate. Prefer to read from command line argument, else use default.
    if (num_cars <= 0) {
        LOG_ERROR("Invalid input. Number of cars must be greater than 0. Exiting.");
        exit(EXIT_FAILURE);
    }

    #ifdef _WIN32
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    #endif

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        LOG_ERROR("SDL could not initialize! SDL_Error: %s", SDL_GetError());
        return EXIT_FAILURE;
    }
    LOG_DEBUG("SDL initialized successfully.");
    // Create window
    SDL_Window* window = SDL_CreateWindow("Awesim", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT, SDL_WINDOW_SHOWN);
    if (window == NULL) {
        LOG_ERROR("Window could not be created! SDL_Error: %s", SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }
    LOG_DEBUG("Window created successfully with size %dx%d.", WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
    // Add icon
    SDL_Surface* icon = IMG_Load("assets/icons/icon.png");
    if (icon == NULL) {
        LOG_ERROR("Unable to load icon: %s", SDL_GetError());
    } else {
        SDL_SetWindowIcon(window, icon);
        SDL_FreeSurface(icon);
        LOG_DEBUG("Icon set successfully.");
    }
    // Create renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == NULL) {
        LOG_ERROR("Renderer could not be created! SDL_Error: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    }
    if (!init_text_rendering("assets/fonts/Roboto.ttf")) {
        LOG_ERROR("Text rendering initialization failed");
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return EXIT_FAILURE;
    } else {
        LOG_DEBUG("Text rendering initialized successfully.");
    }

    LOG_INFO("SDL initialized successfully. Window and renderer created.");


    // Main rendering loop
    bool quit = false;
    SDL_Event event;

    Seconds dt = 0.02;                              // time resolution for integration.
    Seconds seconds_to_simulate = 1e9;              // total time (in sim) to simulate, after which the program will exit.

    Simulation* sim = (Simulation*)malloc(sizeof(Simulation));
    LOG_DEBUG("Allocated memory for simulation, size %.2f kilobytes.", sizeof(*sim) / 1024.0);
    awesim_setup(sim, city_width, num_cars, dt);


    int text_font_size = 20;                // font size for stats rendering
    double simulation_speedup = 1;          // we will simulate simultation_speedup seconds per wall-second.far.
    double virtual_time = 0;                // virtual time in seconds
    double render_fps = 0;                  // to measure render FPS

    double t_prev = get_sys_time_seconds();
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
                LOG_INFO("Quit event received. Exiting.");
            } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_RIGHT) {
                simulation_speedup += simulation_speedup < 0.99 ? 0.1 : 1.0;
                LOG_TRACE("Simulation speedup increased to %f", simulation_speedup);
            } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_LEFT) {
                simulation_speedup -= simulation_speedup < 1.01 ? 0.1 : 1.0;
                simulation_speedup = fmax(simulation_speedup, 0.0);
                LOG_TRACE("Simulation speedup decreased to %f", simulation_speedup);
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
                 simulation_speedup);
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

        // Simulate:
        double delta_wall_t = get_sys_time_seconds() - t_prev;
        virtual_time += delta_wall_t * simulation_speedup;
        if (virtual_time < seconds_to_simulate) {
            simulate(sim, virtual_time - sim->time);                     // make sim catch up to virtual time. This is will cause (virtual_time - sim->time) / dt transition updates.
        }
        t_prev = get_sys_time_seconds();
    }
    
    LOG_DEBUG("Freeing simulation memory.");
    free(sim);
    LOG_INFO("Simulation cleaned up successfully.");
    LOG_DEBUG("Cleaning up SDL resources.");
    cleanup_text_rendering();
    LOG_DEBUG("Text rendering cleaned up successfully.");
    SDL_DestroyRenderer(renderer);
    LOG_DEBUG("Renderer destroyed successfully.");
    SDL_DestroyWindow(window);
    LOG_DEBUG("Window destroyed successfully.");
    SDL_Quit();
    LOG_DEBUG("SDL Quit called successfully.");

    LOG_INFO("SDL cleaned up successfully. Exiting program.");
    
    return 0;
}