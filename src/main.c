#include "awesim.h"
#include "render.h"
#include "logging.h"
#include <stdio.h>
#include <sys/time.h>
#include <SDL2/SDL_image.h>
#ifdef _WIN32
#include <windows.h>
#endif


double get_sys_time_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)((long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec) / 1000000.0; // Convert milliseconds to seconds
}

int mouse_start_x, mouse_start_y;
bool dragging = false;

int main(int argc, char* argv[]) {

    #ifdef _WIN32
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    #endif

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        LOG_ERROR("SDL could not initialize! SDL_Error: %s", SDL_GetError());
        return -1;
    }
    LOG_DEBUG("SDL initialized successfully.");
    // Create window
    SDL_Window* window = SDL_CreateWindow("Awesim", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT, SDL_WINDOW_SHOWN);
    if (window == NULL) {
        LOG_ERROR("Window could not be created! SDL_Error: %s", SDL_GetError());
        SDL_Quit();
        return -1;
    }
    LOG_DEBUG("Window created successfully with size %dx%d.", WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT);
    // Add icon
    // SDL_Surface *icon = SDL_LoadBMP("icon.bmp");
    SDL_Surface* icon = IMG_Load("icon.png");
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
        return -1;
    }

    LOG_INFO("SDL initialized successfully. Window and renderer created.");


    // Main rendering loop
    bool quit = false;
    SDL_Event event;
    
    int num_cars = 256;                               // number of cars to simulate
    Seconds dt = 0.02;                              // time resolution for integration.
    Seconds seconds_to_simulate = 1e9;              // total time (in sim) to simulate, after which the program will exit.
    Simulation* sim = awesim(num_cars, dt);

    double t0 = get_sys_time_seconds();
    double simulation_speedup = 1;       // we will simulate simultation_speedup seconds per wall-second.
    double render_fps = 0;              // to measure render FPS
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) quit = true;
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
                if (event.wheel.y > 0) SCALE *= 1.04f;
                else SCALE /= 1.04f;
                SCALE = fclamp(SCALE, 1.0f, 50.0f);
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
                    SCALE = fclamp(SCALE, 1.0, 50.0);  // Clamp scale between 1x and 50x
                    // Update panning to keep the gesture center stable
                    PAN_X = (int)(world_x * SCALE + width / 2 - gesture_x);
                    PAN_Y = (int)(world_y * SCALE + height / 2 - gesture_y);
                }
            }
        }
        
        double _t = get_sys_time_seconds();
        render_sim(renderer, sim, true, true, false, true, false);
        SDL_RenderPresent(renderer);
        double render_time = get_sys_time_seconds() - _t; // time taken to render the simulation
        render_fps = 0.5 * render_fps + 0.5 / render_time; // exponential smoothing
        printf("render_fps: %f\r", render_fps);

        double wall_time = (get_sys_time_seconds() - t0);
        double virtual_time = simulation_speedup * wall_time;
        double delta_t = virtual_time - sim->time;  // the sim is lagging behind by these many virtual seconds
        if (virtual_time < seconds_to_simulate) {
            simulate(sim, delta_t);                     // make sim catch up to virtual time. This is will cause delta_t / dt transition updates.
        }
    }
    
    sim_deep_free(sim);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}