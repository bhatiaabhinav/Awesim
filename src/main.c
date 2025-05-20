#include "awesim.h"
#include "render.h"
#include <stdio.h>
#include <sys/time.h>


double get_sys_time_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)((long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec) / 1000000.0; // Convert milliseconds to seconds
}

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return -1;
    }
    // Create window
    SDL_Window* window = SDL_CreateWindow("Awesim", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_SIZE_WIDTH, WINDOW_SIZE_HEIGHT, SDL_WINDOW_SHOWN);
    if (window == NULL) {
        fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }
    // Create renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == NULL) {
        fprintf(stderr, "Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }
    


    // Main rendering loop
    bool quit = false;
    SDL_Event event;
    
    int num_cars = 16;                               // number of cars to simulate
    Seconds dt = 0.02;                              // time resolution for integration.
    Seconds seconds_to_simulate = 1e9;              // total time (in sim) to simulate, after which the program will exit.
    Simulation* sim = awesim(num_cars, dt);

    double t0 = get_sys_time_seconds();
    double simulation_speedup = 2;       // we will simulate simultation_speedup seconds per wall-second.
    double render_fps = 0;              // to measure render FPS
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            }
        }
        
        double _t = get_sys_time_seconds();
        render_sim(renderer, sim, true, true, true, true, false);
        SDL_RenderPresent(renderer);
        double render_time = get_sys_time_seconds() - _t; // time taken to render the simulation
        render_fps = 0.9 * render_fps + 0.1 / render_time; // exponential smoothing
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