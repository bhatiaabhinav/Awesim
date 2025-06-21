#include "awesim.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#ifdef _WIN32
#include <windows.h>
#endif

static char* SERVER_IP = "127.0.0.1";
static int SERVER_PORT = 4242;

int main(int argc, char* argv[]) {
    Meters city_width = argc >= 2 ? meters(atof(argv[1])) : meters(1000); // width of the city in meters. Prefer to read from command line argument, else use default.
    if (city_width <= 0) {
        LOG_ERROR("Invalid input. City width must be greater than 0. Exiting.");
        exit(EXIT_FAILURE);
    }
    int num_cars = argc >= 3 ? atoi(argv[2]) : 256;  // number of cars to simulate. Prefer to read from command line argument, else use default.
    if (num_cars <= 0) {
        LOG_ERROR("Invalid input. Number of cars must be greater than 0. Exiting.");
        exit(EXIT_FAILURE);
    }
    Seconds seconds_to_simulate = argc >= 4 ? atoi(argv[3]) : 1e9; // total time (in sim) to simulate, after which the program will exit. Prefer to read from command line argument, else use default.

    Seconds dt = 0.02;                              // time resolution for integration.
    ClockReading initial_clock_reading = clock_reading(0, 8, 0, 0); // start clock at 8:00 AM on Monday.
    Weather weather = WEATHER_SUNNY;                // default weather condition.
    printf("Simulation parameters: City width = %.2f meters, Number of cars = %d, Time resolution dt = %.2f seconds, Initial clock reading = %02d:%02d:%02d, Weather = %s, Seconds to simulate = %.2f\n",
           city_width, num_cars, dt, initial_clock_reading.hours, initial_clock_reading.minutes, (int)initial_clock_reading.seconds, weather_strings[weather], seconds_to_simulate);

    Simulation* sim = sim_malloc();
    if (!sim) {
        LOG_ERROR("Failed to allocate memory for Simulation. Exiting.");
        return EXIT_FAILURE;
    }
    awesim_setup(sim, city_width, num_cars, dt, initial_clock_reading, weather);
    sim_set_synchronized(sim, true, 1.0); // Enable synchronization with wall time at 1x speedup

    // Connect to render server
    SERVER_IP = argc >= 5 ? argv[4] : SERVER_IP;            // Default to localhost if not specified
    SERVER_PORT = argc >= 6 ? atoi(argv[5]) : SERVER_PORT;  // Default port if not specified

    if (sim_connect_to_render_server(sim, SERVER_IP, SERVER_PORT)) {
        sim_set_should_quit_when_rendering_window_closed(sim, true);
        LOG_INFO("Running in rendered mode with server at %s:%d", SERVER_IP, SERVER_PORT);
    } else {
        LOG_WARN("Failed to connect to render server, running in headless mode");
    }

    simulate(sim, seconds_to_simulate); // Run the simulation for the specified duration
    printf("Simulation completed. Total simulated time: %.2f seconds.\n", sim_get_time(sim));

    if (sim->is_connected_to_render_server) {
        sim_disconnect_from_render_server(sim);
    }
    sim_free(sim);
    LOG_INFO("Simulation cleaned up");
    LOG_INFO("Exiting program.");
    return 0;
}