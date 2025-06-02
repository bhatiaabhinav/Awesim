#include "awesim.h"
#include "logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_sys_time_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)((long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec) / 1000000.0; // Convert milliseconds to seconds
}

int main(int argc, char** argv) {
    Meters city_width = meters(1000.0);             // width of the city in meters
    int num_cars = 16;                               // number of cars to simulate
    const Seconds dt = 0.02;                        // time resolution for integration.
    Simulation* sim = (Simulation*)malloc(sizeof(Simulation));
    LOG_DEBUG("Allocated memory for simulation, size %.2f kilobytes.", sizeof(*sim) / 1024.0);
    awesim_setup(sim, city_width, num_cars, dt);
    const int benchmark_n_transitions = 1000000;

    double t0 = get_sys_time_seconds();
    simulate(sim, dt * benchmark_n_transitions);
    double t1 = get_sys_time_seconds();

    double sim_tps = benchmark_n_transitions / (t1 - t0);
    printf("Transitions per second: %f\n", sim_tps);
    double sim_seconds_per_second = sim_tps * dt;
    double sim_minutes_per_second = sim_seconds_per_second / 60;
    double sim_hours_per_second = sim_minutes_per_second / 60;
    printf("With time resolution dt = %f:\n", dt);
    printf("\tSimulation seconds per second: %f\n", sim_seconds_per_second);
    printf("\tSimulation minutes per second: %f\n", sim_minutes_per_second);
    printf("\tSimulation hours per second: %f\n", sim_hours_per_second);

    free(sim);
    return 0;
}

