#pragma once

#include "sim.h"
#include <stdio.h>

void awesim_map_setup(Map* map, Meters city_width);
void awesim_map_setup_old(Map* map, Meters city_width);

// agent_start_lane_id_options, if NULL will lead to the agent being spawned in a random lane. If non-NULL, it should point to an array of lane IDs, and the agent will be spawned in one of those lanes (randomly selected if more than one). The array should be terminated with ID_NULL to indicate the end of options.
void awesim_setup(Simulation* sim, Meters city_width, int num_cars, Seconds dt, ClockReading initial_clock_reading, Weather weather, bool use_deprecated_map, LaneId* agent_start_lane_id_options);

