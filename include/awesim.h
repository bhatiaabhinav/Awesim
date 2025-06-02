#pragma once

#include "sim.h"
#include <stdio.h>

void awesim_map_setup(Map* map, Meters city_width);

void awesim_setup(Simulation* sim, Meters city_width, int num_cars, Seconds dt);
