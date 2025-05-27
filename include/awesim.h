#pragma once

#include "sim.h"
#include <stdio.h>

Map* awesim_map(Meters city_width);

Simulation* awesim(Meters city_width, int num_cars, Seconds dt);
