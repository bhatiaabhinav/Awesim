#pragma once

#include "sim.h"
#include <stdio.h>

Map* awesim_map();

Simulation* awesim(int num_cars, Seconds dt);
