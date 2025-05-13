#include "utils.h"
#include <stdlib.h>
#include <math.h>

bool approxeq(double a, double b, double epsilon) {
    return fabs(a - b) < epsilon;
}

double rand_0_to_1(void) {
    return (double)rand() / RAND_MAX;
}

double rand_uniform(double min, double max) {
    return min + (max - min) * rand_0_to_1();
}

int rand_int_range(int min, int max) {
    int range = max - min + 1;
    return min + rand() % range;
}

int rand_int(int max) {
    return rand() % max;
}