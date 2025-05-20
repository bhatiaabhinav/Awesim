#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
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
    if (range <= 0) {
        fprintf(stderr, "rand_int_range: Error: max must be greater than min\n");
        exit(EXIT_FAILURE);
    }
    return min + rand() % range;
}

int rand_int(int max) {
    if (max <= 0) {
        fprintf(stderr, "rand_int: Error: max must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
    return rand() % max;
}

double fclamp(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}