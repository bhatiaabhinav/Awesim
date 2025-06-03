#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>  // for uint32_t

// --- LCG PRNG Implementation ---
#define LCG_A 1664525
#define LCG_C 1013904223
#define LCG_M ((uint32_t)0xFFFFFFFF)

static uint32_t lcg_state = 1;  // default nonzero seed

void seed_rng(uint32_t seed) {
    if (seed == 0) seed = 1;  // avoid 0 (common LCG pitfall)
    lcg_state = seed;
}

uint32_t lcg_rand() {
    lcg_state = (LCG_A * lcg_state + LCG_C);
    return lcg_state;
}

// --- Utility Functions ---

bool approxeq(double a, double b, double epsilon) {
    return fabs(a - b) < epsilon;
}

double rand_0_to_1(void) {
    return (double)lcg_rand() / (double)UINT32_MAX;
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
    return min + (lcg_rand() % range);
}

int rand_int(int max) {
    if (max <= 0) {
        fprintf(stderr, "rand_int: Error: max must be greater than 0\n");
        exit(EXIT_FAILURE);
    }
    return lcg_rand() % max;
}

double fclamp(double value, double min, double max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}