#ifndef RL_UTILS_H
#define RL_UTILS_H

#include "tensor.h"

// ===========================================================================
// Observation normaliser  (Welford parallel/batch update)
// ===========================================================================

typedef struct {
    int    dim;        // observation dimensionality
    float* mean;       // [dim]  running mean
    float* var;        // [dim]  running population variance
    float  count;      // total sample count
    // pre-allocated scratch buffers (avoid repeated malloc/free)
    float* _buf_mean;  // [dim]
    float* _buf_var;   // [dim]
    float* _shifts;    // [dim]
    float* _scales;    // [dim]
} ObsNormalizer;

ObsNormalizer obs_normalizer_make(int dim);
void          obs_normalizer_free(ObsNormalizer* on);
void          obs_normalizer_update(ObsNormalizer* on, const float* obs, int batch);
void          obs_normalizer_apply(const ObsNormalizer* on, float* obs, int batch);

// ===========================================================================
// Fisher–Yates shuffle
// ===========================================================================

void shuffle(int* arr, int n);

#endif // RL_UTILS_H
