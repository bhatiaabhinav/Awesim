#ifndef NN_H
#define NN_H

#include "tensor.h"

// ===========================================================================
// Layer types (extensible — add new entries for future layer kinds)
// ===========================================================================

typedef enum {
    LAYER_LINEAR,
    LAYER_RELU,
} LayerType;

// ===========================================================================
// Layer base & vtable
// ===========================================================================

typedef struct Layer Layer;

typedef struct {
    void (*forward)(Layer* self, const float* input, float* output, int batch_size);
    void (*backward)(Layer* self, const float* d_output, float* d_input, int batch_size);
    int  (*in_dim)(Layer* self);
    int  (*out_dim)(Layer* self);
    int  (*num_params)(Layer* self);
    void (*fill_params)(Layer* self, Tensor** params, Tensor** grads);
    void (*zero_grad)(Layer* self);
    void (*free_fn)(Layer* self);
} LayerVTable;

struct Layer {
    LayerType          type;
    const LayerVTable* vtable;
    bool               cache_io;          // set by SequentialNN's backprop flag
    float*             cached_input;      // [cached_batch_cap * in_dim]
    int                cached_batch_cap;
};

// ---- Dispatch helpers ----

static inline void layer_forward(Layer* l, const float* in, float* out, int bs) {
    l->vtable->forward(l, in, out, bs);
}
static inline void layer_backward(Layer* l, const float* d_out, float* d_in, int bs) {
    l->vtable->backward(l, d_out, d_in, bs);
}
static inline int  layer_in_dim(Layer* l)    { return l->vtable->in_dim(l); }
static inline int  layer_out_dim(Layer* l)   { return l->vtable->out_dim(l); }
static inline void layer_zero_grad(Layer* l) { l->vtable->zero_grad(l); }
static inline void layer_free(Layer* l)      { if (l) l->vtable->free_fn(l); }

// ===========================================================================
// Linear layer  (output = input @ W + b)
// ===========================================================================

typedef struct {
    Layer   base;
    int     in_features;
    int     out_features;
    Tensor* W;        // [in_features x out_features]
    Tensor* b;        // [out_features]
    Tensor* grad_W;   // [in_features x out_features]
    Tensor* grad_b;   // [out_features]
} LinearLayer;

Layer* linear_layer_malloc(int in_features, int out_features);

// ===========================================================================
// ReLU layer  (output = max(0, input))
// ===========================================================================

typedef struct {
    Layer base;
    int   dim;
} ReLULayer;

Layer* relu_layer_malloc(int dim);

// ===========================================================================
// Sequential neural network
// ===========================================================================

typedef struct SequentialNN SequentialNN;

struct SequentialNN {
    int      num_layers;
    Layer**  layers;          // owned array of Layer*
    bool     backprop;        // when true, layers cache IO for backward

    // Flat parameter / gradient views (shallow pointers into layers' Tensors)
    int      num_param_tensors;
    Tensor** param_views;
    Tensor** grad_views;
};

SequentialNN* seq_nn_malloc(int num_layers, Layer** layers, bool backprop);
void          seq_nn_forward(SequentialNN* nn, const float* input,
                             float* output, int batch_size);
void          seq_nn_backward(SequentialNN* nn, const float* d_output,
                              float* d_input, int batch_size);
void          seq_nn_zero_grad(SequentialNN* nn);
void          seq_nn_copy_params(SequentialNN* dst, SequentialNN* src);
void          seq_nn_set_backprop(SequentialNN* nn, bool backprop);
SequentialNN* seq_nn_clone(SequentialNN* src, bool backprop);
void          seq_nn_free(SequentialNN* nn);

// Convenience: build MLP — Linear, ReLU, Linear, ReLU, ..., Linear
// sizes = [input_dim, h1, ..., output_dim], num_sizes entries.
SequentialNN* seq_nn_make_mlp(int num_sizes, const int* sizes, bool backprop);

// Clip gradient tensors by global L2 norm
void grad_clip_norm(Tensor** grads, int n, float max_norm);

// ===========================================================================
// Adam optimizer
// ===========================================================================

typedef struct {
    int      num_params;
    Tensor** m;   // first moment estimates  (array of Tensor*)
    Tensor** v;   // second moment estimates  (array of Tensor*)
    int      t;   // time step
    float    lr;
    float    beta1;
    float    beta2;
    float    eps;
} AdamState;

AdamState* adam_malloc(Tensor** params, int num_params,
                       float lr, float beta1, float beta2, float eps);
void       adam_free(AdamState* adam);
void       adam_step(AdamState* adam, Tensor** params, Tensor** grads);

// Convenience wrappers for SequentialNN
static inline AdamState* adam_malloc_seq(SequentialNN* nn,
                                         float lr, float b1, float b2, float eps) {
    return adam_malloc(nn->param_views, nn->num_param_tensors, lr, b1, b2, eps);
}
static inline void adam_step_seq(AdamState* a, SequentialNN* nn) {
    adam_step(a, nn->param_views, nn->grad_views);
}

// Lightweight Adam for a single scalar (e.g. dual variables in log space)
// Stack-allocatable — no heap allocation needed.
typedef struct {
    float m;      // first moment estimate
    float v;      // second moment estimate
    int   t;      // time step
    float lr;
    float beta1;
    float beta2;
    float eps;
} AdamScalar;

static inline AdamScalar adam_scalar_init(float lr, float beta1, float beta2, float eps) {
    return (AdamScalar){.m = 0, .v = 0, .t = 0, .lr = lr, .beta1 = beta1, .beta2 = beta2, .eps = eps};
}

// Performs one Adam step: param -= lr * m_hat / (sqrt(v_hat) + eps)
// grad is the gradient (descent direction, i.e. derivative of loss w.r.t. param)
static inline void adam_scalar_step(AdamScalar* a, float* param, float grad) {
    a->t++;
    a->m = a->beta1 * a->m + (1 - a->beta1) * grad;
    a->v = a->beta2 * a->v + (1 - a->beta2) * grad * grad;
    float m_hat = a->m / (1 - powf(a->beta1, a->t));
    float v_hat = a->v / (1 - powf(a->beta2, a->t));
    *param -= a->lr * m_hat / (sqrtf(v_hat) + a->eps);
}

#endif // NN_H
