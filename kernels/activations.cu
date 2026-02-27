#include <math.h>

extern "C" __global__ void relu_forward(const float* x, float* y, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
  }
}

extern "C" __global__ void sigmoid_forward(const float* x, float* y, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = 1.0f / (1.0f + expf(-x[idx]));
  }
}

extern "C" __global__ void tanh_forward(const float* x, float* y, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    y[idx] = tanhf(x[idx]);
  }
}
