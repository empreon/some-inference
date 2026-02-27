#include "common.cuh"
#include <cstdint>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

#if (TILE_SIZE % 4) != 0
#error "TILE_SIZE must be divisible by 4 for matmul_vectorized."
#endif

__device__ __forceinline__ float4 load_float4_with_bounds(
    const float* row_ptr,
    int col_start,
    int row_width) {
  float4 out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  if (col_start + 3 < row_width) {
    const float* ptr = row_ptr + col_start;
    if ((reinterpret_cast<std::uintptr_t>(ptr) & 0xF) == 0) {
      return *reinterpret_cast<const float4*>(ptr);
    }
    out.x = ptr[0];
    out.y = ptr[1];
    out.z = ptr[2];
    out.w = ptr[3];
    return out;
  }

  if (col_start < row_width) out.x = row_ptr[col_start];
  if (col_start + 1 < row_width) out.y = row_ptr[col_start + 1];
  if (col_start + 2 < row_width) out.z = row_ptr[col_start + 2];
  if (col_start + 3 < row_width) out.w = row_ptr[col_start + 3];
  return out;
}

extern "C" __global__ void matmul_naive(const float* A,
                                        const float* B,
                                        float* C,
                                        int M,
                                        int K,
                                        int N) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float value = 0.0f;
    for (int idx = 0; idx < K; ++idx) {
      value += A[ROW_MAJOR_INDEX(row, idx, K)] *
               B[ROW_MAJOR_INDEX(idx, col, N)];
    }
    C[ROW_MAJOR_INDEX(row, col, N)] = value;
  }
}

extern "C" __global__ void matmul_tiled(const float* A,
                                        const float* B,
                                        float* C,
                                        int M,
                                        int K,
                                        int N) {
  __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x;
  const int row = blockIdx.y * TILE_SIZE + local_row;
  const int col = blockIdx.x * TILE_SIZE + local_col;

  float value = 0.0f;
  const int tile_count = CEIL_DIV(K, TILE_SIZE);

  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int k_a = tile_idx * TILE_SIZE + local_col;
    const int k_b = tile_idx * TILE_SIZE + local_row;

    tile_a[local_row][local_col] =
        (row < M && k_a < K) ? A[ROW_MAJOR_INDEX(row, k_a, K)] : 0.0f;
    tile_b[local_row][local_col] =
        (k_b < K && col < N) ? B[ROW_MAJOR_INDEX(k_b, col, N)] : 0.0f;

    __syncthreads();

    for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
      value += tile_a[local_row][k_local] * tile_b[k_local][local_col];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[ROW_MAJOR_INDEX(row, col, N)] = value;
  }
}

extern "C" __global__ void matmul_vectorized(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int M,
                                             int K,
                                             int N) {
  constexpr int kVecWidth = 4;
  __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

  const int local_row = threadIdx.y;
  const int local_col_vec = threadIdx.x;
  const int row = blockIdx.y * TILE_SIZE + local_row;
  const int col_base = blockIdx.x * TILE_SIZE + local_col_vec * kVecWidth;

  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int tile_count = CEIL_DIV(K, TILE_SIZE);

  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int k_col_start = tile_idx * TILE_SIZE + local_col_vec * kVecWidth;

    float4 a_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (row < M) {
      const float* a_row = A + ROW_MAJOR_INDEX(row, 0, K);
      a_vec = load_float4_with_bounds(a_row, k_col_start, K);
    }
    tile_a[local_row][local_col_vec * kVecWidth + 0] = a_vec.x;
    tile_a[local_row][local_col_vec * kVecWidth + 1] = a_vec.y;
    tile_a[local_row][local_col_vec * kVecWidth + 2] = a_vec.z;
    tile_a[local_row][local_col_vec * kVecWidth + 3] = a_vec.w;

    const int k_b = tile_idx * TILE_SIZE + local_row;
    float4 b_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (k_b < K) {
      const float* b_row = B + ROW_MAJOR_INDEX(k_b, 0, N);
      b_vec = load_float4_with_bounds(b_row, col_base, N);
    }
    tile_b[local_row][local_col_vec * kVecWidth + 0] = b_vec.x;
    tile_b[local_row][local_col_vec * kVecWidth + 1] = b_vec.y;
    tile_b[local_row][local_col_vec * kVecWidth + 2] = b_vec.z;
    tile_b[local_row][local_col_vec * kVecWidth + 3] = b_vec.w;

    __syncthreads();

    if (row < M) {
#pragma unroll
      for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
        const float a_scalar = tile_a[local_row][k_local];
        const int b_col_start = local_col_vec * kVecWidth;
        acc.x += a_scalar * tile_b[k_local][b_col_start + 0];
        acc.y += a_scalar * tile_b[k_local][b_col_start + 1];
        acc.z += a_scalar * tile_b[k_local][b_col_start + 2];
        acc.w += a_scalar * tile_b[k_local][b_col_start + 3];
      }
    }

    __syncthreads();
  }

  if (row < M) {
    if (col_base + 0 < N) C[ROW_MAJOR_INDEX(row, col_base + 0, N)] = acc.x;
    if (col_base + 1 < N) C[ROW_MAJOR_INDEX(row, col_base + 1, N)] = acc.y;
    if (col_base + 2 < N) C[ROW_MAJOR_INDEX(row, col_base + 2, N)] = acc.z;
    if (col_base + 3 < N) C[ROW_MAJOR_INDEX(row, col_base + 3, N)] = acc.w;
  }

}
