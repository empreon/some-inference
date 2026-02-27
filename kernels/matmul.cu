#include "common.cuh"
#include <cstdint>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

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

extern "C" __global__ void matmul_vectorized(const float* A,
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
  const int k_vec = 4;
  const bool is_vector_loader = (local_col % k_vec) == 0;

  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int k_base = tile_idx * TILE_SIZE;

    if (is_vector_loader) {
      const int k_a = k_base + local_col;
      float a0 = 0.0f;
      float a1 = 0.0f;
      float a2 = 0.0f;
      float a3 = 0.0f;

      if (row < M) {
        if (k_a + 3 < K) {
          const float* a_ptr = A + ROW_MAJOR_INDEX(row, k_a, K);
          if ((reinterpret_cast<std::uintptr_t>(a_ptr) & 0xF) == 0) {
            const float4 a_vec = *reinterpret_cast<const float4*>(a_ptr);
            a0 = a_vec.x;
            a1 = a_vec.y;
            a2 = a_vec.z;
            a3 = a_vec.w;
          } else {
            a0 = a_ptr[0];
            a1 = a_ptr[1];
            a2 = a_ptr[2];
            a3 = a_ptr[3];
          }
        } else {
          if (k_a < K) a0 = A[ROW_MAJOR_INDEX(row, k_a, K)];
          if (k_a + 1 < K) a1 = A[ROW_MAJOR_INDEX(row, k_a + 1, K)];
          if (k_a + 2 < K) a2 = A[ROW_MAJOR_INDEX(row, k_a + 2, K)];
          if (k_a + 3 < K) a3 = A[ROW_MAJOR_INDEX(row, k_a + 3, K)];
        }
      }

      tile_a[local_row][local_col + 0] = a0;
      tile_a[local_row][local_col + 1] = a1;
      tile_a[local_row][local_col + 2] = a2;
      tile_a[local_row][local_col + 3] = a3;

      const int k_b = k_base + local_row;
      const int b_col = blockIdx.x * TILE_SIZE + local_col;
      float b0 = 0.0f;
      float b1 = 0.0f;
      float b2 = 0.0f;
      float b3 = 0.0f;

      if (k_b < K) {
        if (b_col + 3 < N) {
          const float* b_ptr = B + ROW_MAJOR_INDEX(k_b, b_col, N);
          if ((reinterpret_cast<std::uintptr_t>(b_ptr) & 0xF) == 0) {
            const float4 b_vec = *reinterpret_cast<const float4*>(b_ptr);
            b0 = b_vec.x;
            b1 = b_vec.y;
            b2 = b_vec.z;
            b3 = b_vec.w;
          } else {
            b0 = b_ptr[0];
            b1 = b_ptr[1];
            b2 = b_ptr[2];
            b3 = b_ptr[3];
          }
        } else {
          if (b_col < N) b0 = B[ROW_MAJOR_INDEX(k_b, b_col, N)];
          if (b_col + 1 < N) b1 = B[ROW_MAJOR_INDEX(k_b, b_col + 1, N)];
          if (b_col + 2 < N) b2 = B[ROW_MAJOR_INDEX(k_b, b_col + 2, N)];
          if (b_col + 3 < N) b3 = B[ROW_MAJOR_INDEX(k_b, b_col + 3, N)];
        }
      }

      tile_b[local_row][local_col + 0] = b0;
      tile_b[local_row][local_col + 1] = b1;
      tile_b[local_row][local_col + 2] = b2;
      tile_b[local_row][local_col + 3] = b3;
    }

    __syncthreads();

    if (row < M && col < N) {
#pragma unroll
      for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
        value += tile_a[local_row][k_local] * tile_b[k_local][local_col];
      }
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[ROW_MAJOR_INDEX(row, col, N)] = value;
  }

}
