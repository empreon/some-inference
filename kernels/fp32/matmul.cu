#include "common.cuh"
#include <cstdint>

#ifndef MM_TILED_TILE
#define MM_TILED_TILE 16
#endif

#ifndef MM_VEC_TILE
#define MM_VEC_TILE 32
#endif

#ifndef MM_VEC_WIDTH
#define MM_VEC_WIDTH 4
#endif

#ifndef MM_VBLOCK_ROWS
#define MM_VBLOCK_ROWS 2
#endif

#if (MM_VEC_TILE % MM_VEC_WIDTH) != 0
#error "MM_VEC_TILE must be divisible by MM_VEC_WIDTH."
#endif

#if (MM_VEC_TILE % MM_VBLOCK_ROWS) != 0
#error "MM_VEC_TILE must be divisible by MM_VBLOCK_ROWS."
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
  constexpr int kTileSize = MM_TILED_TILE;
  if (blockDim.x != kTileSize || blockDim.y != kTileSize) {
    return;
  }

  __shared__ float tile_a[kTileSize][kTileSize];
  __shared__ float tile_b[kTileSize][kTileSize];

  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x;
  const int row = blockIdx.y * kTileSize + local_row;
  const int col = blockIdx.x * kTileSize + local_col;

  float value = 0.0f;
  const int tile_count = CEIL_DIV(K, kTileSize);

  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int k_a = tile_idx * kTileSize + local_col;
    const int k_b = tile_idx * kTileSize + local_row;

    tile_a[local_row][local_col] =
        (row < M && k_a < K) ? A[ROW_MAJOR_INDEX(row, k_a, K)] : 0.0f;
    tile_b[local_row][local_col] =
        (k_b < K && col < N) ? B[ROW_MAJOR_INDEX(k_b, col, N)] : 0.0f;

    __syncthreads();

    for (int k_local = 0; k_local < kTileSize; ++k_local) {
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
  constexpr int kVecTileSize = MM_VEC_TILE;
  constexpr int kVecWidth = MM_VEC_WIDTH;
  constexpr int kVBlockRows = MM_VBLOCK_ROWS;
  constexpr int kExpectedBlockX = kVecTileSize / kVecWidth;
  constexpr int kExpectedBlockY = kVecTileSize / kVBlockRows;

  if (blockDim.x != kExpectedBlockX || blockDim.y != kExpectedBlockY) {
    return;
  }

  __shared__ float tile_a[kVecTileSize][kVecTileSize];
  __shared__ float tile_b[kVecTileSize][kVecTileSize];

  const int local_row_block = threadIdx.y;
  const int local_col_vec = threadIdx.x;
  const int row_base = blockIdx.y * kVecTileSize + local_row_block * kVBlockRows;
  const int col_base = blockIdx.x * kVecTileSize + local_col_vec * kVecWidth;

  float4 acc[kVBlockRows];
  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
    acc[row_offset] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const int tile_count = CEIL_DIV(K, kVecTileSize);

  for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
    const int k_col_start = tile_idx * kVecTileSize + local_col_vec * kVecWidth;

    #pragma unroll
    for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
      const int tile_row = local_row_block * kVBlockRows + row_offset;
      const int row = row_base + row_offset;

      float4 a_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (row < M) {
        const float* a_row = A + ROW_MAJOR_INDEX(row, 0, K);
        a_vec = load_float4_with_bounds(a_row, k_col_start, K);
      }
      tile_a[tile_row][local_col_vec * kVecWidth + 0] = a_vec.x;
      tile_a[tile_row][local_col_vec * kVecWidth + 1] = a_vec.y;
      tile_a[tile_row][local_col_vec * kVecWidth + 2] = a_vec.z;
      tile_a[tile_row][local_col_vec * kVecWidth + 3] = a_vec.w;

      const int k_b = tile_idx * kVecTileSize + tile_row;
      float4 b_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (k_b < K) {
        const float* b_row = B + ROW_MAJOR_INDEX(k_b, 0, N);
        b_vec = load_float4_with_bounds(b_row, col_base, N);
      }
      tile_b[tile_row][local_col_vec * kVecWidth + 0] = b_vec.x;
      tile_b[tile_row][local_col_vec * kVecWidth + 1] = b_vec.y;
      tile_b[tile_row][local_col_vec * kVecWidth + 2] = b_vec.z;
      tile_b[tile_row][local_col_vec * kVecWidth + 3] = b_vec.w;
    }

    __syncthreads();

    #pragma unroll
    for (int k_local = 0; k_local < kVecTileSize; ++k_local) {
      const int b_col_start = local_col_vec * kVecWidth;
      const float4 b_values = make_float4(
          tile_b[k_local][b_col_start + 0],
          tile_b[k_local][b_col_start + 1],
          tile_b[k_local][b_col_start + 2],
          tile_b[k_local][b_col_start + 3]);

      #pragma unroll
      for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
        const int row = row_base + row_offset;
        if (row >= M) {
          continue;
        }
        const int tile_row = local_row_block * kVBlockRows + row_offset;
        const float a_scalar = tile_a[tile_row][k_local];

        acc[row_offset].x += a_scalar * b_values.x;
        acc[row_offset].y += a_scalar * b_values.y;
        acc[row_offset].z += a_scalar * b_values.z;
        acc[row_offset].w += a_scalar * b_values.w;
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int row_offset = 0; row_offset < kVBlockRows; ++row_offset) {
    const int row = row_base + row_offset;
    if (row >= M) {
      continue;
    }
    if (col_base + 0 < N) C[ROW_MAJOR_INDEX(row, col_base + 0, N)] = acc[row_offset].x;
    if (col_base + 1 < N) C[ROW_MAJOR_INDEX(row, col_base + 1, N)] = acc[row_offset].y;
    if (col_base + 2 < N) C[ROW_MAJOR_INDEX(row, col_base + 2, N)] = acc[row_offset].z;
    if (col_base + 3 < N) C[ROW_MAJOR_INDEX(row, col_base + 3, N)] = acc[row_offset].w;
  }
}
