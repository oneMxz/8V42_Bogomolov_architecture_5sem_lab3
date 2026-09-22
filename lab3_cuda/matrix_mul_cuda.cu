#include "matrix_mul_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// Макрос проверки ошибок CUDA
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// ЯДРО V1: наивное умножение
// C[i][j] = sum_k A[i][k] * BT[j][k]
// Один поток вычисляет один элемент C.
// ============================================================
__global__ void naive_kernel(const double* A, const double* BT, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    double sum = 0.0;
    const double* a_row  = &A[(size_t)row * N];
    const double* bt_row = &BT[(size_t)col * N];
    for (int k = 0; k < N; ++k) {
        sum += a_row[k] * bt_row[k];
    }
    C[(size_t)row * N + col] = sum;
}

// ЯДРО V2: с тайлингом и shared memory
// TILE = 16 (или 32). Каждый блок грузит тайл A и тайл BT
// в shared memory и переиспользует их TILE раз.
constexpr int TILE = 16;

__global__ void tiled_kernel(const double* A, const double* BT, double* C, int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    double sum = 0.0;

    // Проходим по всем тайлам вдоль k
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + tx; // столбец A для этого потока
        int b_col = t * TILE + ty; // столбец BT для этого потока

        // Загружаем тайл A: As[ty][tx] = A[row][a_col]
        if (row < N && a_col < N)
            As[ty][tx] = A[(size_t)row * N + a_col];
        else
            As[ty][tx] = 0.0;

        // Загружаем тайл BT: Bs[ty][tx] = BT[col][b_col]
        // (помним, что BT[row_in_BT = j][col_in_BT = k], т.е. BT[j][k])
        if (col < N && b_col < N)
            Bs[ty][tx] = BT[(size_t)col * N + b_col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        // Вычисляем вклад этого тайла
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        // As[ty][k] — элемент A[row][t*TILE + k]
        // Bs[k][tx] — элемент BT[col][t*TILE + k]
        // Это именно то, что нужно: A[i][k] * BT[j][k]

        __syncthreads();
    }

    if (row < N && col < N)
        C[(size_t)row * N + col] = sum;
}

void cuda_prepare(const double* A, const double* BT,
                  double** d_A, double** d_BT, double** d_C,
                  size_t bytes) {
    CUDA_CHECK(cudaMalloc(d_A,  bytes));
    CUDA_CHECK(cudaMalloc(d_BT, bytes));
    CUDA_CHECK(cudaMalloc(d_C,  bytes));
    CUDA_CHECK(cudaMemcpy(*d_A,  A,  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_BT, BT, bytes, cudaMemcpyHostToDevice));
}

void cuda_release(double* d_A, double* d_BT, double* d_C) {
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_BT));
    CUDA_CHECK(cudaFree(d_C));
}

void cuda_copy_back(double* C, double* d_C, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));
}

void run_cuda_naive(double* d_A, double* d_BT, double* d_C, int N) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);
    naive_kernel<<<grid, block>>>(d_A, d_BT, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void run_cuda_tiled(double* d_A, double* d_BT, double* d_C, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);
    tiled_kernel<<<grid, block>>>(d_A, d_BT, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// Вспомогательные функции

void fill_random_cuda(double* M, unsigned seed, int N) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (long long i = 0; i < (long long)N * N; ++i)
        M[i] = dist(gen);
}

void transpose_cuda(const double* B, double* BT, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            BT[(size_t)j * N + i] = B[(size_t)i * N + j];
}

bool compare_cuda(const double* X, const double* Y, int N, double eps) {
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        double diff  = std::fabs(X[i] - Y[i]);
        double scale = std::max(1.0, std::fabs(X[i]));
        if (diff / scale > eps) return false;
    }
    return true;
}