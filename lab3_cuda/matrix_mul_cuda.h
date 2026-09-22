#ifndef MATRIX_MUL_CUDA_H
#define MATRIX_MUL_CUDA_H

#include <cstddef>

constexpr int CUDA_N = 4096;

// Выделение device-памяти под A, BT, C и копирование A, BT на GPU
void cuda_prepare(const double* A, const double* BT,
                  double** d_A, double** d_BT, double** d_C,
                  size_t bytes);

// Освобождение device-памяти
void cuda_release(double* d_A, double* d_BT, double* d_C);

// Копирование результата C обратно на CPU
void cuda_copy_back(double* C, double* d_C, size_t bytes);

// Запуск ядра (только ядро + синхронизация, без копирования)
void run_cuda_naive(double* d_A, double* d_BT, double* d_C, int N);
void run_cuda_tiled(double* d_A, double* d_BT, double* d_C, int N);

// Вспомогательные функции
void fill_random_cuda(double* M, unsigned seed, int N);
void transpose_cuda(const double* B, double* BT, int N);
bool compare_cuda(const double* X, const double* Y, int N, double eps = 1e-6);

#endif