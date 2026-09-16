#include <iostream>
#include <immintrin.h>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include "vector_mul.h"

/*=========================================
Скалярное умножение матриц
===========================================*/

void mul_scalar(const double* A, const double* B, double* C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void mul_scalar_T(const double* A, const double* BT, double* C) {
    for (int i = 0; i < (int)N; ++i) {
        const double* a_row = &A[(std::size_t)i * N];
        for (int j = 0; j < (int)N; ++j) {
            const double* bt_row = &BT[(std::size_t)j * N];
            double sum = 0.0;
            for (std::size_t k = 0; k < N; ++k)
                sum += a_row[k] * bt_row[k];
            C[(std::size_t)i * N + j] = sum;
        }
    }
}

/*==================================================
Произведение матриц с использованием SIMD инструкций 
====================================================*/
//SSE-умножение с транспонированием B 
// На каждый k берём из BT[j][k] и BT[j+1][k] — два разных row.
// Собираем их в регистр через _mm_set_pd.
void mul_sseT(const double* A, const double* BT, double* C) {
    if (N < 2) return;
    for (int i = 0; i < N; ++i) {
        const double* a_row = &A[(long long)i * N];
        for (int j = 0; j+1< N; j += 2) {
            __m128d sum = _mm_setzero_pd();
            const double* bt0 = &BT[(long long)j * N];
            const double* bt1 = &BT[(long long)(j + 1) * N];
            for (int k = 0; k < N; ++k) {
                __m128d a  = _mm_set1_pd(a_row[k]);
                // b = [BT[j][k], BT[j+1][k]]
                __m128d b  = _mm_set_pd(bt1[k], bt0[k]);
                sum = _mm_add_pd(sum, _mm_mul_pd(a, b));
            }
            _mm_storeu_pd(&C[(long long)i * N + j], sum);
        }
    }
}

/*=========================================
Вспомогательные методы
===========================================*/

//Генерация матриц размером 4096*4096 с произвольным набором данных
void fill_random(double* M, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (long long i = 0; i < static_cast<long long>(N * N); ++i) {
        M[i] = dist(gen);
    }
}

// Транспонирование матрицы 
void transpose(const double* B, double* B_T) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            B_T[(long long)j * N + i] = B[(long long)i * N + j];
        }
    }
}

// Cравнение
bool compare(const double* X, const double* Y, double eps) {
    for (std::size_t i = 0; i < N * N; ++i) {
        double diff  = std::fabs(X[i] - Y[i]);
        double scale = std::max(1.0, std::fabs(X[i]));
        if (diff / scale > eps) return false;
    }
    return true;
}