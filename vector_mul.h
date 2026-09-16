#ifndef VECTOR_MUL_H
#define VECTOR_MUL_H
#include <cstddef>

constexpr std::size_t N = 4096;


void mul_scalar(const double* A, const double* B, double* C);
void mul_scalar_T(const double* A, const double* BT, double* C);
void mul_sseT(const double* A, const double* BT, double* C);// BT = B^T; C = A * BT  (SSE с транспонированием)

void fill_random(double* M, unsigned seed);
void transpose(const double* B, double* B_T);
bool compare(const double* X, const double* Y, double eps = 1e-6);// Сравнение двух матриц с относительной погрешностью eps

#endif