#include <iostream>
#include <cstddef>
#include <chrono>
#include <iomanip>
#include <immintrin.h>     // SSE/AVX intrinsics
#include "vector_mul.h"


int main(){
    const std::size_t n_sz = N;
    size_t bytes_size = sizeof(double)*n_sz*n_sz;

    std::cout << "Рассчет размера памяти, требуемый для хранения матриц\n";
    std::cout << "Размер одного числа типа double (число с плавающей точкой, двойной точности): " << sizeof(double) << " байт.\n";
    std::cout << "Размер одной матрицы: " << bytes_size / 1024 / 1024 << " Мбайт\n";
    std::cout << "Размер, необходимый для хранения 3 матриц для счета и 3 матриц результата: " << 6 * bytes_size / 1024 / 1024 << " Мбайт\n\n";
    if (N < 2) {
        std::cerr << "N слишком мало для SIMD\n";
        return 1;
    }

    //Матрицы исходных данных 
    double* A = new double[(size_t)N * N];
    double* B = new double[(size_t)N * N];
    double* BT = new double[(size_t)N * N]; // транспонирование матрицы B для ускорения вычислений

    //Матрицы результата
    double* C_scalar  = new double[(size_t)N * N];
    double* C_scalarT  = new double[(size_t)N * N];
    double* C_sseT  = new double[(size_t)N * N];

    //Заполнение случайными значениями
    fill_random(A, 1);
    fill_random(B, 2);
    transpose(B,BT);

    //Вычисление произведение матриц с замером времени на подсчет.
    auto t0 = std::chrono::steady_clock::now();
    mul_scalar(A, B, C_scalar);

    auto t1 = std::chrono::steady_clock::now();
    mul_scalar_T(A, BT, C_scalarT);

    auto t2 = std::chrono::steady_clock::now();
    mul_sseT(A, BT, C_sseT);

    auto t3 = std::chrono::steady_clock::now();

    double times_scalar = std::chrono::duration<double>(t1 - t0).count();
    double times_scalar_T = std::chrono::duration<double>(t2 - t1).count();
    double times_sse_T = std::chrono::duration<double>(t3 - t2).count();

    //Вывод
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nВремя выполнения (в секундах)\n";
    std::cout << "Scalar: " << times_scalar << " s\n";
    std::cout << "ScalarT: " << times_scalar_T << " s\n";
    std::cout << "SSET:    " << times_sse_T    << " s \n\n";
    std::cout << "SSET ускорение vs Scalar:  " << times_scalar   / times_sse_T << "x\n";
    std::cout << "SSET ускорение vs ScalarT: " << times_scalar_T / times_sse_T << "x\n";
    std::cout << "Проверка корректности\n";

    std::cout << "Scalar vs scalarT: "
              << (compare(C_scalar, C_scalarT) ? "OK" : "FAIL") << "\n";
    std::cout << "SSET vs scalarT: "
              << (compare(C_scalarT, C_sseT) ? "OK" : "FAIL") << "\n";

    
    
    std::cout << "\nУгловые элементы:\n";
    std::cout << "C_scalar[0][0]     = " << C_scalar[0] << "\n";
    std::cout << "C_scalarT[0][0]     = " << C_scalarT[0] << "\n";
    std::cout << "C_sseT[0][0]       = " << C_sseT[0]   << "\n";
    std::cout << "C_scalar[N-1][N-1] = " << C_scalar[(n_sz-1)*n_sz + (n_sz-1)] << "\n";
    std::cout << "C_scalarT[N-1][N-1] = " << C_scalarT[(n_sz-1)*n_sz + (n_sz-1)] << "\n";
    std::cout << "C_sseT[N-1][N-1]   = " << C_sseT[(n_sz-1)*n_sz + (n_sz-1)]   << "\n";

    //Пауза для скриншота диспетчера
    std::cout << "\n[ПАУЗА] Смотрите диспетчер задач.\n";
    std::cout << "Ожидается ~" << 6 * bytes_size / 1024 / 1024 << " МиБ.\n";
    std::cout << "Нажмите Enter...\n";
    std::cin.ignore();

    delete[] A; 
    delete[] B;
    delete[] BT;
    delete[] C_scalar;
    delete[] C_scalarT;
    delete[] C_sseT;
    return 0;
}