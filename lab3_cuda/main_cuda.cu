#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstddef>
#include <windows.h>
#include "matrix_mul_cuda.h"

int main() {
    SetConsoleOutputCP(CP_UTF8);
    const int N = CUDA_N;
    const size_t bytes = (size_t)N * N * sizeof(double);

    std::cout << "Расчёт размера памяти\n";
    std::cout << "Размер одного double: " << sizeof(double) << " байт\n";
    std::cout << "Размер одной матрицы: " << bytes / 1024 / 1024 << " Мбайт\n";
    std::cout << "Матриц одновременно на CPU: 5 (A, B, BT, C_naive, C_tiled)\n";
    std::cout << "  = " << 5 * bytes / 1024 / 1024 << " Мбайт\n";
    std::cout << "Матриц одновременно на GPU: 3 (A, BT, C)\n";
    std::cout << "  = " << 3 * bytes / 1024 / 1024 << " Мбайт VRAM\n\n";

    // ---------- Выделение памяти на CPU ----------
    double* A        = new double[(size_t)N * N];
    double* B        = new double[(size_t)N * N];
    double* BT       = new double[(size_t)N * N];
    double* C_naive  = new double[(size_t)N * N];
    double* C_tiled  = new double[(size_t)N * N];
    fill_random_cuda(A, 1, N);
    fill_random_cuda(B, 2, N);
    transpose_cuda(B, BT, N);

    double *d_A, *d_BT, *d_C;
    cuda_prepare(A, BT, &d_A, &d_BT, &d_C, bytes);

    //GPU V1: наивная
    auto g0 = std::chrono::steady_clock::now();
    run_cuda_naive(d_A, d_BT, d_C, N);
    auto g1 = std::chrono::steady_clock::now();
    double t_naive = std::chrono::duration<double>(g1 - g0).count();
    cuda_copy_back(C_naive, d_C, bytes);

    //GPU V2: с тайлингом
    auto g2 = std::chrono::steady_clock::now();
    run_cuda_tiled(d_A, d_BT, d_C, N);
    auto g3 = std::chrono::steady_clock::now();
    double t_tiled = std::chrono::duration<double>(g3 - g2).count();
    cuda_copy_back(C_tiled, d_C, bytes);

    //Вывод
    std::cout << "Время выполнения (в секундах)\n";
    std::cout << "GPU naive: " << t_naive  << " s\n";
    std::cout << "GPU tiled: " << t_tiled  << " s\n\n";

    std::cout << "Проверка корректности\n";
    std::cout << "GPU naive vs GPU tiled: "
              << (compare_cuda(C_tiled, C_naive, N) ? "OK" : "FAIL") << "\n";

    std::cout << "\nУгловые элементы:\n";
    auto idx = [N](int i, int j){ return (size_t)i * N + j; };
    std::cout << "C_naive[0][0]          = " << C_naive [idx(0,0)] << "\n";
    std::cout << "C_tiled[0][0]          = " << C_tiled [idx(0,0)] << "\n";
    std::cout << "C_naive[N-1][N-1]      = " << C_naive [idx(N-1,N-1)] << "\n";
    std::cout << "C_tiled[N-1][N-1]      = " << C_tiled [idx(N-1,N-1)] << "\n";

    std::cout << "\n[ПАУЗА] Смотрите диспетчер задач.\n";
    std::cout << "Ожидается: CPU ~" << 5 * bytes / 1024 / 1024 << " МБ, "
              << "VRAM ~" << 3 * bytes / 1024 / 1024 << " МБ\n";
    std::cout << "Нажмите Enter...\n";
    std::cin.ignore();

    cuda_release(d_A, d_BT, d_C);

    delete[] A;
    delete[] B;
    delete[] BT;
    delete[] C_naive;
    delete[] C_tiled;
    return 0;
}