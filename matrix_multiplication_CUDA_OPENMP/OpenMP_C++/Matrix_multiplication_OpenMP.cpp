#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>
#include <fstream>

void matMulOpenMP(const float *A, const float *B, float *C_OpenMP, int size) 
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C_OpenMP[i * size + j] = sum;
        }
    }
}

int main(int argc, char *argv[])
{
    int num_threads = 4;
    omp_set_num_threads(num_threads);

    std::ofstream outputFile;
    outputFile.open("execution_times_openMP_4.csv");

    for(int i=0; i<=10; i++){

    int N = 100 * i;

    // Allocate memory for matrices on the host (CPU)
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C_OpenMP = new float[N * N];


    // Random numbers generator to initialize matrices A and B
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (size_t i = 0; i < N * N; i++)
    {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }

    // Matrix multiplication procedure

    // Start timer for CPU Matrix Multiplication
    auto start_CPU_OpenMP = std::chrono::high_resolution_clock::now();

    // Calculate result matrix on CPU
    matMulOpenMP(A, B, C_OpenMP, N);

    // Stop timer for CPU Matrix Multiplication 
    auto end_CPU_OpenMP = std::chrono::high_resolution_clock::now();

    

    std::chrono::duration<double> duration_CPU_OpenMP = end_CPU_OpenMP - start_CPU_OpenMP;
    if (N==0){
        std::cout<<""<<std::endl;
    }
    else{
    //Print size of calculated matrices and time of multiplication process execution
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Calculation time (OpenMP) for N = " << N << " is: " << duration_CPU_OpenMP.count() << " seconds." << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    }

    if (N != 0) {
        outputFile << N << ","
        << duration_CPU_OpenMP.count() << ","<<std::endl;
        }    

    // Deallocate memory for matrices on the host (CPU)
    delete[] A;
    delete[] B;
    delete[] C_OpenMP;
}

outputFile.close();

}
