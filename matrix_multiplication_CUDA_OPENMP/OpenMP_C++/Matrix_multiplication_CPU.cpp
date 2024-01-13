#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>
#include <fstream>


void matMulCPU(const float *A, const float *B, float *C, int size)
{
    // Loop through each row of matrix A
    for (int i = 0; i < size; ++i)
    {
        // Loop through each column of matrix B
        for (int j = 0; j < size; ++j)
        {
            // Initialize sum variable, where sum of row[i] and col[j] will be stored
            float sum = 0.0f;

            // Loop through each element of the row in A and column in B
            for (int k = 0; k < size; ++k)
            {
                // Multiply corresponding elements and accumulate the sum
                sum += A[i * size + k] * B[k * size + j];
            }
            // Store the calculated sum in the corresponding element of C matrix
            C[i * size + j] = sum;
        }
    }
}

int main(int argc, char *argv[])
{

    std::ofstream outputFile;
    outputFile.open("execution_times_CPU.csv");

    for(int i=0; i<=10; i++){


    int N = 100 * i;

    // Allocate memory for matrices on the host (CPU)
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C_CPU = new float[N * N];

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
    auto start_CPU = std::chrono::high_resolution_clock::now();

    // Calculate result matrix on CPU
    matMulCPU(A, B, C_CPU, N);

    // Stop timer for CPU Matrix Multiplication 
    auto end_CPU = std::chrono::high_resolution_clock::now();

    
    // Calculate compuatuion time of CPU matrix multiplication
    std::chrono::duration<double> duration_CPU = end_CPU - start_CPU;

    if (N==0){
        std::cout<<""<<std::endl;
    }
    else{
    //Print size of calculated matrices and time of multiplication process execution
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Calculation time (CPU) for N = " << N << " is: " << duration_CPU.count() << " seconds." << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    }

    if (N != 0) {
        outputFile << N << ","
        << duration_CPU.count() << "," "\n";
        }    

    // Deallocate memory for matrices on the host (CPU)
    delete[] A;
    delete[] B;
    delete[] C_CPU;

}

outputFile.close();

}
