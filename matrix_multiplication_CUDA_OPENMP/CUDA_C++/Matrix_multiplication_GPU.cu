#include <iostream>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <omp.h>
#include <fstream>

__global__ void matMulGPU(const float *A, const float *B, float *C, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && column < size)
    {
        float sum = 0;
        for (int k = 0; k < size; ++k)
        {
            sum += A[row * size + k] * B[k * size + column];
        }
        C[row * size + column] = sum;
    }
}



int main(int argc, char *argv[])
{

    cudaSetDevice(0);

    std::ofstream outputFile;
    outputFile.open("execution_times_GPU_bsize1.csv");

    for(int i=0; i<=10; i++){

    int BLOCK_SIZE = 1;

    int N = 100 * i;

    // Allocate memory for matrices on the host (CPU)
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C_GPU = new float[N * N];
    float *C_CPU = new float[N * N];
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

    //Start the timer of GPU total time
    auto start_GPU = std::chrono::high_resolution_clock::now();

    auto start_memmory_allocation_GPU = std::chrono::high_resolution_clock::now();
    
    // Allocate mmory on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    auto stop_memmory_allocation_GPU = std::chrono::high_resolution_clock::now();
   
    auto start_memmory_copy_to_GPU = std::chrono::high_resolution_clock::now();
    // Copy matrices A and B from host (CPU) to device (GPU)
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    auto stop_memmory_copy_to_GPU = std::chrono::high_resolution_clock::now();

    // Set grid and block dimensions for CUDA kernel
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    auto start_kernel_execution_GPU = std::chrono::high_resolution_clock::now();
    // Launch CUDA matrix multiplication kernel
    matMulGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Wait for kernel to complete
    cudaDeviceSynchronize();
    auto stop_kernel_execution_GPU= std::chrono::high_resolution_clock::now();

    auto start_memmory_copy_to_CPU = std::chrono::high_resolution_clock::now();
    // Copy result matrix C from device to host
    cudaMemcpy(C_GPU, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
   
   auto stop_memmory_copy_to_CPU = std::chrono::high_resolution_clock::now();

    // Free device (GPU) memory 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Stop the timer
    auto end_GPU = std::chrono::high_resolution_clock::now();
    
    // Calulate execution time
    std::chrono::duration<double> duration_GPU = end_GPU - start_GPU;
    std::chrono::duration<double> CudaMalloc = stop_memmory_allocation_GPU - start_memmory_allocation_GPU;
    std::chrono::duration<double> CudaMemcpy_to_GPU = stop_memmory_copy_to_GPU - start_memmory_copy_to_GPU;
    std::chrono::duration<double> CudaMemcpy_to_CPU = stop_memmory_copy_to_CPU - start_memmory_copy_to_CPU;
    std::chrono::duration<double> kernel_execution = stop_kernel_execution_GPU - start_kernel_execution_GPU;

    //Print execution time for certain size of the matrices.
    if (N==0){
        std::cout<<""<<std::endl;
    }
    else{
    std::cout <<"Calculation time (GPU) for N = " << N << " is: " 
    << duration_GPU.count() << " s, " << "CudaMalloc: " 
    << CudaMalloc.count() << " s, " << "CudaMemcpy to GPU: " 
    << CudaMemcpy_to_GPU.count() << " s, "<<"CudaMemcpy to CPU: "
    << CudaMemcpy_to_CPU.count() << " s, "<<"Kernel execution: " 
    << kernel_execution.count() << " s"<<std::endl;
    }

   
    if (N != 0) {
        outputFile << N << ","
        << duration_GPU.count() << ","
        << CudaMalloc.count() << ","
        << CudaMemcpy_to_GPU.count() << ","
        << CudaMemcpy_to_CPU.count() << ","
        << kernel_execution.count() << "\n";
        }    

    // Deallocate memory for matrices on the host (CPU)
    delete[] A;
    delete[] B;
    delete[] C_GPU;
}

outputFile.close();

}
