#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DEBUG 0
#define MAX 1000 // Change this to your desired maximum value

__host__
void errorexit(const char *s) {
    printf("\n%s\n", s);
    exit(EXIT_FAILURE);
}

__host__
void generate(int *matrix, int matrixSize) {
    srand(time(NULL));
    for (int i = 0; i < matrixSize; i++) {
        matrix[i] = 1 + rand() % MAX; // Generate numbers in the range <1, MAX>
    }
}

__global__
void calculation(int *matrix, int *histogram, int matrixSize) {
    int my_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (my_index < matrixSize) {
        atomicAdd(&histogram[matrix[my_index] - 1], 1); // Increment the count of the corresponding number in the histogram
    }
}

int main(int argc, char **argv) {

    // Define array size and allocate memory on host
    int matrixSize = 10240000;
    int *hMatrix = (int *)malloc(matrixSize * sizeof(int));

    // Generate random numbers
    generate(hMatrix, matrixSize);

    if (DEBUG) {
        printf("Generated numbers: \n");
        for (int i = 0; i < matrixSize; i++) {
            printf("%d ", hMatrix[i]);
        }
        printf("\n");
    }

    // Allocate memory for histogram - host
    int *hHistogram = (int *)calloc(MAX* sizeof(int));

    // Allocate memory for histogram and array - device
    int *dHistogram = NULL;
    int *dMatrix = NULL;

    if (cudaSuccess != cudaMalloc((void **)&dHistogram, MAX * sizeof(int)))
        errorexit("Error allocating memory on the GPU");

    if (cudaSuccess != cudaMalloc((void **)&dMatrix, matrixSize * sizeof(int)))
        errorexit("Error allocating memory on the GPU");

    // Copy array to device
    if (cudaSuccess != cudaMemcpy(dMatrix, hMatrix, matrixSize * sizeof(int), cudaMemcpyHostToDevice))
        errorexit("Error copying input data to device");

    int threadsInBlock = 1024;
    int blocksInGrid = (matrixSize + threadsInBlock - 1) / threadsInBlock;

    // Run kernel on GPU
    calculation<<<blocksInGrid, threadsInBlock>>>(dMatrix, dHistogram, matrixSize);

    // Copy results from GPU
    if (cudaSuccess != cudaMemcpy(hHistogram, dHistogram, MAX * sizeof(int), cudaMemcpyDeviceToHost))
        errorexit("Error copying results");

    // Display histogram results
    printf("Histogram of occurrences for numbers in range <1, %d>:\n", MAX);
    for (int i = 0; i < MAX; i++) {
        printf("%d: %d times\n", i + 1, hHistogram[i]);
    }

    // Free memory
    free(hMatrix);
    free(hHistogram);

    if (cudaSuccess != cudaFree(dHistogram))
        errorexit("Error when deallocating space on the GPU");
    if (cudaSuccess != cudaFree(dMatrix))
        errorexit("Error when deallocating space on the GPU");

    return 0;
}
