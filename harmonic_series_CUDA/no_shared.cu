#include <stdio.h>

__global__ void harmonicSumWithoutShared(float *result, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;

    while (index < n) {
        sum += 1.0f / (index + 1);
        index += blockDim.x * gridDim.x;
    }
    
    atomicAdd(result, sum);
}

int main() {
    int n = 10000; // Define the number of elements in the harmonic series
    float *d_result, h_result = 0.0f;
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    harmonicSumWithoutShared<<<numBlocks, blockSize>>>(d_result, n);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum of first %d elements of the harmonic series without shared memory: %f\n", n, h_result);

    cudaFree(d_result);

    return 0;
}
