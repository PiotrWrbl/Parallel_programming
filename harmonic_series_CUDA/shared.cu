#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void geometricSumShared(float *result, int n) {
    
    __shared__ float partialSum[BLOCK_SIZE];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    while (index < n) {
        sum += 1.0f / (1 << index);
        i += blockDim.x * gridDim.x;
    }

    partialSum[tid] = sum;
    __syncthreads();

    // Reduction within the block using shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, partialSum[0]);
    }
}

int main() {

    for (int j = 1; j<=20; j++){    
    int n = 20; // Define the number of elements in the series
    float *d_result, h_result = 0.0f;
    cudaMalloc((void **)&d_result, sizeof(float));
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    geometricSumShared<<<numBlocks, BLOCK_SIZE>>>(d_result, n);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum of first %d elements of the harmonic series with shared memory: %f\n", n, h_result);

    cudaFree(d_result);
    }

    return 0;
}
