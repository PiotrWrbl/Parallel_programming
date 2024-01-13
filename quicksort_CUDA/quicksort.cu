#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

const int N = 100;

__device__ void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void quickSort(int* array, int left, int right) {
    if (left >= right) return;

    int pivot = array[right]; //choose pivot
    int i = left - 1;

    //partitioning 
    for (int j = left; j <= right - 1; j++) {
        if (array[j] < pivot) {
            i++;
            swap(&array[i], &array[j]);
        }
    }

    swap(&array[i + 1], &array[right]); //place the pivot at correct position
    int partition = i + 1;

    if (left < right) //We first sort smaller partition
        {
        if (partition - left < right - partition) {
            quickSort<<<1, 1>>>(array, left, partition - 1);
            quickSort<<<1, 1>>>(array, partition + 1, right);
        } else {
            quickSort<<<1, 1>>>(array, partition + 1, right);
            quickSort<<<1, 1>>>(array, left, partition - 1);
        }
    }
}

int main() {
    
    srand(time(NULL));

     int* harray = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        harray[i] = rand() % 1000; 
    }

    printf("Original array:\n");
   
    for (int i = 0; i < N; i++) {
        printf("%d ", harray[i]);
    }
    printf("\n");

    int* darray;
    cudaMalloc((void**)&darray, N * sizeof(int));
    cudaMemcpy(darray, harray, N * sizeof(int), cudaMemcpyHostToDevice);

    quickSort<<<1, 1>>>(darray, 0, N - 1);
    cudaDeviceSynchronize();

    cudaMemcpy(harray, darray, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted array:\n");
    
    for (int i = 0; i < N; i++) {
        printf("%d ", harray[i]);
    }
    printf("\n");

    cudaFree(darray);
    free(harray);

    return 0;
}
