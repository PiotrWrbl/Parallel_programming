/*
CUDA - generation of array of N elements and calculates even and odd numbers occurence - with streams
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DEBUG 0
__host__
void errorexit(const char *s) {
    printf("\n%s\n",s); 
    exit(EXIT_FAILURE);   
}

__host__ 
void generate(int *matrix, int matrixSize) {
  srand(time(NULL));
  for(int i=0; i<matrixSize; i++) {
    matrix[i] = rand()%1000;
  }
}

__global__ 
void calculation(int *matrix, int *even, int *odd, int matrixSize, int streamChunk, int streamId) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x+streamId*streamChunk;

    if(my_index < matrixSize) {
      if(matrix[my_index] % 2) {
        atomicAdd(odd, 1);
      } else {
        atomicAdd(even, 1);
      }
    } 
}

int main(int argc,char **argv) {

  ///define number of streams
  int numberOfStreams = 4;
  
  //define array size and allocate memory on host
  int matrixSize=100000;
  int *hMatrix=(int*)malloc(matrixSize*sizeof(int));

  //get number of chunks to operate per stream
  int streamChunk = matrixSize/numberOfStreams;

  printf("Stream chunk is %d \n", streamChunk);
 
  //define kernel size per stream
  int threadsinblock=1000;
  int blocksingrid=1+((streamChunk-1)/threadsinblock); 

  printf("blocksingrid is %d \n", blocksingrid);
  

  
  //allocate memory for odd and even numbers counters - host
  int *hEven=(int*)malloc(sizeof(int));
  int *hOdd=(int*)malloc(sizeof(int));


  //create streams
  cudaStream_t streams[numberOfStreams];
  for(int i=0;i<numberOfStreams;i++) {
      if (cudaSuccess!=cudaStreamCreate(&streams[i]))
           errorexit("Error creating stream");
    }

  //allocate memory for odd and even numbers counters and array on device and for array on host with cudaMallocHost
  int *dEven=NULL;
  int *dOdd=NULL;
  int *dMatrix=NULL;

  if (cudaSuccess!=cudaMallocHost((void **) &hMatrix, matrixSize*sizeof(int)))
      errorexit("Error allocating memory on the CPU");

  //generate random numbers
  generate(hMatrix, matrixSize);

  if(DEBUG) {
    printf("Generated numbers: \n");
    for(int i=0; i<matrixSize; i++) {
      printf("%d ", hMatrix[i]);
    }
    printf("\n");
  }


  if (cudaSuccess!=cudaMalloc((void **)&dEven,sizeof(int)))
      errorexit("Error allocating memory on the GPU");

  if (cudaSuccess!=cudaMalloc((void **)&dOdd,sizeof(int)))
      errorexit("Error allocating memory on the GPU");
  
  if (cudaSuccess!=cudaMalloc((void **)&dMatrix,matrixSize*sizeof(int)))
      errorexit("Error allocating memory on the GPU");

  //initialize allocated counters with 0
  if (cudaSuccess!=cudaMemset(dEven,0, sizeof(int)))
      errorexit("Error initializing memory on the GPU");

  if(cudaSuccess!=cudaMemset(dOdd,0, sizeof(int)))
      errorexit("Error initializing memory on the GPU");

  //execute operation in each stream - copy chunk of data and run calculations
  for(int i=0; i<numberOfStreams; i++) {
    cudaMemcpyAsync(&dMatrix[streamChunk*i],&hMatrix[streamChunk*i],streamChunk*sizeof(int),cudaMemcpyHostToDevice, streams[i]);      
    calculation<<<blocksingrid, threadsinblock, threadsinblock*sizeof(double), streams[i]>>>(dMatrix, dEven, dOdd, matrixSize, streamChunk, i);
  }

  cudaDeviceSynchronize();

  //copy results from GPU
  if (cudaSuccess!=cudaMemcpy(hEven, dEven, sizeof(int),cudaMemcpyDeviceToHost))
     errorexit("Error copying results");

  if (cudaSuccess!=cudaMemcpy(hOdd, dOdd, sizeof(int),cudaMemcpyDeviceToHost))
     errorexit("Error copying results");
  
    printf("Found %d even numbers \n", *hEven);
    printf("Found %d odd numbers \n", *hOdd);
    printf("Found %d total numbers \n", *hEven + *hOdd);

//Free memory and destroy streams
    for(int i=0;i<numberOfStreams;i++) {
      if (cudaSuccess!=cudaStreamDestroy(*(streams+i)))
         errorexit("Error creating stream");
    }

  free(hOdd);
  free(hEven);
  
  if (cudaSuccess!=cudaFreeHost(hMatrix))
     errorexit("Error when deallocating space on the CPU");
  if (cudaSuccess!=cudaFree(dEven))
    errorexit("Error when deallocating space on the GPU");
  if (cudaSuccess!=cudaFree(dOdd))
    errorexit("Error when deallocating space on the GPU");
  if (cudaSuccess!=cudaFree(dMatrix))
    errorexit("Error when deallocating space on the GPU");
}
