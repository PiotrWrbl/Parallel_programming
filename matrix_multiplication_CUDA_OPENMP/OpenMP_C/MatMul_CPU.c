#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matMulCPU(const float *A, const float *B, float *C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    FILE *outputFile;
    outputFile = fopen("execution_times_CPU.csv", "w");
    if (outputFile == NULL) {
        perror("Error opening the file.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i <= 10; i++) {
        int N = 100 * i;
        float *A = (float *)malloc(N * N * sizeof(float));
        float *B = (float *)malloc(N * N * sizeof(float));
        float *C_CPU = (float *)malloc(N * N * sizeof(float));

        srand(time(NULL));

        for (int i = 0; i < N * N; i++) {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
        }

        clock_t start_CPU = clock();
        matMulCPU(A, B, C_CPU, N);
        clock_t end_CPU = clock();

        double duration_CPU = ((double)(end_CPU - start_CPU)) / CLOCKS_PER_SEC;

        if (N != 0) {
            printf("-----------------------------------------------------\n");
            printf("Calculation time (CPU) for N = %d is: %f seconds.\n", N, duration_CPU);
            printf("-----------------------------------------------------\n");

            fprintf(outputFile, "%d,%f\n", N, duration_CPU);
        }

        free(A);
        free(B);
        free(C_CPU);
    }

    fclose(outputFile);
    return EXIT_SUCCESS;
}
