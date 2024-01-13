#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void matMulOpenMP(const float *A, const float *B, float *C_OpenMP, int size) {
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

int main(int argc, char *argv[]) {
    int num_threads = 4;
    omp_set_num_threads(num_threads);

    FILE *outputFile;
    outputFile = fopen("execution_times_openMP_4.csv", "w");
    if (outputFile == NULL) {
        perror("Error opening the file.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i <= 10; i++) {
        int N = 100 * i;
        float *A = (float *)malloc(N * N * sizeof(float));
        float *B = (float *)malloc(N * N * sizeof(float));
        float *C_OpenMP = (float *)malloc(N * N * sizeof(float));

        srand(time(NULL));

        for (int i = 0; i < N * N; i++) {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
        }

        double start_CPU_OpenMP = omp_get_wtime();
        matMulOpenMP(A, B, C_OpenMP, N);
        double end_CPU_OpenMP = omp_get_wtime();

        double duration_CPU_OpenMP = end_CPU_OpenMP - start_CPU_OpenMP;

        if (N != 0) {
            printf("-----------------------------------------------------\n");
            printf("Calculation time (OpenMP) for N = %d is: %f seconds.\n", N, duration_CPU_OpenMP);
            printf("-----------------------------------------------------\n");

            fprintf(outputFile, "%d,%f\n", N, duration_CPU_OpenMP);
        }

        free(A);
        free(B);
        free(C_OpenMP);
    }

    fclose(outputFile);
    return EXIT_SUCCESS;
}
