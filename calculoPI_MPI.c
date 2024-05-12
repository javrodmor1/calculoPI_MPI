#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void calculatePiSeries(long num_steps, int rank, int size) {
    double pi;
    double x, sum = 0.0;
    double step = 1.0 / (double)num_steps;
    
    for (long i = rank; i < num_steps; i += size) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    
    double local_pi = step * sum;
    double global_pi;
    
    MPI_Reduce(&local_pi, &global_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Valor estimado de Pi usando la serie: %.7f\n", global_pi);
    }
}

void calculatePiMonteCarlo(long long samples, int rank, int size) {
    unsigned long long count = 0;
    unsigned int seed = (unsigned int)(time(NULL)) + rank;
    
    for (long long i = 0; i < samples; ++i) {
        double x, y;
        x = ((double)rand_r(&seed)) / ((double)RAND_MAX);
        y = ((double)rand_r(&seed)) / ((double)RAND_MAX);

        if (x * x + y * y <= 1.0) {
            count++;
        }
    }
    
    unsigned long long global_count;
    
    MPI_Reduce(&count, &global_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Valor estimado de Pi usando Monte Carlo: %.7f\n", 4.0 * global_count / (samples * size));
    }
}

int main(int argc, char* argv[]) {
    
    int rank, size;
    long long monte_carlo_samples = 100000; // Valor por defecto para Monte Carlo
    long num_steps = 100000; // Valor por defecto para la serie
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc > 1) {
        monte_carlo_samples = atoll(argv[1]); // Para su uso en línea de comandos
        num_steps = atoll(argv[1]); // También cambiamos el número de pasos para la serie
    }
    
    if (rank == 0) {
        printf("Numero de procesos: %d\n", size);
    }
    
    start_time = MPI_Wtime(); // Comenzar a medir el tiempo

    
    // Calculo de pi usando la serie
    calculatePiSeries(num_steps, rank, size);
    
    // Cálculo de pi usando el método de monte carlo
    calculatePiMonteCarlo(monte_carlo_samples, rank, size);
    
    end_time = MPI_Wtime(); // Finalizar la medición del tiempo

    
    MPI_Barrier(MPI_COMM_WORLD); // Sincronizar todos los procesos antes de finalizar
    MPI_Finalize();
    
    if (rank == 0) {
        printf("Tiempo total de ejecución: %.6f segundos\n", end_time - start_time);
    }

    return 0;
}
