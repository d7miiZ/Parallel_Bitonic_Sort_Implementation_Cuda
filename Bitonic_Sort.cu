#include <stdio.h>
#include <math.h>
#include <inttypes.h>

__global__ void sort(unsigned long long *a, int step, int stage, unsigned long long sl, unsigned long long N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int shift = N / 2;
    int on = (index % N) < (N / 2);
    int ascinding = (index / sl) % 2 == 0 ? 1 : 0;

    if (on)
    {
        if (ascinding)
        {
            if (a[index] > a[index + shift])
            {
                unsigned long long temp = a[index];
                a[index] = a[index + shift];
                a[index + shift] = temp;
            }
        }
        else
        {
            if (a[index] < a[index + shift])
            {
                unsigned long long temp = a[index];
                a[index] = a[index + shift];
                a[index + shift] = temp;
            }
        }
    }
}

int main(void)
{
    unsigned long long *a;
    unsigned long long *d_a;
    int steps;
    int i, j;
    int dev;
    int threads, block;
    cudaDeviceProp prop;

    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    printf("Choose A Number For x in 2^x: ");
    scanf("%d", &steps);
    unsigned long long n = pow(2, steps);
    
    printf("\nNumber Of Elements Will Be %llu", n);
    unsigned long long size = n * sizeof(unsigned long long);

    if (n > prop.maxThreadsPerBlock)
    {
        threads = prop.maxThreadsPerBlock;
        block = n / prop.maxThreadsPerBlock;
    }
    else
    {
        threads = n;
        block = 1;
    }
    cudaMalloc((void **)&d_a, size);

    a = (unsigned long long *)malloc(size);

    uint64_t num;
    for (i = 0; i < n; i++)
    {
        num = rand();
        
        a[i] = num;
    }

    printf("\nArray Before Sorting:\n");
    for (j = 0; j < n; ++j)
        printf("%llu\n", a[j]);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);


    int stage;
    int step;
    for (step = 1; step <= steps; step++)
    {
        unsigned long long sl = pow(2, step);
        for (stage = 1; stage <= step; stage++)
        {
            unsigned long long N = sl / (pow(2, stage - 1));
            sort<<<block, threads>>>(d_a, step, stage, sl, N);
        }
    }

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

    printf("\nThe sorted array:\n");
    for (j = 0; j < n; ++j)
        printf("%llu\n", a[j]);

    free(a);
    cudaFree(d_a);
    return 0;
}
