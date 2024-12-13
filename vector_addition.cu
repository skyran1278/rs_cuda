#include <iostream>
#include <cuda_runtime.h> // Include the CUDA runtime header

// CUDA kernel to add elements of two arrays
__global__ void add(int *a, int *b, int *c)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

const int arraySize = 10;
__managed__ int a[arraySize], b[arraySize], c[arraySize];

int main()
{

    // Initialize arrays on the host
    for (int i = 0; i < arraySize; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // <<<blocks, threads per block>>>
    // Launch the kernel on the GPU
    add<<<1, arraySize>>>(a, b, c);

    cudaDeviceSynchronize();

    // Display the results
    std::cout << "Array C (Result of A + B): ";
    for (int i = 0; i < arraySize; i++)
    {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // const int arraySize = 10;
    // int a[arraySize], b[arraySize], c[arraySize];

    // // Initialize arrays on the host
    // for (int i = 0; i < arraySize; i++)
    // {
    //     a[i] = i;
    //     b[i] = i * 2;
    // }

    // // Allocate memory on the GPU
    // int *d_a, *d_b, *d_c;
    // cudaMalloc((void **)&d_a, arraySize * sizeof(int));
    // cudaMalloc((void **)&d_b, arraySize * sizeof(int));
    // cudaMalloc((void **)&d_c, arraySize * sizeof(int));

    // // Copy data from host to device
    // cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // // Launch the kernel on the GPU
    // add<<<1, arraySize>>>(d_a, d_b, d_c);

    // // Copy result back to host
    // cudaMemcpy(c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // // Display the results
    // std::cout << "Array C (Result of A + B): ";
    // for (int i = 0; i < arraySize; i++)
    // {
    //     std::cout << c[i] << " ";
    // }
    // std::cout << std::endl;

    // // Free memory on the GPU
    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_c);

    return 0;
}
