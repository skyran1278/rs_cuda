#include <iostream>

__global__ void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > n)
        return;

    y[index] = x[index] + y[index];
}

int main(void)
{
    int N = 1000000;
    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}
