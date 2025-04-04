#include <torch/extension.h>

__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void add(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int n = a.size(0);
    int threads = 1024;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
}