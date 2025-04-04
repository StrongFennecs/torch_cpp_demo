#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h> 

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

torch::Tensor add(const torch::Tensor& a, const torch::Tensor& b) {
    // some torch provided checks
    TORCH_CHECK(a.is_cuda(), "tensor a must be a cuda tensor");
    TORCH_CHECK(b.is_cuda(), "tensor b must be a cuda tensor");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "input tensors must have the same size");

    // make sure to create output on the same device
    // lots of torch functions can be accessed via torch::
    torch::Tensor res = torch::empty_like(a);  

    // use the current stream for torch
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int n = a.numel();
    int threads = 1024;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(), res.data_ptr<float>(), n);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "cuda kernel failed: ", cudaGetErrorString(err));

    return res;
}