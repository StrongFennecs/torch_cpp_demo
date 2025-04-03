#include <torch/extension.h>
#include <vector>

torch::Tensor add(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    return add(a, b, c);
}

// Binding the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "A function that adds two tensors");
}