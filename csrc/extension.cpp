#include <torch/extension.h>

// cuda forward declarations
// its easier to return a new type like tensor rather than some ptr w pybind
// stuff to watch out for: this must match .cu extensions exactly, otherwise 
// you get an undefined symbol error
torch::Tensor add(const torch::Tensor& a, const torch::Tensor& b);

// binding the function to python
// can bind other non torch stuff here
// if you just want torch only extensions
// TORCH_LIBRARY is an alternative
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "A function that adds two tensors");
}