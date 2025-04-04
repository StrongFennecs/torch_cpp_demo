## torch cpp demo

An example template repository to get you setup building c++ extensions in pytorch.

### requirements

A machine with a nvidia gpu, and the nvidia toolkit installed for the nvidia cuda compiler nvcc.

### setup

We like using `uv` ([&rarr; installation](https://docs.astral.sh/uv/#installation)) . `uv venv` to create a virtual env, and then run `uv pip install -e .`

### how to call c++ 

To write cuda extension first you need a c++ (for example `csrc/extension.cpp`) file which defines the functions that will be called from python. These c++ functions get bound to python functions with `pybind11`. 

In this example, the functions defined in `extension.cpp` will forward declare functions that are defined in cuda (.cu) files. In this example repo the cuda code in `kernel.cu`. It's good practice for the c++ functions to do some checks and then forward its calls to the cuda functions. The cuda `.cu` contains the actual cuda kernels. 

The pytorch cpp_extension package is used to build the c++ sources with a c++ compiler and the cuda sources with nvcc. 

More detail can be found here:

https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension
