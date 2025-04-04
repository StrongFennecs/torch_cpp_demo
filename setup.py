import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_cuda_compute_capability():
    if not torch.cuda.is_available():
        return None
    
    props = torch.cuda.get_device_properties(0)
    return props.major * 10 + props.minor

ext_modules = []

if torch.cuda.is_available():
    compute_capability = get_cuda_compute_capability()
    nvcc_flags = ['-O3']
    
    if compute_capability:
        nvcc_flags.append(f'-gencode=arch=compute_{compute_capability},code=sm_{compute_capability}')
    
    ext_modules.append(
        CUDAExtension(
            name="extension",
            sources=[
                "csrc/extension.cpp",
                "csrc/add.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_flags,
            },
        )
    )

setup(
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)