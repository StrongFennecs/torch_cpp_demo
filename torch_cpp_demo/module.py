
import torch
from torch_cpp_demo import extension

def cuda_add(a, b):
    output_tensor = torch.empty_like(a)
    extension.add(a, b, output_tensor)
    return output_tensor

