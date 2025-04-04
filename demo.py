import torch 
import torch_cpp_demo
print("adding two 2x2 identity matrices, a, b")

device =  torch.device("cuda:0")
a = torch.eye(2, device="cuda:0", dtype=torch.float32)
b = torch.eye(2, device="cuda:0", dtype=torch.float32)

print(f"a is on {a.device} and b is on {b.device}")
res = torch_cpp_demo.custom_add(a,b)
print(f"result: {res}")