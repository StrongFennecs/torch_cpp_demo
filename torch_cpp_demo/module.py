import torch
# the module defined in setup.py
from extension import add 

def custom_add(a, b):
    # can call your extension
    return add(a, b)


