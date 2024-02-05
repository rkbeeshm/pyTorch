import torch

# Indexing 
x = torch.arange(1,10).reshape(1,3,3)
print(x)
print(x[0, 2, 2])

## PyTorch tensors and numpy
# * Data in NumPy, want in PyTorch tensor -> 'torch.from_numpy(ndarray)'
# * PyTorch tensor -> NumPy -> torch.Tensor.numpy()

import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array) # numpy default datatype float 64
tensor = torch.from_numpy(array).type(torch.float32)
print(array, tensor)

##
## Reproducibility of tensor using rand seed ?
##

# to reduce the randomness in  neural network and pytorch comes
# the concept of a "random seed".

# set the random seed
RANDOM_SEED = 42 # most commanly used ?
torch.manual_seed(RANDOM_SEED)
# ^
# |
# |

## Running tensors and PyTorch objects on the GPUs
print(torch.cuda.is_available())

## setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else "CUPA" #works for 

## colab
print(tensor.device)
# https://pytorch.org/docs/stable/notes/cuda.html#best-practices

##
## Putting a tensor (and models) on the GPU
##

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to("cuda") # .to(device)
print(tensor_on_gpu.device)

# Move tensor to CPU
tensor_back_on_cpu = tensor_on_gpu.cpu() # .to("CPU") does not work
print(tensor_back_on_cpu.device)