import torch

## introduction to tensors
### creating tensor

# scalar
scalar = torch.tensor(7)

# You could represent an image as a tensor with shape
# [3, 64, 64] -> 3 three color channel (r,g,b) and the height and width

print(scalar)
print(scalar.item())

# vector
vector = torch.tensor([7,7])
print(vector)
print(vector.tolist()) # !scalar use tolist()
print(vector.ndim)
print(vector.shape)

# Matrix
matrix = torch.tensor([[2,3],
                       [2,4]])

print(matrix.ndim)
print(matrix.shape)

# tensor
Tensor = torch.tensor([[[1,2],[2,3],[4,5]]])

print(Tensor.ndim)
print(Tensor.shape)

## scalar and vector should be initailized with lower case
## matrix and tensor should be initailized with upper case
## standard practice research papers and codes!

# random tensors

# random tensor are important because many NN start with some
# random number and adjust it self after iterations

RANDOM_TENSOR = torch.rand(3,4)

print(RANDOM_TENSOR)
print(RANDOM_TENSOR.shape)

# create a random tensor with similar shape to an image
RANDOM_TENSOR_IMAGE_SIZE = torch.rand(size=(3 ,224, 224)) 
# same as torch.rand(3, 224, 224)

print(RANDOM_TENSOR_IMAGE_SIZE)

# creating a tensor of all zeroes

ZEROES = torch.zeros(size=(3, 4))
ONES = torch.ones(size=(3, 4))

# creating a range of tensors and tensors-like
one_to_ten = torch.arange(0, 10)
print(one_to_ten)

# creating a tensor like method
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)

# Tensor  datatypes

#########
######### Note: Tensor datatype is one of the 3 big error you might run into
#########

# Tensors not right type
# Tensors not right shape
# Tensors not on the right device

# default type is float32
dtype_tensor = torch.tensor([3.0, 6.0, 9.0],
                            dtype=None,             # datatype in tensor
                            device=None,            # what device is your on   
                            requires_grad=False)    # whether or not to track gradients with this tensors operation

# to convert defined tensor to another type of datatype

dtype_tensor_16 = dtype_tensor.type(torch.float16)

store = dtype_tensor*dtype_tensor_16

print(store) # this just give float32 type variable

### Getting information from tensor (attributes)
# tensor.dtype
# tensor.device
# tensor.shape

print(dtype_tensor.device) # currently working on CPU it seems!


### Manipulating tensors 

# Tensors operation include:
# * add
# * sub
# * multi
# * div
# * matrix multi

# these are the function a NN does to train it self

t_ensor = torch.tensor([[1, 2, 3],
                        [3, 4, 5]])
print(t_ensor + 10)
print(t_ensor * 10)
print(t_ensor / 2)

# Trying pyTorch in-bulit function
print(torch.mul(t_ensor, 10))
print(torch.add(t_ensor, 10))
print(torch.div(t_ensor, 2))

# Tensor multiplication 
# There two ways in Deep learning and NN
# Element wise (scalar)
# Matrix multiplication 

print(torch.matmul(t_ensor,t_ensor.T)) # tensor.matmul or tensor.mm

# Transpose to fix tensor shape
# syntax

print(t_ensor)
print(t_ensor.mT)

##
## Tensor aggregation 
## Finding min, max, mean, sum, etc
##

t_ensor = torch.arange(0, 100, 10) # default dtype is int64 (be carefull)
# t_ensor = torch.arange(0.0, 100.0, 10)
print(torch.min(t_ensor))
print(torch.max(t_ensor))
print(t_ensor.dtype)
print(torch.mean(t_ensor.type(torch.float16)))
print(t_ensor.sum())
 
## Finding the positional min and max
# find the position (index) in tensor that has min value
print(t_ensor.argmax()) ## useful when would be using the softmax activation function
print(t_ensor[5])

###
### Reshaping, stacking, squeezing and unsqueezing tensors
###

# Reshape - reshapes an input tensor to a defined shape
# view - retusn a view of an input tensor of certain shape but keep the same
# memory as the original tensor
# stacking - combine multiple tensors on top of each (vstack) or side by side (hstack)
# squeeze - removes all '1' dimensions from a tensor
# unsqueeze - add...
# permute - return a view of the input with dimensions permuted (swapped) certain way

x = torch.arange(1., 10.)
print(x, x.shape)
print(x.reshape(1,9))
print(x.reshape(9,1))
print(x.reshape(3,3))

# change the view
# some sought of value projection value are same but shape could change
z = x.view(1,9)
print(x, z)
z[:, 0] = 16
print(x, z)

# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)
x_stacked = torch.stack([x, x, x, x], dim=1)
print(x_stacked)
x_stacked = torch.hstack([x, x, x, x]) # dimension remains the same
print(x_stacked) # nice vstack would?
x_stacked = torch.vstack([x, x, x, x])
print(x_stacked) # x_stacked = torch.stack([x, x, x, x], dim=0)!

# Squeeze tensor
x = torch.ones(2,3,4,1,4,1,1,3)
x_squeeze = torch.squeeze(x)
print(x_squeeze.shape)
x_unsqueeze = x_squeeze.unsqueeze(0)
print(x_unsqueeze.shape)

# permute rearranges the dimensions of a target tensor in a specified order
# move dimension
# B = x.permute(2, 4 ,3 ,1)

#before
# dimension | 1 | 2 | 3 | 4 |
# size      | 5 | 4 | 3 | 2 |

#after
# dimension | 1 | 2 | 3 | 4 |
# size      | 4 | 2 | 3 | 5 |

# move dimesion 2 of x to first subscript position of B,
# dimension 4 to second sub-script, and so on. 

x_original = torch.rand(size=(224, 224, 3)) 
# [height , width , color channel]

# permute the original tensor to rearrange the structure
x_permuted = torch.permute(x_original, (2, 0, 1))
print(x_permuted.shape)