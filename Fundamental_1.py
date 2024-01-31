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

# creating a tensor like
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)