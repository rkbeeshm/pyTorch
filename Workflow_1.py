## PyTorch workflow

import torch

## data (prepare and load)
## build model
## fitting the model to data
## making predicitions and evaluating a model (inference)
## saving and loading a model
## putting it all together

from torch import nn # nn contains all of the building blocks for neural network

import matplotlib.pyplot as plt

## 1. Data (preparing and loading)

# linear regression formula to make a straight line

weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end , step).unsqueeze(dim=1)
Y = weight*X + bias

print(X[:10], Y[:10], len(X))

## splitting data into training and test sets

# Training, validation (not often) and test

train_split = int(0.8 * len(X))
print(train_split)
X_train, y_train = X[:train_split], Y[:train_split]
X_test, y_test = X[train_split:], Y[train_split:]

print(X_test, y_test)
print(len(X_train), len(y_train), len(X_test), len(y_test))
print(X[:10]) ###
              ### Python syntax represents slicing it retrieves all
              ### elements form the begining up to, but not including the element at the index 10

def plot_prediction(train_data=X_train,
                    train_label=y_train,
                    test_data=X_test,
                    test_label=y_test,
                    predicitions=None):
    """
    Plots training data
    """
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_label, c="g", s=4, label="Test data")

    if predicitions is not None:
        plt.scatter(test_data, predicitions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    

#plot_prediction()
#plt.show()

## 2. Build Model
# Create linear regression model class


# what our model does
# Start with random values 
# look at training data and adjust the random values to better
# represent the ideal values

# Through two main algorithims:
    # Gradient descent
    # Backpropagation

class LinearRegressionModel(nn.Module):
    
    def __init__(self):
        super().__init__()

        # Initialize mode parameters
        self.weights = nn.Parameter(torch.rand(1,                      ## start with a random weight and try to adjust it to the ideal weight
                                               requires_grad=True,     ## if the parameter requires gradient
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1,                         ## start with a random weight and try to adjust it to the ideal weight
                                               requires_grad=True,     ## if the parameter requires gradient
                                               dtype=torch.float))
        
    # Forward method to define the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x"  is the input data
        return self.weights * x + self.bias # this is the linear regression


"""
subclass nn.Module
(this contains all the building blocks for neural networks)

initialise model parameters to be used in various
computations (these could be different layers from 
torch.nn, simple parameters, hard-coded values or functions)

requires_grad = True means PyTorch will track the gradient
of this specific parameter for use with torch.autograd and gradient 
descent

"""

### PyTorch model building essentials
# 
# * torch.nn - contains all of the building for computational graphs (a neural
# network can be considered as CG)
# * torch.nn.parameters - what parameter should our model try and learn
# * torch.nn.Module - the base class for all neural network modules , if you subclass it
# you should overwrite forward (we have done it above)
# * torch.optim - this where the optimizers in PyTorch live, they will help
# with gradient descent
# * def forward() - All nn.Module subclasses require you to overwrite forward(),
# this method defines what heppens in the forward computation

# Checking the contents of our PyTorch model

torch.manual_seed(42)

# create an instance of the model (this is a subclass of nn.Module)
model_0 = LinearRegressionModel()

print(list(model_0.parameters()))

# list named parameters
print(model_0.state_dict()) #OrderedDict([('weights', tensor([0.8823])), ('bias', tensor([0.9150]))])

### Making prediction using "torch.inference_mode()"
"To check our models predictive power"

with torch.inference_mode(): 
    y_preds = model_0(X_test)

# you could get the same with just
# y_preds = model_0(X_test)
# but this keeps track off the grad descent term thus prediction would be slow
# helps code run faster

#print(y_preds)
# plot_prediction(predicitions=y_preds)
# plt.show()

# 3. Train our model
# one way to measure how poor the predictons are use loss function
#
# *Loss function*
# *Optimizer*

# Setting up loss function 

loss_fn = nn.L1Loss()

# and optimizer

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01) # lr - learning rate

##
## Building a training loop (and a testing loop)
##
# * loop through the data
# * Forward pass
# * calculate the loss
# * Optimizer zero grad
# * loss backward
# * optimizer step 

# An epoch is one loop through the data

epochs = 10

for epochs in range (epochs):
    # set  the model to training mode
    model_0.train() 
    
    # 1. forward pass
    y_preds = model_0(X_train)

    # 2. calculate the loss
    loss = loss_fn(y_preds, y_train)

    # 3. optimizer zero grad
    optimizer.zero_grad()

    # 4. Perfrom backpropgation on the loss function
    loss.backward()

    # 5. step the optimizer 
    optimizer.step()

    # Testing
    model_0.eval()

    print(model_0.state_dict())

with torch.inference_mode():
    y_preds_new = model_0(X_train)