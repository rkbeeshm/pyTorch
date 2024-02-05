import torch

from torch import nn
import matplotlib.pyplot as plt

# Check PyTorch version
print(torch.__version__)

# Create device-agnostic code.
# This means if we've got access to a GPU, our code will
# use it 

# Setup device agnostic code

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

# Create some data using the linear regression formula 
# y = weight * X + bias

weight = 0.1
bias = 0.6

# Create range values
start = 0
end = 1
steps = 0.001

# Create X and Y (features and labels)
X = torch.arange(start, end, steps).unsqueeze(dim=1) # without unsqueeze, errors will pop due to dimensionalty issue
Y = weight * X + bias

# Split data
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[: train_split]
X_test, Y_test = X[train_split:], Y[train_split:]

# Plot prediction function very handy
def plot_prediction(train_data=X_train,
                    train_label=Y_train,
                    test_data=X_test,
                    test_label=Y_test,
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

# Create a liner model
class LinearRegressionModelV2(nn.module):
    def __init__(self):
        super.__init__()

        # use nn.linear() for creating the model parameter
        # also called: linear transform, probing layer, fully connected layer, dense layer
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear_layer(x)
    
# set manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
# Setup device agnostic code for model
model_1.to(device) # Model is on GPU


# Setup a loss function
loss_fn = nn.L1Loss()

# Setup our optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), 
                            lr=0.01)

epochs = 200

# Put data on the target device
# Setup device agnostic code for data
X_test = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)


for epochs in range(epochs):
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate the loss
    loss = loss_fn(y_pred, Y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform back prop
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)

        test_loss = loss_fn(test_pred, Y_train)
    
    if epochs % 10 == 0:
        print(f"Epoch: {epochs} | Loss: {loss} | Test loss: {test_loss}")
