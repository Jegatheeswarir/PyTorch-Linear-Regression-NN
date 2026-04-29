# PyTorch-Linear-Regression-NN
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions.
In this experiment, a neural network model is introduced with a single linear layer that learns the parameters (weight and bias) using gradient descent.

## Neural Network Model
A simple feedforward neural network with one linear layer is used:
```
Input (X) → Linear Layer → Output (y)
```
## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values from 1 to 50 and add random noise to introduce variations in output values.

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using `torch.nn.Linear()` and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, and update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict for a new input value.
## PROGRAM DETAILS

**Name:** JEGATHEESWARI R
**Register Number:** 212223230092

---

## REQUIRED LIBRARIES

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
```

---

## PROGRAM

```python
import torch
import torch.nn as nn  # Neural network module
import matplotlib.pyplot as plt  # For plotting

# Generate Input Data (X) and Output Data (y)
torch.manual_seed(71)  # For reproducibility
X = torch.linspace(1, 50, 50).reshape(-1, 1)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2 * X + 1 + e

# Plot the original data
plt.scatter(X, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()

# Define the Linear Model Class
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# Initialize the Model
torch.manual_seed(59)  # Ensure same initial weights
model = Model(1, 1)

# Print Initial Weights and Bias
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("\nName: JEGATHEESWARI R")
print("Register No: 212223230092")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n')

# Define Loss Function & Optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train the Model
epochs = 100
losses = []

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_function(y_pred, y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')

# Plot Loss Curve
plt.plot(range(epochs), losses, color='blue')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.show()

# Final Weights & Bias
final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()
print("\nName: JEGATHEESWARI R")
print("Register No: 212223230092")
print(f'\nFinal Weight: {final_weight:.8f}, Final Bias: {final_bias:.8f}')

# Best-Fit Line Calculation
x1 = torch.tensor([X.min().item(), X.max().item()])
y1 = x1 * final_weight + final_bias

# Plot Original Data & Best-Fit Line
plt.scatter(X, y, label="Original Data")
plt.plot(x1, y1, 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()

# Prediction for x = 120
x_new = torch.tensor([[120.0]])  # New input as a tensor
y_new_pred = model(x_new).item()  # Predict using trained model
print("\nName: JEGATHEESWARI R")
print("Register No: 212223230092")
print(f"\nPrediction for x = 120: {y_new_pred:.8f}")
```


## OUTPUTS

### 1. Generated Dataset Plot

<img width="596" height="423" alt="image" src="https://github.com/user-attachments/assets/a20ec613-883d-4eb0-8ed3-6b7f47387a87" />


### 2. Training Process
<img width="666" height="433" alt="image" src="https://github.com/user-attachments/assets/85888ca1-ccff-469f-afde-048bb50755b0" />


### 3. Loss Curve

<img width="633" height="391" alt="image" src="https://github.com/user-attachments/assets/ee4a3cd7-ff5c-4a9e-b266-d78027a9e464" />


### 4. Best-Fit Line


### 5. Prediction

<img width="317" height="85" alt="image" src="https://github.com/user-attachments/assets/01a2d062-aafd-45b2-ab5f-678561ba6513" />


## RESULT

Thus, a neural network regression model was successfully developed and trained using PyTorch.
