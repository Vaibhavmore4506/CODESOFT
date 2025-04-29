import numpy as np 
import matplotlib.pyplot as plt 
 
# Sigmoid activation function 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
# ReLU activation function 
def relu(x): 
    return np.maximum(0, x) 
 
# Leaky ReLU activation function 
def leaky_relu(x, alpha=0.01): 
    return np.maximum(alpha * x, x) 
 
# Softmax activation function 
def softmax(x): 
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True) 
 
# Generate input values 
x = np.linspace(-5, 5, 100) 
 
# Compute activation values 
y_sigmoid = sigmoid(x) 
y_relu = relu(x) 
y_leaky_relu = leaky_relu(x) 
 
def plot_activation(x, y, title): 
    plt.plot(x, y, label=title) 
    plt.axhline(0, color='black', linewidth=0.5, linestyle='dashed') 
    plt.axvline(0, color='black', linewidth=0.5, linestyle='dashed') 
    plt.title(title) 
    plt.xlabel('Input') 
    plt.ylabel('Output') 
    plt.legend() 
    plt.grid() 
 
# Plot activation functions 
plt.figure(figsize=(10, 6)) 
 
plt.subplot(2, 2, 1) 
plot_activation(x, y_sigmoid, 'Sigmoid') 
 
plt.subplot(2, 2, 2) 
plot_activation(x, y_relu, 'ReLU') 
 
plt.subplot(2, 2, 3) 
plot_activation(x, y_leaky_relu, 'Leaky ReLU') 
# Softmax requires a different approach since it's multi-dimensional 
x_softmax = np.array([[i] for i in range(-5, 6)])  # 1D input for softmax 
y_softmax = softmax(x_softmax.T).T  # Compute softmax 
plt.subplot(2, 2, 4) 
plt.bar(range(-5, 6), y_softmax.flatten(), label='Softmax') 
plt.title('Softmax') 
plt.xlabel('Input') 
plt.ylabel('Probability') 
plt.legend() 
plt.grid() 
plt.tight_layout() 
plt.show() 