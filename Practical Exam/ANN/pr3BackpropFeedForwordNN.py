import numpy as np 
 
# Sigmoid activation function and its derivative 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
def sigmoid_derivative(x): 
    return x * (1 - x) 
 
# Sample Training Data 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs 
y = np.array([[0], [1], [1], [0]])  # Expected Outputs 
 
# Network Architecture 
input_neurons = 2 
hidden_neurons = 2 
output_neurons = 1 
 
# Randomly Initialize Weights and Biases 
np.random.seed(42) 
hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons)) 
hidden_bias = np.random.uniform(size=(1, hidden_neurons)) 
output_weights = np.random.uniform(size=(hidden_neurons, output_neurons)) 
output_bias = np.random.uniform(size=(1, output_neurons)) 
 
# Training Parameters 
learning_rate = 0.5 
epochs = 10000 
 
# Training Loop 
for epoch in range(epochs): 
    # Forward Pass 
    hidden_input = np.dot(X, hidden_weights) + hidden_bias 
    hidden_output = sigmoid(hidden_input) 
    final_input = np.dot(hidden_output, output_weights) + output_bias 
    final_output = sigmoid(final_input) 
 
    # Compute Error 
    error = y - final_output 
     
    # Backpropagation 
    d_output = error * sigmoid_derivative(final_output) 
    error_hidden = d_output.dot(output_weights.T) 
    d_hidden = error_hidden * sigmoid_derivative(hidden_output) 
 
    # Update Weights and Biases 
    output_weights += hidden_output.T.dot(d_output) * learning_rate 
output_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate 
hidden_weights += X.T.dot(d_hidden) * learning_rate 
hidden_bias += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate 
# Testing the trained network 
print("Trained Neural Network Output:") 
hidden_input = np.dot(X, hidden_weights) + hidden_bias 
hidden_output = sigmoid(hidden_input) 
final_input = np.dot(hidden_output, output_weights) + output_bias 
final_output = sigmoid(final_input) 
print(final_output.round()) 