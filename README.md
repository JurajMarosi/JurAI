# C++ Neural Network from Scratch

A zero-dependency feedforward neural network built entirely in modern C++. It does not use any external linear algebra libraries and all matrix operations, forward propagation, and backpropagation are implemented from the ground up.

## Features
* **100% C++ launguage code base**
* **Custom Matrix Logic:** Handles dot products, transposition, element-wise updates, and random weight initialization.
* **Layer-Based Architecture:** Allows stacking hidden layers with flexible configurations.
* **Activations Supported:** ReLU (recommended for hidden layers) and Sigmoid (for output classification).
* **Training:** Uses Stochastic Gradient Descent (SGD) with Mean Squared Error (MSE) loss.
* **Built-in Evaluation:** Includes a helper method to print a detailed breakdown of inputs, targets, predictions, and row-by-row accuracy.

## How it Works

### Activations
* **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
* **ReLU:** $f(x) = \max(0, x)$

### Loss & Optimization
The network calculates error via Mean Squared Error (MSE):
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Gradients are calculated using the chain rule during backpropagation, and weights/biases are updated based on the configured learning rate.

## Project Structure

* `myMatrix.hpp` / `.cpp` - Matrix class, memory allocation, and random value generation.
* `neuralLayer.hpp` / `.cpp` - Individual layer logic (holds weights, biases, and activation states).
* `neuralNetwork.hpp` / `.cpp` - Core network class handling `train()`, `predict()`, and `printSummary()`.
* `summaryPrinter.hpp` - Contains helper method to print table with all the important values and predicitons.
* `neuralException.hpp` - Custom exception handling for matrix dimension mismatches.
* `exampleOne.cpp` - Uses neural network to predict outcomes of XOR operation.
* `exampleTwo.cpp` - Uses neural network to predict wheater is certain random point in a circle.

## Building and Running

### Prerequisites
A C++ compiler supporting C++17 or newer (GCC, Clang, or MSVC).

### Compile
```bash
g++ -std=c++17 exampleOne.cpp myMatrix.cpp neuralLayer.cpp neuralNetwork.cpp mseLoss.cpp -o main
