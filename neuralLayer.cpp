#include "neuralLayer.hpp"
#include "neuralException.hpp"

using namespace std;

/**
 * @brief Constructs a new Neural Layer with specified dimensions and activation function.
 * Allocates and initializes internal matrices for weights, biases, activations,
 * inputs, and their respective gradients.
 * @param inputNum Number of input connections entering this layer.
 * @param neuronNum Number of neurons contained within this layer.
 * @param function The activation function type (SIGMOID, RELU).
 */
NeuralLayer::NeuralLayer(int inputNum, int neuronNum, ActivationType function)
    : weights(inputNum, neuronNum, MyMatrix::ZERO), biases(1, neuronNum, MyMatrix::ZERO), lastActivation(1, neuronNum, MyMatrix::ZERO),
      lastInput(1, inputNum, MyMatrix::ZERO), weightGradients(inputNum, neuronNum, MyMatrix::ZERO),
      biasGradients(1, neuronNum, MyMatrix::ZERO), funcType(function), inputCount(inputNum), neuronCount(neuronNum) {
    initializeWeights();
}

/**
 * @brief Destroys the Neural Layer object and cleans up resources.
 */
NeuralLayer::~NeuralLayer() {}

/**
 * @brief Retrieves the total number of input dimensions expected by this layer.
 * @return Count of input connections.
 */
int NeuralLayer::getInputCount() const { return inputCount; }

/**
 * @brief Retrieves the total number of neurons configured within this layer.
 * @return Count of layer neurons.
 */
int NeuralLayer::getNeuronCount() const { return neuronCount; }

/**
 * @brief Computes the forward pass propagation for incoming input data.
 * Evaluates the linear transformation Z = XW + b, where X is the input, W is the
 * weight matrix, and b is the bias vector. Applies the configured activation function
 * and stores states for subsequent backpropagation steps.
 *
 * @param inputs Matrix containing input features across sample batches.
 * @return MyMatrix the activated outputs of this layer.
 * @throws NeuralException if input columns do not match the weight matrix rows.
 */
MyMatrix NeuralLayer::forwardPass(const MyMatrix &inputs) {
    if (inputs.getColumns() != weights.getRows()) {
        throw NeuralException("Incompatible inputs matrix and weights matrix dimensions!");
    }

    lastInput = inputs;

    MyMatrix result = (inputs * weights) + biases;

    lastActivation = (funcType == SIGMOID) ? result.sigmoid() : result.relu();

    return lastActivation;
}

/**
 * @brief Executes backpropagation through the layer to evaluate error gradients.
 * Calculates the entry-wise error delta using the derivative of the activation function,
 * adds up gradients for both biases and weights across the batch, and computes the
 * loss gradient propagated to the preceding layer.
 * @param outputGradient The derivative of the loss function with respect to layer outputs.
 * @return MyMatrix the propagated error gradient with respect to layer inputs.
 */
MyMatrix NeuralLayer::backwardPass(const MyMatrix &outputGradient) {

    MyMatrix activationDerivative = (funcType == SIGMOID) ? lastActivation.sigmoidDerivative() : lastActivation.reluDerivative();

    MyMatrix delta = activationDerivative % outputGradient;

    biasGradients.zero();
    for (int j = 0; j < neuronCount; ++j) {
        double batchSum = 0.0;
        for (int i = 0; i < delta.getRows(); ++i) {
            batchSum += delta.getValue(i, j);
        }
        biasGradients.setValue(0, j, batchSum);
    }

    weightGradients = lastInput.transpose() * delta;

    return delta * weights.transpose();
}

/**
 * @brief Updates parameters via gradient descent using added up derivatives.
 * Adjusts weight and bias matrices using computed gradients scaled by the learning rate.
 * Resets gradients back to zero.
 * @param learningRate Scaling step size factor for optimizing updates.
 */
void NeuralLayer::updateWeightsAndBiases(double learningRate) {
    weightGradients *= learningRate;
    biasGradients *= learningRate;

    weights -= weightGradients;
    biases -= biasGradients;

    weightGradients.zero();
    biasGradients.zero();
}

/**
 * @brief Initializes the weight matrix parameters using bounded uniform scaling.
 * Employs Xavier/Glorot variance bounding principles for Sigmoid activation paths,
 * or a simplified variance configuration for Rectified Linear Unit (ReLU)
 * to stabilize gradient tracking variance.
 */
void NeuralLayer::initializeWeights() {
    double limit = 0.0;

    if (funcType == SIGMOID) {
        limit = sqrt(6.0) / sqrt(inputCount + neuronCount);
    } else {
        limit = sqrt(6.0) / sqrt(inputCount);
    }

    for (int i = 0; i < inputCount; ++i) {
        for (int j = 0; j < neuronCount; ++j) {
            weights.setValue(i, j, MyMatrix::generateRandVal(-limit, limit));
        }
    }
}