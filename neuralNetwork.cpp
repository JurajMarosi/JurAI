#include "neuralNetwork.hpp"
#include "mseLoss.hpp"
#include "neuralException.hpp"

using namespace std;

/**
 * @brief Constructs an empty Neural Network instance.
 */
NeuralNetwork::NeuralNetwork() {}

/**
 * @brief Destroys the Neural Network object and cleans up allocated resources.
 */
NeuralNetwork::~NeuralNetwork() {}

/**
 * @brief Appends a new neural layer to the network.
 * @param inputNum Number of input connetions entering this layer.
 * @param neuronNum Number of neurons contained within this layer.
 * @param function The activation function type (SIGMOID, RELU).
 * @throws NeuralException if the layer's input configuration mismatches the preceding layer's output count.
 */
void NeuralNetwork::addLayer(int inputNum, int neuronNum, NeuralLayer::ActivationType function) {
    if (!network.empty() && network.back().getNeuronCount() != inputNum) {
        throw NeuralException("New network layer does not follow the previous layer!");
    }

    network.push_back(NeuralLayer(inputNum, neuronNum, function));
}

/**
 * @brief Performs forward propagation through all layers to generate a prediction.
 * @param inputs Matrix containing input features across sample batches.
 * @return MyMatrix final activation states of the network's output layer.
 * @throws NeuralException if the neural network does not contain any layers.
 */
MyMatrix NeuralNetwork::predict(const MyMatrix &inputs) {
    if (network.empty()) {
        throw NeuralException("Neural network does not have any layers!");
    }

    MyMatrix currentLayerOutput = inputs;

    for (NeuralLayer &l : network) {
        currentLayerOutput = l.forwardPass(currentLayerOutput);
    }

    return currentLayerOutput;
}

/**
 * @brief Performs backpropagation through the network layers in reverse order.
 * @param lossDerivative The derivative of the loss function evaluated with respect to the output layer's activations.
 */
void NeuralNetwork::correct(const MyMatrix &lossDerivative) {
    MyMatrix currentGradient = lossDerivative;

    for (size_t i = network.size(); i > 0; i--) {
        currentGradient = network[i - 1].backwardPass(currentGradient);
    }
}

/**
 * @brief Instructs every layer in the network to update its weights and biases.
 * @param learningRate Optimization step size scale factor.
 */
void NeuralNetwork::learn(double learningRate) {
    for (size_t i = 0; i < network.size(); ++i) {
        network[i].updateWeightsAndBiases(learningRate);
    }
}

/**
 * @brief Trains the model using batch gradient descent with Mean Squared Error (MSE) loss.
 * Runs an optimization loop executing the full pipeline: forward pass evaluation,
 * loss computation, backpropagation gradient correction, and weight/bias adjustments.
 * @param inputs Complete training feature matrix.
 * @param targets Corresponding ground truth label matrix.
 * @param loops The total number of training epochs (iterations) to perform.
 * @param learningRate Gradient descent learning rate optimization scalar.
 */
void NeuralNetwork::train(const MyMatrix &inputs, const MyMatrix &targets, int loops, double learningRate) {
    MSELoss loss;

    for (int loop = 1; loop <= loops; ++loop) {

        MyMatrix finalOutput = predict(inputs);

        double currentLoss = loss.calculate(finalOutput, targets);

        MyMatrix lossGradient = loss.calculateDerivative(finalOutput, targets);

        correct(lossGradient);

        learn(learningRate);
    }
}