#include "neuralLayer.hpp"
#include "neuralException.hpp"

using namespace std;

NeuralLayer::NeuralLayer(int inputNum, int neuronNum, ActivationType function)
    : weights(inputNum, neuronNum, MyMatrix::RANDOM), biases(1, neuronNum, MyMatrix::ZERO), lastActivation(1, neuronNum, MyMatrix::ZERO),
      lastInput(1, inputNum, MyMatrix::ZERO), weightGradients(inputNum, neuronNum, MyMatrix::ZERO),
      biasGradients(1, neuronNum, MyMatrix::ZERO), funcType(function), inputCount(inputNum), neuronCount(neuronNum) {}

NeuralLayer::~NeuralLayer() {}

int NeuralLayer::getInputCount() const { return inputCount; }
int NeuralLayer::getNeuronCount() const { return neuronCount; }

MyMatrix NeuralLayer::forwardPass(const MyMatrix &inputs) {
    if (inputs.getColumns() != weights.getRows()) {
        throw NeuralException("Incompatible inputs matrix and weights matrix dimensions!");
    }

    lastInput = inputs;

    MyMatrix result = (inputs * weights) + biases;

    lastActivation = (funcType == SIGMOID) ? result.sigmoid() : result.relu();

    return lastActivation;
}

MyMatrix NeuralLayer::backwardPass(const MyMatrix &outputGradient) {

    MyMatrix activationDerivative = (funcType == SIGMOID) ? lastActivation.sigmoidDerivative() : lastActivation.reluDerivative();

    MyMatrix delta = activationDerivative % outputGradient;

    biasGradients = delta;
    weightGradients = lastInput.transpose() * delta;

    return delta * weights.transpose();
}

void NeuralLayer::updateWeightsAndBiases(double learningRate) {
    weights = weights - (weightGradients * learningRate);
    biases = biases - (biasGradients * learningRate);

    weightGradients.zero();
    biasGradients.zero();
}
