#include "neuralLayer.hpp"
#include "neuralException.hpp"

using namespace std;

NeuralLayer::NeuralLayer(int inputNum, int neuronNum, ActivationType function)
    : weights(inputNum, neuronNum, MyMatrix::ZERO), biases(1, neuronNum, MyMatrix::ZERO), lastActivation(1, neuronNum, MyMatrix::ZERO),
      lastInput(1, inputNum, MyMatrix::ZERO), weightGradients(inputNum, neuronNum, MyMatrix::ZERO),
      biasGradients(1, neuronNum, MyMatrix::ZERO), funcType(function), inputCount(inputNum), neuronCount(neuronNum) {
    initializeWeights();
}

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

void NeuralLayer::updateWeightsAndBiases(double learningRate) {
    weightGradients *= learningRate;
    biasGradients *= learningRate;

    weights -= weightGradients;
    biases -= biasGradients;

    weightGradients.zero();
    biasGradients.zero();
}

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