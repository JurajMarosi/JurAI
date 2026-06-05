#include "neuralLayer.hpp"
#include "neuralException.hpp"

using namespace std;

NeuralLayer::NeuralLayer(int inputNum, int neuronNum, ActivationType function)
    : weights(inputNum, neuronNum, MyMatrix::RANDOM), biases(1, neuronNum, MyMatrix::ZERO), lastActivation(1, neuronNum, MyMatrix::ZERO),
      inputCount(inputNum), neuronCount(neuronNum) {
    funcType = function;
}

NeuralLayer::~NeuralLayer() {}

int NeuralLayer::getInputCount() const { return inputCount; }
int NeuralLayer::getNeuronCount() const { return neuronCount; }

MyMatrix NeuralLayer::forwardPass(const MyMatrix &inputs) {
    if (inputs.getColumns() != weights.getRows()) {
        throw NeuralException("Incompatible inputs matrix and weights matrix dimensions!");
    }

    MyMatrix result = (inputs * weights) + biases;

    lastActivation = funcType == SIGMOID ? result.sigmoid() : result.relu();

    return lastActivation;
}
