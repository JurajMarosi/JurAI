#include "neuralLayer.hpp"

#include <iostream>

using namespace std;

NeuralLayer::NeuralLayer(int inputNum, int neuronNum, ActivationType function)
    : weights(inputNum, neuronNum, MyMatrix::RANDOM), biases(1, neuronNum, MyMatrix::ZERO), lastActivation(1, neuronNum, MyMatrix::ZERO) {
    funcType = function;
}

NeuralLayer::~NeuralLayer() {}

MyMatrix NeuralLayer::forwardPass(MyMatrix inputs) {
    if (inputs.getColumns() != weights.getRows()) {
        cout << "ZLE DIMENZIE! Vstupy a vahy do seba nezapadaju." << endl;
    }

    MyMatrix result = (inputs * weights) + biases;

    if (funcType == SIGMOID) {
        result = result.sigmoid();
    } else {
        result = result.relu();
    }

    lastActivation = result;

    return result;
}
