#include "neuralNetwork.hpp"
#include "neuralException.hpp"

using namespace std;

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::addLayer(int inputNum, int neuronNum, ActivationType function) {
    if (!network.empty() && network.back().getNeuronCount() != inputNum) {
        throw NeuralException("New network layer does not follow the previous layer!");
    }

    network.push_back(NeuralLayer(inputNum, neuronNum, function));
}

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

void NeuralNetwork::correct(const MyMatrix &lossDerivative) {
    MyMatrix currentGradient = lossDerivative;

    for (size_t i = network.size(); i > 0; i--) {
        currentGradient = network.at(i - 1).backwardPass(currentGradient);
    }
}

void NeuralNetwork::learn(double learningRate) {
    for (size_t i = 0; i < network.size(); ++i) {
        network.at(i).updateWeightsAndBiases(learningRate);
    }
}