#include "neuralNetwork.hpp"
#include "mseLoss.hpp"
#include "neuralException.hpp"

using namespace std;

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::~NeuralNetwork() {}

void NeuralNetwork::addLayer(int inputNum, int neuronNum, NeuralLayer::ActivationType function) {
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
        currentGradient = network[i - 1].backwardPass(currentGradient);
    }
}

void NeuralNetwork::learn(double learningRate) {
    for (size_t i = 0; i < network.size(); ++i) {
        network[i].updateWeightsAndBiases(learningRate);
    }
}

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
