#pragma once

#include "neuralLayer.hpp"

#include <vector>

class NeuralNetwork {
  private:
    std::vector<NeuralLayer> network;

  public:
    NeuralNetwork();
    ~NeuralNetwork();

    void addLayer(int inputNum, int neuronNum, NeuralLayer::ActivationType function);

    MyMatrix predict(const MyMatrix &inputs);

    void correct(const MyMatrix &lossDerivative);

    void learn(double learningRate);

    void train(const MyMatrix &inputs, const MyMatrix &targets, int loops, double learningRate);
};