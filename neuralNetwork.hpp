#pragma once

#include "neuralLayer.hpp"

#include <vector>

class NeuralNetwork {
  private:
    std::vector<NeuralLayer> network;

  public:
    NeuralNetwork();
    ~NeuralNetwork();

    void addLayer(int inputNum, int neuronNum, ActivationType function);

    MyMatrix predict(const MyMatrix &inputs);
};