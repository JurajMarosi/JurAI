#pragma once

#include "myMatrix.hpp"

enum ActivationType { SIGMOID, RELU };

class NeuralLayer {
  private:
    MyMatrix weights;
    MyMatrix biases;
    MyMatrix lastActivation;
    ActivationType funcType;

  public:
    NeuralLayer(int inputNum, int neuronNum, ActivationType function);
    ~NeuralLayer();

    MyMatrix forwardPass(MyMatrix inputs);
};
