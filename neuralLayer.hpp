#pragma once

#include "myMatrix.hpp"

enum ActivationType { SIGMOID, RELU };

class NeuralLayer {
  private:
    MyMatrix weights;
    MyMatrix biases;
    MyMatrix lastActivation;
    ActivationType funcType;

    int inputCount;
    int neuronCount;

  public:
    NeuralLayer(int inputNum, int neuronNum, ActivationType function);
    ~NeuralLayer();

    int getInputCount() const;
    int getNeuronCount() const;

    MyMatrix forwardPass(const MyMatrix &inputs);
};
