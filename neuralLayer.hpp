#pragma once

#include "myMatrix.hpp"

class NeuralLayer {
  private:
    MyMatrix weights;
    MyMatrix biases;
    MyMatrix lastActivation;
    MyMatrix lastInput;

    MyMatrix weightGradients;
    MyMatrix biasGradients;

    int inputCount;
    int neuronCount;

  public:
    enum ActivationType { SIGMOID, RELU };
    ActivationType funcType;

    NeuralLayer(int inputNum, int neuronNum, NeuralLayer::ActivationType function);
    ~NeuralLayer();

    int getInputCount() const;
    int getNeuronCount() const;

    MyMatrix forwardPass(const MyMatrix &inputs);
    MyMatrix backwardPass(const MyMatrix &outputGradient);
    void updateWeightsAndBiases(double learningRate);
    void initializeWeights();
};
