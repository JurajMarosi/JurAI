#include "myMatrix.hpp"
#include "neuralException.hpp"
#include "neuralNetwork.hpp"
#include <cassert>
#include <iostream>

using namespace std;

void testEmptyNetException() {
    NeuralNetwork nn;
    MyMatrix inputs(1, 3, MyMatrix::ZERO);

    bool exceptionThrown = false;

    try {
        nn.predict(inputs);
        assert(false);
    } catch (NeuralException &e) {
        exceptionThrown = true;

        string errorMsg = e.what();
        assert(errorMsg == "Neural network does not have any layers!");
    }

    assert(exceptionThrown);

    cout << "Empty neural network exception was thrown successfully!" << endl;
}

void testLayerCompException() {
    NeuralNetwork nn;

    nn.addLayer(3, 5, RELU);

    bool exceptionThrown = false;

    try {
        nn.addLayer(4, 2, SIGMOID);
        assert(false);
    } catch (NeuralException &e) {
        exceptionThrown = true;

        string errorMsg = e.what();
        assert(errorMsg == "New network layer does not follow the previous layer!");
    }

    assert(exceptionThrown);

    cout << "Layer compatibility exception was thrown successfully!" << endl;
}

void testPredict() {
    NeuralNetwork nn;

    nn.addLayer(3, 5, RELU);
    nn.addLayer(5, 2, SIGMOID);

    MyMatrix inputs(1, 3, MyMatrix::ZERO);
    inputs.setValue(0, 0, 1.0);
    inputs.setValue(0, 1, 0.5);
    inputs.setValue(0, 2, -1.0);

    MyMatrix finalOutput = nn.predict(inputs);

    assert(finalOutput.getRows() == 1);
    assert(finalOutput.getColumns() == 2);

    for (int j = 0; j < finalOutput.getColumns(); ++j) {
        double val = finalOutput.getValue(0, j);
        assert(val >= 0.0 && val <= 1.0);
    }

    cout << "Neural network prediciton works successfully!" << endl;
}

int main() {
    cout << "--- NEURAL NETWORK UNIT TESTS ---" << endl;

    testEmptyNetException();
    testLayerCompException();
    testPredict();

    return 0;
}