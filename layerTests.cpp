#include "myMatrix.hpp"
#include "neuralException.hpp"
#include "neuralLayer.hpp"
#include <cassert>
#include <iostream>

using namespace std;

void testLayerInit() {
    NeuralLayer layer(3, 5, RELU);

    assert(layer.getInputCount() == 3);
    assert(layer.getNeuronCount() == 5);

    cout << "Neural Layer constructor and getters are working succesfully!" << endl;
}

void testForwardPass() {
    NeuralLayer layer(3, 2, RELU);

    MyMatrix inputs(1, 3, MyMatrix::ZERO);
    inputs.setValue(0, 0, 5.0);
    inputs.setValue(0, 1, -2.0);
    inputs.setValue(0, 2, 1.5);

    MyMatrix output = layer.forwardPass(inputs);

    assert(output.getRows() == 1);
    assert(output.getColumns() == 2);

    for (int j = 0; j < output.getColumns(); ++j) {
        assert(output.getValue(0, j) >= 0.0);
    }

    cout << "Forward pass is working successfully!" << endl;
}

void testException() {
    NeuralLayer layer(3, 2, RELU);

    MyMatrix incInputs(1, 4, MyMatrix::ZERO);

    bool exceptionThrown = false;

    try {
        layer.forwardPass(incInputs);

        assert(false);
    } catch (NeuralException &e) {
        exceptionThrown = true;

        string errorMsg = e.what();
        assert(errorMsg == "Incompatible inputs matrix and weights matrix dimensions!");
    }

    assert(exceptionThrown);

    cout << "Incompatible inputs exception was thrown successfully!" << endl;
}

int main() {
    cout << "--- NEURAL LAYER UNIT TESTS ---" << endl;

    testLayerInit();
    testForwardPass();
    testException();

    return 0;
}