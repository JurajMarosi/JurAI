#include "myMatrix.hpp"
#include "neuralException.hpp"
#include "neuralNetwork.hpp"
#include "summaryPrinter.hpp"
#include <iostream>

using namespace std;

int main() {
    try {
        MyMatrix inputs(4, 2, MyMatrix::ZERO);
        inputs.setValue(0, 0, 0.0);
        inputs.setValue(0, 1, 0.0);
        inputs.setValue(1, 0, 0.0);
        inputs.setValue(1, 1, 1.0);
        inputs.setValue(2, 0, 1.0);
        inputs.setValue(2, 1, 0.0);
        inputs.setValue(3, 0, 1.0);
        inputs.setValue(3, 1, 1.0);

        MyMatrix targets(4, 1, MyMatrix::ZERO);
        targets.setValue(0, 0, 0.0);
        targets.setValue(1, 0, 1.0);
        targets.setValue(2, 0, 1.0);
        targets.setValue(3, 0, 0.0);

        NeuralNetwork network;
        network.addLayer(2, 4, NeuralLayer::RELU);
        network.addLayer(4, 1, NeuralLayer::SIGMOID);

        network.train(inputs, targets, 10000, 0.8);

        MyMatrix final = network.predict(inputs);

        printTable(inputs, targets, final);

    } catch (const NeuralException &e) {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}