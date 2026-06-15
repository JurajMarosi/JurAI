#include "myMatrix.hpp"
#include "neuralException.hpp"
#include "neuralNetwork.hpp"
#include "summaryPrinter.hpp"
#include <iostream>

using namespace std;

int main() {
    try {
        int trainSamples = 30;
        double radius = 0.6;
        double radiusSquared = radius * radius;

        MyMatrix X(trainSamples, 2, MyMatrix::ZERO);
        MyMatrix Y(trainSamples, 1, MyMatrix::ZERO);

        int insideCount = 0;
        for (int i = 0; i < trainSamples; ++i) {
            double x = MyMatrix::generateRandVal();
            double y = MyMatrix::generateRandVal();

            X.setValue(i, 0, x);
            X.setValue(i, 1, y);

            if ((x * x + y * y) <= radiusSquared) {
                Y.setValue(i, 0, 1.0);
                insideCount++;
            } else {
                Y.setValue(i, 0, 0.0);
            }
        }

        NeuralNetwork nn;
        nn.addLayer(2, 12, NeuralLayer::RELU);
        nn.addLayer(12, 1, NeuralLayer::SIGMOID);

        nn.train(X, Y, 100000, 1.2);

        double testPoints[8][2] = {{0.0, 0.0}, {0.2, -0.1}, {0.5, 0.3}, {0.9, 0.9}, {-0.8, 0.2}, {0.0, 0.59}, {0.0, 0.61}, {-0.1, -0.9}};

        MyMatrix testX(8, 2, MyMatrix::ZERO);
        for (int i = 0; i < 8; ++i) {
            testX.setValue(i, 0, testPoints[i][0]);
            testX.setValue(i, 1, testPoints[i][1]);
        }

        MyMatrix testPredictions = nn.predict(testX);

        MyMatrix finalPredictions = nn.predict(X);

        printTable(X, Y, finalPredictions);

    } catch (const NeuralException &e) {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}