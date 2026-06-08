#pragma once

#include "myMatrix.hpp"
#include "neuralException.hpp"
#include <iostream>

class SummaryPrinter {
  private:
  public:
    SummaryPrinter();
    ~SummaryPrinter();
};

SummaryPrinter::SummaryPrinter() {}

SummaryPrinter::~SummaryPrinter() {}

void printTable(const MyMatrix &inputs, const MyMatrix &targets, const MyMatrix &predictions) {
    if (inputs.getRows() != targets.getRows() || targets.getRows() != predictions.getRows()) {
        throw NeuralException("Summary print: The number of rows in the matrices does not match!");
    }
    if (targets.getColumns() != predictions.getColumns()) {
        throw NeuralException("Summary print: The number of target and prediction columns does not match!");
    }

    int rows = inputs.getRows();
    int inputCols = inputs.getColumns();
    int outputCols = targets.getColumns();

    std::cout << "====================================================================================\n";

    for (int i = 0; i < inputCols; ++i) {
        std::cout << " Input X" << i << " |";
    }
    for (int i = 0; i < outputCols; ++i) {
        std::cout << " Input Y" << i << " |";
    }
    for (int i = 0; i < outputCols; ++i) {
        std::cout << " Prediction P" << i << " |";
    }
    std::cout << " Accuracy\n";

    int totalWidth = (inputCols * 11) + (outputCols * 15) + (outputCols * 15) + 18;
    std::cout << std::string(totalWidth, '-') << std::endl;

    double totalNetworkAccuracy = 0.0;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < inputCols; ++c) {
            printf("   %5.2f   |", inputs.getValue(r, c));
        }
        for (int c = 0; c < outputCols; ++c) {
            printf("     %5.1f      |", targets.getValue(r, c));
        }
        for (int c = 0; c < outputCols; ++c) {
            printf("     %.5f    |", predictions.getValue(r, c));
        }
        double rowAccuracySum = 0.0;
        for (int c = 0; c < outputCols; ++c) {
            double targetVal = targets.getValue(r, c);
            double predVal = predictions.getValue(r, c);

            double error = abs(targetVal - predVal);
            double accuracy = (1.0 - error) * 100.0;
            if (accuracy < 0.0)
                accuracy = 0.0;

            rowAccuracySum += accuracy;
        }
        double rowAvgAccuracy = rowAccuracySum / outputCols;
        totalNetworkAccuracy += rowAvgAccuracy;

        printf("     %6.2f%%\n", rowAvgAccuracy);
    }

    std::cout << std::string(totalWidth, '-') << "\n";
    printf(" OVERALL AVERAGE NEURAL NETWORK ACCURACY: %.2f%%\n", totalNetworkAccuracy / rows);
    std::cout << "====================================================================================\n\n";
}
