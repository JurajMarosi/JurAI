#include "mseLoss.hpp"
#include "neuralException.hpp"

MSELoss::MSELoss() {}

MSELoss::~MSELoss() {}

double MSELoss::calculate(const MyMatrix &predictions, const MyMatrix &targets) {
    if (predictions.getRows() != targets.getRows() || predictions.getColumns() != targets.getColumns()) {
        throw NeuralException("Prediction matrix and target matrix dimensions do not match!");
    }

    double sumLoss = 0.0;

    int predictionRows = predictions.getRows();
    int predicitonCol = predictions.getColumns();
    int totalElements = predictionRows * predicitonCol;

    for (int i = 0; i < predictionRows; ++i) {
        for (int j = 0; j < predicitonCol; ++j) {
            double predictionVal = predictions.getValue(i, j);
            double targetVal = targets.getValue(i, j);

            double valDiff = predictionVal - targetVal;
            sumLoss += valDiff * valDiff;
        }
    }

    return sumLoss / totalElements;
}

MyMatrix MSELoss::calculateDerivative(const MyMatrix &predictions, const MyMatrix &targets) {
    if (predictions.getRows() != targets.getRows() || predictions.getColumns() != targets.getColumns()) {
        throw NeuralException("Prediction matrix and target matrix dimensions do not match!");
    }

    int predictionRows = predictions.getRows();
    int predicitonCol = predictions.getColumns();
    int totalElements = predictionRows * predicitonCol;

    MyMatrix lossDerivative(predictionRows, predicitonCol, MyMatrix::ZERO);

    for (int i = 0; i < predictionRows; ++i) {
        for (int j = 0; j < predicitonCol; ++j) {
            double predictionVal = predictions.getValue(i, j);
            double targetVal = targets.getValue(i, j);

            double valDerivative = 2 * (predictionVal - targetVal) / totalElements;
            lossDerivative.setValue(i, j, valDerivative);
        }
    }

    return lossDerivative;
}
