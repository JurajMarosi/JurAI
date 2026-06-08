#include "mseLoss.hpp"
#include "neuralException.hpp"

/**
 * @brief Constructs a new Mean Squared Error (MSE) loss function instance.
 */
MSELoss::MSELoss() {}

/**
 * @brief Destroys the MSE loss function instance.
 */
MSELoss::~MSELoss() {}

/**
 * @brief Computes the Mean Squared Error (MSE) loss between the prediction and target matrices.
 * Evaluates the total cost by averaging the squared differences across all corresponding elements.
 * @param predictions Matrix containing the network's predicted output values.
 * @param targets Matrix containing the expected ground truth values.
 * @return The computed scalar MSE loss.
 * @throws NeuralException if the dimensions of the prediction and target matrices do not match perfectly.
 */
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

/**
 * @brief Computes the partial derivative of the Mean Squared Error (MSE) loss with respect to the predictions.
 * Calculates the gradient matrix required for the network backpropagation pass using the analytical derivative:
 * @param predictions Matrix containing the network's predicted output values.
 * @param targets Matrix containing the expected ground truth values.
 * @return Gradient matrix of identical dimensions containing the calculated partial derivatives.
 * @throws NeuralException if the dimensions of the prediction and target matrices do not match perfectly.
 */
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