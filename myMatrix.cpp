#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include "myMatrix.hpp"
#include "neuralException.hpp"

using namespace std;

MyMatrix::MyMatrix(int r, int c, initMode mode) : rows(r), columns(c) {
    values.resize(rows * columns, 0.0);
    if (mode == RANDOM) {
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = generateRandVal();
        }
    }
}

MyMatrix::~MyMatrix() {}

void MyMatrix::print() const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            int index = i * columns + j;
            cout << values[index] << ' ';
        }
        cout << '\n';
    }
}

double MyMatrix::generateRandVal(double lowBound, double highBound) {
    thread_local std::random_device rd;
    thread_local std::mt19937 re(rd());

    std::uniform_real_distribution<double> unif(lowBound, highBound);
    return unif(re);
}

double MyMatrix::getValue(int row, int column) const { return values[getIndex(row, column)]; }

void MyMatrix::setValue(int row, int column, double value) { values[getIndex(row, column)] = value; }

int MyMatrix::getRows() const { return rows; }

int MyMatrix::getColumns() const { return columns; }

int MyMatrix::getIndex(int row, int column) const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }
    if (row >= rows || row < 0 || column >= columns || column < 0) {
        throw NeuralException("Index out of bounds!");
    }
    return (row * columns) + column;
}

bool MyMatrix::isNull() const {
    if (rows <= 0 || columns <= 0) {
        return true;
    }
    return values.empty();
}

MyMatrix MyMatrix::transpose() const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }

    MyMatrix transposedMyMatrix(columns, rows, ZERO);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            int oldIndex = i * columns + j;
            int newIndex = j * rows + i;

            transposedMyMatrix.values[newIndex] = values[oldIndex];
        }
    }
    return transposedMyMatrix;
}

void MyMatrix::zero() {
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = 0.0;
    }
}

MyMatrix MyMatrix::operator+(const MyMatrix &addend) const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }

    if (columns != addend.columns) {
        throw NeuralException("The matrices must have the same number of columns for adding!");
    }

    MyMatrix sumMyMatrix(rows, columns, ZERO);

    if (rows != addend.rows && addend.rows == 1) {
        for (size_t i = 0; i < values.size(); i++) {

            size_t colIndex = i % columns;
            sumMyMatrix.values[i] = values[i] + addend.values[colIndex];
        }
    } else {
        if (rows != addend.rows) {
            throw NeuralException("Matrix dimensions must match completely if broadcasting is not applicable!");
        }

        for (size_t i = 0; i < values.size(); i++) {
            sumMyMatrix.values[i] = values[i] + addend.values[i];
        }
    }

    return sumMyMatrix;
}

MyMatrix MyMatrix::operator+(double scalar) const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }

    MyMatrix sumMyMatrix(rows, columns, ZERO);

    for (size_t i = 0; i < values.size(); i++) {
        sumMyMatrix.values[i] = values[i] + scalar;
    }
    return sumMyMatrix;
}

MyMatrix MyMatrix::operator-(const MyMatrix &subtrahend) const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }

    if (rows != subtrahend.rows || columns != subtrahend.columns) {
        throw NeuralException("The matrices must have the same dimensions for subtraction!");
    }

    MyMatrix diffMyMatrix(rows, columns, ZERO);

    for (size_t i = 0; i < values.size(); i++) {
        diffMyMatrix.values[i] = values[i] - subtrahend.values[i];
    }
    return diffMyMatrix;
}

MyMatrix &MyMatrix::operator-=(const MyMatrix &subtrahend) {
    if (isNull() || subtrahend.isNull()) {
        throw NeuralException("Cannot subtract NULL matrices!");
    }
    if (rows != subtrahend.rows || columns != subtrahend.columns) {
        throw NeuralException("Dimensions must match perfectly for in-place subtraction!");
    }

    for (size_t i = 0; i < values.size(); i++) {
        values[i] -= subtrahend.values[i];
    }

    return *this;
}

MyMatrix MyMatrix::operator*(const MyMatrix &factor) const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }

    if (columns != factor.rows) {
        throw NeuralException("Matrices do not have compatible dimensions for multiplication!");
    }

    MyMatrix mulMyMatrix(rows, factor.columns, ZERO);

    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < columns; k++) {
            int indexA = i * columns + k;
            double valA = values[indexA];

#pragma omp simd
            for (int j = 0; j < factor.columns; j++) {
                int indexB = k * factor.columns + j;
                int indexResult = i * factor.columns + j;

                mulMyMatrix.values[indexResult] += valA * factor.values[indexB];
            }
        }
    }

    return mulMyMatrix;
}

MyMatrix MyMatrix::operator*(double scalar) const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }

    MyMatrix mulMyMatrix(rows, columns, ZERO);

    for (size_t i = 0; i < values.size(); i++) {
        mulMyMatrix.values[i] = values[i] * scalar;
    }
    return mulMyMatrix;
}

MyMatrix &MyMatrix::operator*=(double scalar) {
    if (isNull()) {
        throw NeuralException("Cannot multiply NULL matrix by scalar!");
    }

    for (size_t i = 0; i < values.size(); i++) {
        values[i] *= scalar;
    }

    return *this;
}

MyMatrix MyMatrix::operator%(const MyMatrix &addend) const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }

    if (rows != addend.rows || columns != addend.columns) {
        throw NeuralException("The matrices must have the same dimensions for "
                              "Hadamard multiplication!");
    }

    MyMatrix mulMyMatrix(rows, columns, ZERO);
    for (size_t i = 0; i < values.size(); i++) {
        mulMyMatrix.values[i] = values[i] * addend.values[i];
    }
    return mulMyMatrix;
}

MyMatrix MyMatrix::map(std::function<double(double)> func) const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }

    MyMatrix resultMatrix(rows, columns, ZERO);

    for (size_t i = 0; i < values.size(); i++) {
        resultMatrix.values[i] = func(values[i]);
    }

    return resultMatrix;
}

MyMatrix MyMatrix::sigmoid() const {
    return map([](double x) { return 1.0 / (1.0 + exp(-x)); });
}

MyMatrix MyMatrix::relu() const {
    return map([](double x) { return x < 0.0 ? 0.0 : x; });
}

MyMatrix MyMatrix::sigmoidDerivative() const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL during sigmoid derivative calculation!");
    }

    return map([](double a) { return a * (1.0 - a); });
}

MyMatrix MyMatrix::reluDerivative() const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL during relu derivative calculation!");
    }

    return map([](double x) { return x > 0.0 ? 1.0 : 0.0; });
}
