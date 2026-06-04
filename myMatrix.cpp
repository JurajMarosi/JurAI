#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include "myMatrix.hpp"
#include "myMatrixException.hpp"

using namespace std;

MyMatrix::MyMatrix(int r, int c, initMode mode) : rows(r), columns(c) {
    values.resize(rows * columns, 0.0);
    if (mode == RANDOM) {
        for (size_t i = 0; i < values.size(); i++) {
            values.at(i) = generateRandVal();
        }
    }
}

MyMatrix::~MyMatrix() {}

void MyMatrix::print() const {
    if (isNull()) {
        throw MyMatrixException("MyMatrix is NULL!");
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            int index = i * columns + j;
            cout << values[index] << ' ';
        }
        cout << '\n';
    }
}

double MyMatrix::generateRandVal() {
    double lowBound = -1.0;
    double highBound = 1.0;
    static std::random_device rd;
    static std::mt19937 re(rd());
    static std::uniform_real_distribution<double> unif(lowBound, highBound);
    return unif(re);
}

double MyMatrix::getValue(int row, int column) const { return values.at(getIndex(row, column)); }

void MyMatrix::setValue(int row, int column, double value) { values.at(getIndex(row, column)) = value; }

int MyMatrix::getRows() { return rows; }

int MyMatrix::getColumns() { return columns; }

int MyMatrix::getIndex(int row, int column) const {
    if (isNull()) {
        throw MyMatrixException("MyMatrix is NULL!");
    }
    if (row >= rows || row < 0 || column >= columns || column < 0) {
        throw MyMatrixException("Index out of bounds!");
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
        throw MyMatrixException("MyMatrix is NULL!");
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

MyMatrix MyMatrix::operator+(const MyMatrix &addend) const {
    if (isNull()) {
        throw MyMatrixException("MyMatrix is NULL!");
    }

    if (rows != addend.rows || columns != addend.columns) {
        throw MyMatrixException("The matrices must have the same dimensions for adding!");
    }

    MyMatrix sumMyMatrix(rows, columns, ZERO);

    for (size_t i = 0; i < values.size(); i++) {
        sumMyMatrix.values[i] = values[i] + addend.values[i];
    }
    return sumMyMatrix;
}

MyMatrix MyMatrix::operator+(double scalar) const {
    if (isNull()) {
        throw MyMatrixException("MyMatrix is NULL!");
    }

    MyMatrix sumMyMatrix(rows, columns, ZERO);

    for (size_t i = 0; i < values.size(); i++) {
        sumMyMatrix.values[i] = values[i] + scalar;
    }
    return sumMyMatrix;
}

MyMatrix MyMatrix::operator*(const MyMatrix &factor) const {
    if (isNull()) {
        throw MyMatrixException("MyMatrix is NULL!");
    }

    if (columns != factor.rows) {
        throw MyMatrixException("Matrices do not have compatible dimensions for multiplication!");
    }

    MyMatrix mulMyMatrix(rows, factor.columns, ZERO);

    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < columns; k++) {
            int indexA = i * columns + k;
            double valA = values[indexA];

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
        throw MyMatrixException("MyMatrix is NULL!");
    }

    MyMatrix mulMyMatrix(rows, columns, ZERO);

    for (size_t i = 0; i < values.size(); i++) {
        mulMyMatrix.values[i] = values[i] * scalar;
    }
    return mulMyMatrix;
}

MyMatrix MyMatrix::operator%(const MyMatrix &addend) const {
    if (isNull()) {
        throw MyMatrixException("MyMatrix is NULL!");
    }

    if (rows != addend.rows || columns != addend.columns) {
        throw MyMatrixException("The matrices must have the same dimensions for "
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
        throw MyMatrixException("MyMatrix is NULL!");
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
