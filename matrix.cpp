#include <iostream>
#include <random>
#include <string>

#include "matrix.hpp"
#include "matrixException.hpp"

using namespace std;

Matrix::Matrix(int r, int c, initMode mode) : rows(r), columns(c) {
    values.resize(rows * columns, 0.0);
    if (mode == RANDOM) {
        for (size_t i = 0; i < values.size(); i++) {
            values.at(i) = generateRandVal();
        }
    }
}

Matrix::~Matrix() {}

void Matrix::print() const {
    if (isNull()) {
        throw MatrixException("Matrix is NULL!");
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            int index = i * columns + j;
            cout << values[index] << ' ';
        }
        cout << '\n';
    }
}

double Matrix::generateRandVal() {
    double lowBound = -1.0;
    double highBound = 1.0;
    static std::random_device rd;
    static std::mt19937 re(rd());
    static std::uniform_real_distribution<double> unif(lowBound, highBound);
    return unif(re);
}

double Matrix::getValue(int row, int column) const {
    return values.at(getIndex(row, column));
}

void Matrix::setValue(int row, int column, double value) {
    values.at(getIndex(row, column)) = value;
}

int Matrix::getIndex(int row, int column) const {
    if (isNull()) {
        throw MatrixException("Matrix is NULL!");
    }
    if (row >= rows || row < 0 || column >= columns || column < 0) {
        throw MatrixException("Index out of bounds!");
    }
    return (row * columns) + column;
}

bool Matrix::isNull() const {
    if (rows <= 0 || columns <= 0) {
        return true;
    }
    return values.empty();
}

Matrix Matrix::transpose() const {
    if (isNull()) {
        throw MatrixException("Matrix is NULL!");
    }

    Matrix transposedMatrix(columns, rows, ZERO);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            int oldIndex = i * columns + j;
            int newIndex = j * rows + i;

            transposedMatrix.values[newIndex] = values[oldIndex];
        }
    }
    return transposedMatrix;
}

Matrix Matrix::operator+(const Matrix &addend) const {
    if (isNull()) {
        throw MatrixException("Matrix is NULL!");
    }

    if (rows != addend.rows || columns != addend.columns) {
        throw MatrixException(
            "The matrices must have the same dimensions for adding!");
    }

    Matrix sumMatrix(rows, columns, ZERO);

    for (size_t i = 0; i < values.size(); i++) {
        sumMatrix.values[i] = values[i] + addend.values[i];
    }
    return sumMatrix;
}

Matrix Matrix::operator+(double scalar) const {
    if (isNull()) {
        throw MatrixException("Matrix is NULL!");
    }

    Matrix sumMatrix(rows, columns, ZERO);

    for (size_t i = 0; i < values.size(); i++) {
        sumMatrix.values[i] = values[i] + scalar;
    }
    return sumMatrix;
}

Matrix Matrix::operator*(const Matrix &factor) const {
    if (isNull()) {
        throw MatrixException("Matrix is NULL!");
    }

    if (columns != factor.rows) {
        throw MatrixException(
            "Matrices do not have compatible dimensions for multiplication!");
    }

    Matrix mulMatrix(rows, factor.columns, ZERO);

    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < columns; k++) {
            int indexA = i * columns + k;
            double valA = values[indexA];

            for (int j = 0; j < factor.columns; j++) {
                int indexB = k * factor.columns + j;
                int indexResult = i * factor.columns + j;

                mulMatrix.values[indexResult] += valA * factor.values[indexB];
            }
        }
    }

    return mulMatrix;
}

Matrix Matrix::operator*(double scalar) const {
    if (isNull()) {
        throw MatrixException("Matrix is NULL!");
    }

    Matrix mulMatrix(rows, columns, ZERO);

    for (size_t i = 0; i < values.size(); i++) {
        mulMatrix.values[i] = values[i] * scalar;
    }
    return mulMatrix;
}

Matrix Matrix::operator%(const Matrix &addend) const {
    if (isNull()) {
        throw MatrixException("Matrix is NULL!");
    }

    if (rows != addend.rows || columns != addend.columns) {
        throw MatrixException("The matrices must have the same dimensions for "
                              "Hadamard multiplication!");
    }

    Matrix mulMatrix(rows, columns, ZERO);
    for (size_t i = 0; i < values.size(); i++) {
        mulMatrix.values[i] = values[i] * addend.values[i];
    }
    return mulMatrix;
}
