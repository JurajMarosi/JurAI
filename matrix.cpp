#include <iostream>
#include <random>
#include <string>

#include "matrix.hpp"

using namespace std;

Matrix::Matrix(int r, int c, initMode mode) : rows(r), columns(c) {
    values.resize(rows * columns, 0.0);
    if (mode == RANDOM) {
        for (int i = 0; i < values.size(); i++) {
            values.at(i) = generateRandVal();
        }
    }
}

Matrix::~Matrix() {}

void Matrix::printMatrix() {
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

double Matrix::getMatrixValue(int row, int column) {
    if (row > rows || row < 1 || column > columns || column < 1) {
        cout << "Index out of bounds!" << '\n';
        return -2;
    }
    int index = ((row - 1) * columns) + column - 1;
    return values.at(index);
}

void Matrix::setMatrixValue(int row, int column, double value) {
    if (row > rows || row < 1 || column > columns || column < 1) {
        cout << "Index out of bounds!" << '\n';
        return;
    }
    int index = ((row - 1) * columns) + column - 1;
    values.at(index) = value;
}
