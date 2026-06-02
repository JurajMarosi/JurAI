#include "matrix.hpp"
#include <iostream>

using namespace std;

int main() {
    Matrix *testMatrix = new Matrix(5, 5, Matrix::RANDOM);
    testMatrix->printMatrix();
    cout << '\n';
    cout << testMatrix->getMatrixValue(2, 2) << '\n';

    testMatrix->setMatrixValue(2, 2, 0.35);

    testMatrix->printMatrix();
    cout << '\n';
    cout << testMatrix->getMatrixValue(2, 2) << '\n';
    return 0;
}