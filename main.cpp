#include "matrix.hpp"
#include "matrixException.hpp"
#include <iostream>

using namespace std;

int main() {
    try {
        Matrix testMatrix(5, 5, Matrix::RANDOM);
        testMatrix.print();
        cout << endl;

        Matrix addendMatrix(5, 5, Matrix::RANDOM);
        addendMatrix.print();
        cout << endl;

        double scalar = 2.0;

        Matrix transMatrix = testMatrix * addendMatrix;
        transMatrix.print();

    } catch (MatrixException e) {
        cout << e.what() << endl;
    }
    return 0;
}