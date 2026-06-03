#include "myMatrix.hpp"
#include "myMatrixException.hpp"
#include <iostream>

using namespace std;

int main() {
    try {
        MyMatrix testMyMatrix(5, 5, MyMatrix::RANDOM);
        testMyMatrix.print();
        cout << endl;

        MyMatrix addendMyMatrix(5, 5, MyMatrix::RANDOM);
        addendMyMatrix.print();
        cout << endl;

        double scalar = 2.0;

        MyMatrix transMyMatrix = testMyMatrix * addendMyMatrix;
        transMyMatrix.print();

    } catch (MyMatrixException e) {
        cout << e.what() << endl;
    }
    return 0;
}