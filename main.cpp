#include "myMatrix.hpp"
#include "myMatrixException.hpp"
#include <iostream>

using namespace std;

int main() {
    MyMatrix input(1, 3, MyMatrix::RANDOM);
    MyMatrix weight(3, 2, MyMatrix::RANDOM);
    MyMatrix bias(1, 2, MyMatrix::RANDOM);

    MyMatrix output = (input * weight + bias).sigmoid();

    output.print();

    return 0;
}