#include "myMatrix.hpp"
#include "myMatrixException.hpp"
#include <cassert>
#include <iostream>
#include <string>

using namespace std;

struct TestMatrices {
    MyMatrix A;
    MyMatrix B;
    MyMatrix C;

    TestMatrices() : A(2, 3, MyMatrix::ZERO), B(3, 4, MyMatrix::ZERO), C(3, 4, MyMatrix::ZERO) {
        A.setValue(0, 0, 1.0);
        A.setValue(0, 1, 2.0);
        A.setValue(0, 2, 3.0);
        A.setValue(1, 0, 4.0);
        A.setValue(1, 1, 5.0);
        A.setValue(1, 2, 6.0);

        B.setValue(0, 0, 1.0);
        B.setValue(0, 1, 2.0);
        B.setValue(0, 2, 1.0);
        B.setValue(0, 3, 0.0);
        B.setValue(1, 0, 2.0);
        B.setValue(1, 1, 3.0);
        B.setValue(1, 2, 0.0);
        B.setValue(1, 3, 1.0);
        B.setValue(2, 0, 1.0);
        B.setValue(2, 1, 0.0);
        B.setValue(2, 2, 2.0);
        B.setValue(2, 3, 3.0);

        C.setValue(0, 0, 2.0);
        C.setValue(0, 1, 1.0);
        C.setValue(0, 2, 3.0);
        C.setValue(0, 3, 4.0);
        C.setValue(1, 0, 0.0);
        C.setValue(1, 1, 2.0);
        C.setValue(1, 2, 1.0);
        C.setValue(1, 3, 1.0);
        C.setValue(2, 0, 3.0);
        C.setValue(2, 1, 1.0);
        C.setValue(2, 2, 0.0);
        C.setValue(2, 3, 2.0);
    }
};

void testAddition() {
    TestMatrices data;
    MyMatrix result = data.B + data.C;

    assert(result.getValue(0, 0) == 3.0);
    assert(result.getValue(0, 1) == 3.0);
    assert(result.getValue(0, 2) == 4.0);
    assert(result.getValue(0, 3) == 4.0);

    assert(result.getValue(1, 0) == 2.0);
    assert(result.getValue(1, 1) == 5.0);
    assert(result.getValue(1, 2) == 1.0);
    assert(result.getValue(1, 3) == 2.0);

    assert(result.getValue(2, 0) == 4.0);
    assert(result.getValue(2, 1) == 1.0);
    assert(result.getValue(2, 2) == 2.0);
    assert(result.getValue(2, 3) == 5.0);

    cout << "Matrices addition successfully tested!" << endl;
}

void testMultiplication() {
    TestMatrices data;
    MyMatrix result = data.A * data.B;

    assert(result.getValue(0, 0) == 8.0);
    assert(result.getValue(0, 1) == 8.0);
    assert(result.getValue(0, 2) == 7.0);
    assert(result.getValue(0, 3) == 11.0);

    assert(result.getValue(1, 0) == 20.0);
    assert(result.getValue(1, 1) == 23.0);
    assert(result.getValue(1, 2) == 16.0);
    assert(result.getValue(1, 3) == 23.0);

    cout << "Matrices multiplication successfully tested!" << endl;
}

void testHadamardMul() {
    TestMatrices data;
    MyMatrix result = data.B % data.C;

    assert(result.getValue(0, 0) == 2.0);
    assert(result.getValue(0, 1) == 2.0);
    assert(result.getValue(0, 2) == 3.0);
    assert(result.getValue(0, 3) == 0.0);

    assert(result.getValue(1, 0) == 0.0);
    assert(result.getValue(1, 1) == 6.0);
    assert(result.getValue(1, 2) == 0.0);
    assert(result.getValue(1, 3) == 1.0);

    assert(result.getValue(2, 0) == 3.0);
    assert(result.getValue(2, 1) == 0.0);
    assert(result.getValue(2, 2) == 0.0);
    assert(result.getValue(2, 3) == 6.0);

    cout << "Hadamard multiplication successfully tested!" << endl;
}

void testTransposition() {
    TestMatrices data;
    MyMatrix result = data.A.transpose();

    assert(result.getValue(0, 0) == 1.0);
    assert(result.getValue(0, 1) == 4.0);

    assert(result.getValue(1, 0) == 2.0);
    assert(result.getValue(1, 1) == 5.0);

    assert(result.getValue(2, 0) == 3.0);
    assert(result.getValue(2, 1) == 6.0);

    cout << "Matrix transposition successfully tested!" << endl;
}

void testAddException() {
    TestMatrices data;

    bool exceptionThrown = false;

    try {
        MyMatrix reuslt = data.A + data.B;
    } catch (MyMatrixException e) {
        exceptionThrown = true;

        string errorMsg = e.what();
        assert(errorMsg == "The matrices must have the same dimensions for adding!");
    }

    assert(exceptionThrown);

    cout << "Matrices addition exception was thrown successfully!" << endl;
}

void testMulException() {
    TestMatrices data;

    bool exceptionThrown = false;

    try {
        MyMatrix reuslt = data.B * data.C;
    } catch (MyMatrixException e) {
        exceptionThrown = true;

        string errorMsg = e.what();
        assert(errorMsg == "Matrices do not have compatible dimensions for multiplication!");
    }

    assert(exceptionThrown);

    cout << "Matrices multiplicaton exception was thrown successfully!" << endl;
}

void testHadException() {
    TestMatrices data;

    bool exceptionThrown = false;

    try {
        MyMatrix reuslt = data.A % data.C;
    } catch (MyMatrixException e) {
        exceptionThrown = true;

        string errorMsg = e.what();
        assert(errorMsg == "The matrices must have the same dimensions for "
                           "Hadamard multiplication!");
    }

    assert(exceptionThrown);

    cout << "Hadamard multiplication exception was thrown successfully!" << endl;
}

void testNullMatrix() {
    MyMatrix nullMatrix1(0, 1, MyMatrix::ZERO);
    MyMatrix nullMatrix2(1, 0, MyMatrix::ZERO);

    assert(nullMatrix1.isNull());
    assert(nullMatrix2.isNull());

    cout << "Null matrices tested successfully!" << endl;
}

void testIndexException() {
    TestMatrices data;

    bool exceptionThrown = false;

    try {
        data.A.getIndex(3, 3);
    } catch (MyMatrixException e) {
        exceptionThrown = true;

        string errorMsg = e.what();
        assert(errorMsg == "Index out of bounds!");
    }

    assert(exceptionThrown);

    cout << "Index exception was thrown successfully!" << endl;
}

int main() {
    cout << "--- MATRIX UNIT TESTS ---" << endl;

    testAddition();
    testMultiplication();
    testHadamardMul();
    testTransposition();
    cout << endl;

    testAddException();
    testMulException();
    testHadException();
    cout << endl;

    testNullMatrix();
    testIndexException();

    return 0;
}