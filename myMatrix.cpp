#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include "myMatrix.hpp"
#include "neuralException.hpp"

using namespace std;

/**
 * @brief Constructs a new MyMatrix object with specified dimensions and initialization mode.
 * If the specified mode is RANDOM, elements are initialized with values generated via uniform distribution.
 * @param r Number of rows.
 * @param c Number of columns.
 * @param mode Initialization strategy (ZERO or RANDOM).
 */
MyMatrix::MyMatrix(int r, int c, initMode mode) : rows(r), columns(c) {
    values.resize(rows * columns, 0.0);
    if (mode == RANDOM) {
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = generateRandVal();
        }
    }
}

/**
 * @brief Destroys the MyMatrix object and frees allocated resources.
 */
MyMatrix::~MyMatrix() {}

/**
 * @brief Prints the matrix elements to the standard output stream.
 * @throws NeuralException if the matrix is uninitialized or has zero dimensions.
 */
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

/**
 * @brief Generates a random floating-point value using a thread-safe engine.
 * @param lowBound The lower bound of the uniform distribution (inclusive).
 * @param highBound The upper bound of the uniform distribution (exclusive).
 * @return A random double precision value.
 */
double MyMatrix::generateRandVal(double lowBound, double highBound) {
    thread_local std::random_device rd;
    thread_local std::mt19937 re(rd());

    std::uniform_real_distribution<double> unif(lowBound, highBound);
    return unif(re);
}

/**
 * @brief Retrieves the value at the specified row and column position.
 * @param row The target row index (0-indexed).
 * @param column The target column index (0-indexed).
 * @return The element value at the specified coordinate.
 * @throws NeuralException if the matrix is uninitialized or position exceed matrix bounds.
 */
double MyMatrix::getValue(int row, int column) const { return values[getIndex(row, column)]; }

/**
 * @brief Updates the element at the specified row and column coordinates.
 * * @param row The target row index (0-indexed).
 * @param column The target column index (0-indexed).
 * @param value The new value to assign to the element.
 * @throws NeuralException If the matrix is uninitialized or indices exceed matrix bounds.
 */
void MyMatrix::setValue(int row, int column, double value) { values[getIndex(row, column)] = value; }

/**
 * @brief Gets the total number of rows in the matrix.
 * * @return Integer representing the row count.
 */
int MyMatrix::getRows() const { return rows; }

/**
 * @brief Gets the total number of columns in the matrix.
 * * @return Integer representing the column count.
 */
int MyMatrix::getColumns() const { return columns; }

/**
 * @brief Validates matrix value position and maps them to a 1D vector index.
 * *@param row The row coordinate.
 * @param column The column coordinate.
 * @return The index within the vector.
 * @throws NeuralException if the matrix is NULL or coordinates fall out of bounds.
 */
int MyMatrix::getIndex(int row, int column) const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL!");
    }
    if (row >= rows || row < 0 || column >= columns || column < 0) {
        throw NeuralException("Index out of bounds!");
    }
    return (row * columns) + column;
}

/**
 * @brief Checks if the matrix is structurally empty or uninitialized.
 * @return true if either dimension is non-positive or internal storage is empty.
 */
bool MyMatrix::isNull() const {
    if (rows <= 0 || columns <= 0) {
        return true;
    }
    return values.empty();
}

/**
 * @brief Creates a new matrix that is the transpose of the current matrix.
 * @return A new MyMatrix object representing the transposed matrix.
 * @throws NeuralException ff the current matrix instance is NULL.
 */
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

/**
 * @brief Fills the entire matrix with 0.0 values.
 */
void MyMatrix::zero() {
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = 0.0;
    }
}

/**
 * @brief Adds another matrix to the current matrix using standard element addition or row broadcasting.
 * If the addend matrix has exactly one row, its columns are broadcasted across all rows of the current matrix.
 * Otherwise, the dimensions of both matrices must match perfectly.
 * @param addend The matrix to be added.
 * @return A new MyMatrix containing the sum.
 * @throws NeuralException if dimensions are mismatched or incompatible for broadcasting.
 */
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

/**
 * @brief Adds a scalar value to every element in the matrix.
 * @param scalar The scalar value to add.
 * @return A new MyMatrix where each element is shifted by the scalar.
 * @throws NeuralException if the matrix is NULL.
 */
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

/**
 * @brief Performs element subtraction between two matrices.
 * @param subtrahend The matrix to subtract from the current matrix.
 * @return A new MyMatrix containing the resulting differences.
 * @throws NeuralException if the matrix dimensions are not identical or if either is NULL.
 */
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

/**
 * @brief Modifies the current matrix by subtracting another matrix in-place.
 * @param subtrahend The matrix to subtract.
 * @return Reference to the current modified instance.
 * @throws NeuralException if dimensions do not match or a matrix is NULL.
 */
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

/**
 * @brief Performs standard matrix dot-product multiplication.
 * Uses SIMD instruction set for vectorization loops.
 * @param factor The right-hand side matrix operand.
 * @return A new MyMatrix containing the matrix product.
 * @throws NeuralException if inner dimensions are incompatible or a matrix is NULL.
 */
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

/**
 * @brief Scales every element of the matrix by a scalar value.
 * @param scalar The multiplication scaling factor.
 * @return A new MyMatrix containing the scaled values.
 * @throws NeuralException if the matrix is NULL.
 */
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

/**
 * @brief Modifies the matrix in-place by scaling each element by a scalar value.
 * @param scalar The multiplication scaling factor.
 * @return Reference to the current modified instance.
 * @throws NeuralException if the matrix is NULL.
 */
MyMatrix &MyMatrix::operator*=(double scalar) {
    if (isNull()) {
        throw NeuralException("Cannot multiply NULL matrix by scalar!");
    }

    for (size_t i = 0; i < values.size(); i++) {
        values[i] *= scalar;
    }

    return *this;
}

/**
 * @brief Computes the element Hadamard multiplication of two matrices.
 * @param addend The matrix factor to multiply element-by-element.
 * @return A new MyMatrix containing the entrywise product.
 * @throws NeuralException if matrix dimensions fail to match completely.
 */
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

/**
 * @brief Transforms each element of the matrix using a user-specified mapping function.
 * @param func A unary function wrapper mapping a double value to a new double value.
 * @return A new transformed MyMatrix instance.
 * @throws NeuralException if the current matrix is NULL.
 */
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

/**
 * @brief Applies the Sigmoid activation function element-wise.
 * @return A new MyMatrix containing the Sigmoid activated states.
 */
MyMatrix MyMatrix::sigmoid() const {
    return map([](double x) { return 1.0 / (1.0 + exp(-x)); });
}

/**
 * @brief Applies the Rectified Linear Unit (ReLU) activation function element-wise.
 * @return A new MyMatrix containing the ReLU activated states.
 */
MyMatrix MyMatrix::relu() const {
    return map([](double x) { return x < 0.0 ? 0.0 : x; });
}

/**
 * @brief Computes the element-wise derivative of the Sigmoid activation function.
 * @return A new MyMatrix containing the calculated derivative values.
 * @throws NeuralException if the matrix is NULL.
 */
MyMatrix MyMatrix::sigmoidDerivative() const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL during sigmoid derivative calculation!");
    }

    return map([](double a) { return a * (1.0 - a); });
}

/**
 * @brief Computes the element-wise derivative of the ReLU activation function.
 * @return A new MyMatrix containing the calculated derivative values.
 * @throws NeuralException if the matrix is NULL.
 */
MyMatrix MyMatrix::reluDerivative() const {
    if (isNull()) {
        throw NeuralException("MyMatrix is NULL during relu derivative calculation!");
    }

    return map([](double x) { return x > 0.0 ? 1.0 : 0.0; });
}