#pragma once

#include <functional>
#include <vector>

class MyMatrix {
  private:
    int rows;
    int columns;
    std::vector<double> values;

    int getIndex(int row, int column) const;
    bool isNull() const;

    friend void testNullMatrix();
    friend void testIndexException();

  public:
    enum initMode { RANDOM, ZERO };

    MyMatrix(int rows, int columns, initMode mode);
    ~MyMatrix();

    MyMatrix(const MyMatrix &other) = default;
    MyMatrix &operator=(const MyMatrix &other) = default;

    MyMatrix(MyMatrix &&other) noexcept = default;
    MyMatrix &operator=(MyMatrix &&other) noexcept = default;

    static double generateRandVal(double lowBound = -1.0, double highBound = 1.0);

    void print() const;
    double getValue(int row, int column) const;
    void setValue(int row, int column, double value);
    int getRows() const;
    int getColumns() const;

    MyMatrix transpose() const;
    void zero();

    MyMatrix operator+(const MyMatrix &addend) const;
    MyMatrix operator+(double scalar) const;

    MyMatrix operator-(const MyMatrix &subtrahend) const;
    MyMatrix &operator-=(const MyMatrix &subtrahend);

    MyMatrix operator*(const MyMatrix &factor) const;
    MyMatrix operator*(double scalar) const;
    MyMatrix &operator*=(double scalar);

    MyMatrix operator%(const MyMatrix &factor) const;

    MyMatrix map(std::function<double(double)> func) const;
    MyMatrix sigmoid() const;
    MyMatrix relu() const;
    MyMatrix sigmoidDerivative() const;
    MyMatrix reluDerivative() const;
};