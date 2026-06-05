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

    double generateRandVal();

    void print() const;
    double getValue(int row, int column) const;
    void setValue(int row, int column, double value);
    int getRows() const;
    int getColumns() const;

    MyMatrix transpose() const;

    MyMatrix operator+(const MyMatrix &addend) const;
    MyMatrix operator+(double scalar) const;

    MyMatrix operator*(const MyMatrix &factor) const;
    MyMatrix operator*(double scalar) const;

    MyMatrix operator%(const MyMatrix &factor) const;

    MyMatrix map(std::function<double(double)> func) const;
    MyMatrix sigmoid() const;
    MyMatrix relu() const;
};