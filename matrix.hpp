#include <vector>

class Matrix {
  private:
    int rows;
    int columns;
    std::vector<double> values;

    int getIndex(int row, int column) const;
    bool isNull() const;

  public:
    enum initMode { RANDOM, ZERO };

    Matrix(int rows, int columns, initMode mode);
    ~Matrix();

    double generateRandVal();

    void print() const;
    double getValue(int row, int column) const;
    void setValue(int row, int column, double value);

    Matrix transpose() const;

    Matrix operator+(const Matrix &addend) const;
    Matrix operator+(double scalar) const;

    Matrix operator*(const Matrix &factor) const;
    Matrix operator*(double scalar) const;

    Matrix operator%(const Matrix &factor) const;
};