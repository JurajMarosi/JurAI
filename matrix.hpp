#include <vector>

class Matrix {
  private:
    int rows;
    int columns;
    std::vector<double> values;

  public:
    enum initMode { RANDOM, ZERO };

    Matrix(int rows, int columns, initMode mode);
    ~Matrix();
    void printMatrix();
    double generateRandVal();
    double getMatrixValue(int row, int column);
    void setMatrixValue(int row, int column, double value);
};