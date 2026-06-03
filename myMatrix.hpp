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

    MyMatrix transpose() const;

    MyMatrix operator+(const MyMatrix &addend) const;
    MyMatrix operator+(double scalar) const;

    MyMatrix operator*(const MyMatrix &factor) const;
    MyMatrix operator*(double scalar) const;

    MyMatrix operator%(const MyMatrix &factor) const;
};