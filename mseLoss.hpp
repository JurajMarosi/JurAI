#pragma once

#include "myMatrix.hpp"

class MSELoss {
  public:
    MSELoss();
    ~MSELoss();

    double calculate(const MyMatrix &predictions, const MyMatrix &targets);
    MyMatrix calculateDerivative(const MyMatrix &predictions, const MyMatrix &targets);
};