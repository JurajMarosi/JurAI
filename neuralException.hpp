#pragma once

#include <iostream>
#include <string>

class NeuralException : public std::exception {
  private:
    std::string message;

  public:
    NeuralException(const std::string &msg) : message(msg) {}
    std::string what() { return message; }
};
