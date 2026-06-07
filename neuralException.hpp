#pragma once

#include <iostream>
#include <string>

class NeuralException : public std::exception {
  private:
    std::string message;

  public:
    NeuralException(const std::string &msg) : message(msg) {}
    const char *what() const noexcept override { return message.c_str(); }
};
