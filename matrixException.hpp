#include <iostream>
#include <string>

class MatrixException : public std::exception {
  private:
    std::string message;

  public:
    MatrixException(const std::string &msg) : message(msg) {}
    std::string what() { return message; }
};
