#include <iostream>
#include <string>

class MyMatrixException : public std::exception {
  private:
    std::string message;

  public:
    MyMatrixException(const std::string &msg) : message(msg) {}
    std::string what() { return message; }
};
