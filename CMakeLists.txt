cmake_minimum_required(VERSION 3.15)
project(NeuralNetCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MAIN_FILES
    mseLoss.cpp
    mseLoss.hpp
    myMatrix.cpp
    myMatrix.hpp
    neuralLayer.cpp
    neuralLayer.hpp
    neuralNetwork.cpp
    neuralNetwork.hpp
    summaryPrinter.hpp
    neuralException.hpp
)

add_executable(NeuralNetExampleOne
    ${MAIN_FILES} 
    exampleOne.cpp
)

add_executable(NeuralNetExampleTwo
    ${MAIN_FILES} 
    exampleTwo.cpp
)