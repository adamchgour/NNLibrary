#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <vector>
#include <cmath>


namespace Activation {
    double sigmoid(double x){
        return 1.0 / (1.0 + exp(-x));
    }
    double sigmoid_derivative(double x){
        return sigmoid(x) * (1 - sigmoid(x));
    }
    double relu(double x){
        return x > 0 ? x : 0;
    }
    double relu_derivative(double x){
        return x > 0 ? 1 : 0;
    }
    double tanh(double x){
        return std::tanh(x);
    }
    double tanh_derivative(double x){
        return 1 - std::pow(tanh(x), 2);
    }
}


#endif