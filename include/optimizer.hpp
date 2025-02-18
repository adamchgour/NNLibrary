#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP

#include <vector>
#include <cmath>

class Optimizer { // Abstract class for optimizers
public:
    double learning_rate;
    virtual void update(std::vector<std::vector<double>>& weights, 
        std::vector<double>& bias, 
        const std::vector<std::vector<double>>& gradients_w, 
        const std::vector<double>& gradients_b) = 0; // Equivalent to abstract method in Python
virtual ~Optimizer() = default;  // Virtual destructor to avoid memory leaks
};

class SGD : public Optimizer { // Stochastic Gradient Descent
public:
    SGD(double learning_rate);
    void update(std::vector<std::vector<double>>& weights, 
        std::vector<double>& bias, 
        const std::vector<std::vector<double>>& gradients_w, 
        const std::vector<double>& gradients_b) override;
};

#endif