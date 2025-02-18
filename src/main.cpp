#include <vector>
#include <iostream>
#include "../include/mlp.hpp"
#include "../include/tools.hpp"
#include "../include/layers.hpp"
#include "../include/loss.hpp"
#include "../include/optimizer.hpp"

int main() {
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;  
    Layer layers = Layer(2, 2, Activation::sigmoid, Activation::sigmoid_derivative); 
    MLP mlp(layers); // This part should look like this, but the moment I still need to work on header files

    return 0; 
}