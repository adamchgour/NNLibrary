#include <vector>
#include <iostream>
#include "../include/mlp.hpp"

typedef std::vector<Neuron> Layer;


MLP::MLP(const std::vector<unsigned>& architecture) {
    unsigned num_layers = architecture.size(); // Number of layers in the network
    for (unsigned i = 0; i < num_layers; i++) {
        layers.push_back(Layer());
        unsigned num_neurons = architecture[i]; // Number of neurons in the current layer
        for (unsigned j = 0; j < num_neurons; j++) {
            layers.back().push_back(Neuron(numOutputs));
        }
    }

}

int main() {
    
    return 0; 
}