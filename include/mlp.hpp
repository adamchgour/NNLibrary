#ifndef MPL_HPP
#define MPL_HPP

#include <vector>
#include "layers.hpp"

typedef std::vector<Neuron> Layer;

class Neuron{
    public:
    Neuron();

    private:
    double output;
    std::vector<double> output_weights;
    double bias;

};
class MLP {
    public:
        MLP(const std::vector<unsigned>& architecture); // architecture = [input_size, hidden1, hidden2, ..., output_size]
        std::vector<double> feed_forward(const std::vector<double>& inputs); // Forward pass initialization
        void backward(const std::vector<double>& inputs, const std::vector<double>& targets); // Backward pass initialization
        double predict(const std::vector<double>& inputs); // Predict the output of the network
    private:
        std::vector<double> layers; // layers[layer_i][neuron_j]
    };
#endif // MPL_HPP