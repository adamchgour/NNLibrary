#ifndef MPL_HPP
#define MPL_HPP

#include <vector>
#include "layers.hpp"
#include "loss.hpp"


class MLP {
    public:
        std::vector<Layer> layers;
    
        MLP(const std::vector<int>& architecture); // architecture = [input_size, hidden1, hidden2, ..., output_size]
        std::vector<double> forward(const std::vector<double>& inputs); // Forward pass initialization
        void train(const std::vector<std::vector<double>>& X,const std::vector<std::vector<double>>& Y, int epochs, double lr); // Train the model
    };
#endif // MPL_HPP