#include <iostream>
#include <vector>
#include "../include/mlp.hpp"

int main(){
    NeuralNetwork nn;

// Add layers
    nn.addLayer(2, 4, Activation::relu, Activation::relu_derivative);
    nn.addLayer(4, 1, Activation::sigmoid, Activation::sigmoid_derivative);

// Set the loss function (now with function pointers)
    nn.setLossFunction(LossFunctions::cross_entropy, LossFunctions::cross_entropy_derivative);

// Train as before
    std::vector<Vector> X_train = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    std::vector<Vector> y_train = { {0}, {1}, {1}, {0} };
    nn.fit(X_train, y_train, 2000, 0.1);

// Predict
    for (const auto& x : X_train) {
        Vector y_pred = nn.predict(x);
        std::cout << "Input: ";
        for (const auto& val : x) {
            std::cout << val << " ";
        }
        std::cout << " -> Prediction: ";
        for (const auto& val : y_pred) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
};

