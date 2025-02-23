#ifndef MPL_HPP
#define MPL_HPP
#include <vector>
#include <cmath>
#include "matrix.hpp"

typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;
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

class Layer {
    public:
        Matrix weights;
        Vector bias;
        double A(double); // Activation function

        Layer(int in_features, int out_features){
            weights.resize(in_features*out_features);
            bias.resize(out_features);
            
        }
        Vector forward(const Vector&x , double A(double) ) { // allow us to calculate the output of a layer.
            Vector y;
            for (int i = 0; i < weights.size(); i++){
                y.push_back(0);
                for (int j = 0; j < x.size(); j++){
                    y[i] += weights[i][j] * x[j];
                }
                y[i] = A(y[i] + bias[i]);
            }
            return y;
        };           
};

class NeuralNetwork {
    public:
        std::vector<Layer> layers;

        void backpropagation(
            std::vector<Matrix>& weights,    
            std::vector<Vector>& biases,     
            std::vector<Vector>& activations,
            std::vector<Vector>& activations_derviative,
            std::vector<Vector>& Zs,          
            Vector& deltaL,              
            std::vector<Matrix>& dW,         
            std::vector<Vector>& dB          
        ) {
            int L = weights.size();
            std::vector<Vector> deltas(L);
            deltas[L-1] = deltaL;
        
            // Backpropagation
            for (int l = L - 2; l >= 0; l--) {
                Matrix W_next_T = transpose(weights[l+1]);
                Vector dz = matVecMult(W_next_T, deltas[l+1]);
                Vector d = hadamardProduct(dz, Zs[l]); // δ^(l) = (W^(l+1)^T * δ^(l+1)) ⊙ σ'(Z^(l))
                
                deltas[l] = d;
            }
        
            for (int l = 0; l < L; l++) {
                dW[l] = matMult(deltas[l], activations[l]); // dL/dW^(l) = δ^(l) * A^(l-1)^T
                dB[l] = deltas[l]; // dL/dB^(l) = δ^(l)
            }
        }
};


#endif // MPL_HPP