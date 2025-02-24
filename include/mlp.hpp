#ifndef MPL_HPP
#define MPL_HPP
#include <vector>
#include <cmath>
#include "matrix.hpp"

typedef std::vector<double> Vector;
typedef std::vector<std::vector<double>> Matrix;

namespace Activation {
    Vector sigmoid(const Vector& x) {
        Vector z(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            z[i] = 1.0 / (1.0 + exp(-x[i]));
        }
        return z;
    }

    Vector sigmoid_derivative(const Vector& x) {
        Vector z = sigmoid(x);
        for (size_t i = 0; i < z.size(); i++) {
            z[i] = z[i] * (1 - z[i]);
        }
        return z;
    }

    Vector relu(const Vector& x) {
        Vector z(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            z[i] = x[i] > 0 ? x[i] : 0;
        }
        return z;
    }

    Vector relu_derivative(const Vector& x) {
        Vector z(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            z[i] = x[i] > 0 ? 1 : 0;
        }
        return z;
    }

    Vector tanh(const Vector& x) {
        Vector z(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            z[i] = std::tanh(x[i]);
        }
        return z;
    }

    Vector tanh_derivative(const Vector& x) {
        Vector z(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            z[i] = 1 - std::pow(std::tanh(x[i]), 2);
        }
        return z;
    }
}

namespace LossFunctions{
    double mseLoss(const Vector& y_true, const Vector& y_pred) {
        double loss = 0.0;
        int size = y_true.size();
        for (int i = 0; i < size; i++) {
            double diff = y_pred[i] - y_true[i];
            loss += diff * diff;
        }
        return loss / size;
    }
    double crossEntropyLoss(const Vector& y_true, const Vector& y_pred) {
        double loss = 0.0;
        int size = y_true.size();
        for (int i = 0; i < size; i++) {
            loss -= y_true[i] * log(y_pred[i] + 1e-9);  // To tackle log(0) problem
        }
        return loss / size;
    }
}

namespace Optimizer{
    void gradientDescent(std::vector<Matrix>& weights,
        std::vector<Vector>& biases,
        std::vector<Matrix>& dW,
        std::vector<Vector>& dB,
        double learning_rate) {
        for (size_t l = 0; l < weights.size(); l++) {
            for (size_t i = 0; i < weights[l].size(); i++) {
                for (size_t j = 0; j < weights[l][i].size(); j++) {
                    weights[l][i][j] -= learning_rate * dW[l][i][j];
                }
                biases[l][i] -= learning_rate * dB[l][i];
            }
        }
    }
}

class Layer {
    public:
        Matrix weights;
        Vector bias;
        Vector A(Vector); // Activation function

        Layer(int in_features, int out_features){
            weights.resize(in_features*out_features);
            bias.resize(out_features);
            
        }        
};

class NeuralNetwork {
    public:
        std::vector<Layer> layers;
        Vector Zs;
        Vector activations;
        double loss(Vector& y_true, Vector& y_pred);
        void Optimizer(std::vector<Matrix>weights,
            std::vector<Vector>& biases,
            std::vector<Matrix>& dW,std::vector<Vector>& dB,
            double learning_rate);

        Vector forwardPass(
            const std::vector<Matrix>& weights,  
            const std::vector<Vector>& biases,   
            const Vector& X,                
            std::vector<Vector>& activations,   
            std::vector<Vector>& Zs              
        ) {
            activations[0] = X;
        
            for (size_t l = 0; l < weights.size(); l++) {
                Vector Z = addVectors(matVecMult(weights[l], activations[l]), biases[l]); // Z^(l) = W^(l) * A^(l-1) + b^(l)
                Zs[l] = Z;
                activations[l + 1] = layers[l].A(Z); // A^(l) = σ(Z^(l))
            }
        
            return activations.back(); // Last output
        }

        void backpropagation(
            std::vector<Matrix>& weights,    
            std::vector<Vector>& biases,     
            std::vector<Vector> &activations,
            Vector &activations_derviative(Vector),
            std::vector<Vector>& Zs,          
            Vector& deltaL,              
            std::vector<Matrix>& dW,         
            std::vector<Vector>& dB,
            double learning_rate          
        ) { 
            /* Backpropagation function. Complexity O(len(weights)) */ // TO BE CORRECTED
            int L = weights.size();
            std::vector<Vector> deltas(L);
            deltas[L-1] = deltaL;
        
            // Backpropagation
            for (int l = L - 2; l >= 0; l--) {
                Matrix W_next_T = transpose(weights[l+1]);
                Vector dz = matVecMult(W_next_T, deltas[l+1]);
                Vector d = hadamardProduct(dz, activations_derviative(Zs[l])); // δ^(l) = (W^(l+1)^T * δ^(l+1)) ⊙ σ'(Z^(l)) TO BE MODIFIED
                
                deltas[l] = d; // Store delta
            }
        
            for (int l = 0; l < L; l++) {
                dW[l] = matMult(deltas[l], activations[l]); // dL/dW^(l) = δ^(l) * A^(l-1)^T
                dB[l] = deltas[l]; // dL/dB^(l) = δ^(l)
            }
            Optimizer(weights,biases,dW,dB,learning_rate); // Modify the Weights and biases according to the optimizer method
        }
};


#endif // MPL_HPP