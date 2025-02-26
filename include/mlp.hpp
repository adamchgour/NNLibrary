#ifndef MPL_HPP
#define MPL_HPP
#include <vector>
#include <cmath>
#include <random>
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

namespace LossFunctions {
    // MSE Loss
    double mse(const Vector& y_true, const Vector& y_pred) {
        double loss = 0.0;
        int size = y_true.size();
        for (int i = 0; i < size; i++) {
            double diff = y_pred[i] - y_true[i];
            loss += diff * diff;
        }
        return loss / size;
    }
    
    Vector mse_derivative(const Vector& y_true, const Vector& y_pred) {
        int size = y_true.size();
        Vector derivative(size);
        for (int i = 0; i < size; i++) {
            derivative[i] = 2 * (y_pred[i] - y_true[i]) / size;
        }
        return derivative;
    }
    
    // Cross-Entropy Loss
    double cross_entropy(const Vector& y_true, const Vector& y_pred) {
        double loss = 0.0;
        int size = y_true.size();
        for (int i = 0; i < size; i++) {
            loss -= y_true[i] * log(y_pred[i] + 1e-9);  // To tackle log(0) problem
        }
        return loss / size;
    }
    
    Vector cross_entropy_derivative(const Vector& y_true, const Vector& y_pred) {
        int size = y_true.size();
        Vector derivative(size);
        for (int i = 0; i < size; i++) {
            derivative[i] = -y_true[i] / (y_pred[i] + 1e-9) / size;
        }
        return derivative;
    }
    
    // Binary Cross-Entropy Loss
    double binary_cross_entropy(const Vector& y_true, const Vector& y_pred) {
        double loss = 0.0;
        int size = y_true.size();
        for (int i = 0; i < size; i++) {
            loss -= y_true[i] * log(y_pred[i] + 1e-9) + (1 - y_true[i]) * log(1 - y_pred[i] + 1e-9);
        }
        return loss / size;
    }
    
    Vector binary_cross_entropy_derivative(const Vector& y_true, const Vector& y_pred) {
        int size = y_true.size();
        Vector derivative(size);
        for (int i = 0; i < size; i++) {
            derivative[i] = (-y_true[i] / (y_pred[i] + 1e-9) + (1 - y_true[i]) / (1 - y_pred[i] + 1e-9)) / size;
        }
        return derivative;
    }
    
    // Hinge Loss (for SVM-like networks)
    double hinge(const Vector& y_true, const Vector& y_pred) {
        double loss = 0.0;
        int size = y_true.size();
        for (int i = 0; i < size; i++) {
            // Assuming y_true is -1 or 1
            loss += std::max(0.0, 1.0 - y_true[i] * y_pred[i]);
        }
        return loss / size;
    }
    
    Vector hinge_derivative(const Vector& y_true, const Vector& y_pred) {
        int size = y_true.size();
        Vector derivative(size);
        for (int i = 0; i < size; i++) {
            // Assuming y_true is -1 or 1
            derivative[i] = (y_true[i] * y_pred[i] < 1) ? -y_true[i] / size : 0;
        }
        return derivative;
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

typedef Vector (*ActivationFunction)(const Vector&);
typedef double (*LossFunction)(const Vector&, const Vector&);
typedef Vector (*LossDerivativeFunction)(const Vector&, const Vector&);

class Layer {
    public:
        Matrix weights;
        Vector bias;
        ActivationFunction activation;
        ActivationFunction activation_derivative;

        Layer(int in_features, int out_features, 
            ActivationFunction act_func = Activation::sigmoid,
            ActivationFunction act_deriv = Activation::sigmoid_derivative) 
          : activation(act_func), activation_derivative(act_deriv) {
          
          // Initialize weights matrix with proper dimensions
          weights.resize(out_features, Vector(in_features));
          bias.resize(out_features);
          
          // Initialize with random weights using Xavier initialization
          std::random_device rd;
          std::mt19937 gen(rd());
          double limit = sqrt(6.0 / (in_features + out_features));
          std::uniform_real_distribution<> dis(-limit, limit);
          
          for (int i = 0; i < out_features; i++) {
              for (int j = 0; j < in_features; j++) {
                  weights[i][j] = dis(gen);
              }
              bias[i] = 0.0; // Initialize biases to zero
          }
      }
      
    Vector forward(const Vector& input) {
          Vector z = addVectors(matVecMult(weights, input), bias);
          return activation(z);
      }

    Vector A(const Vector& Z) {
        return activation(Z);
    }
  };

class NeuralNetwork {
    public:
    std::vector<Layer> layers;
    std::vector<Vector> Zs;
    std::vector<Vector> activations;
    LossFunction loss_func;
    LossDerivativeFunction loss_derivative;
    
    NeuralNetwork() 
        : loss_func(LossFunctions::mse), 
          loss_derivative(LossFunctions::mse_derivative) {}
    


void addLayer(int in_features, int out_features, 
            ActivationFunction act_func = Activation::sigmoid,
            ActivationFunction act_deriv = Activation::sigmoid_derivative) {
    layers.push_back(Layer(in_features, out_features, act_func, act_deriv));
  
    // Resize Zs and activations vectors
    Zs.resize(layers.size());
    activations.resize(layers.size() + 1);
}

void setLossFunction(LossFunction loss, LossDerivativeFunction loss_deriv) {
  loss_func = loss;
  loss_derivative = loss_deriv;
}

double loss(const Vector& y_true, const Vector& y_pred) {
  return loss_func(y_true, y_pred);
}

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
            std::vector<Vector>& activations,
            std::vector<Vector>& Zs,          
            const Vector& y_true,
            std::vector<Matrix>& dW,         
            std::vector<Vector>& dB,
            double learning_rate          
        ) { 
            int L = weights.size();
            std::vector<Vector> deltas(L);
            
            // Compute output layer error (delta^L)
            Vector y_pred = activations.back();
            Vector delta = loss_derivative(y_true, y_pred);
            
            // Hadamard product with derivative of activation function for output layer
            deltas[L-1] = hadamardProduct(delta, layers[L-1].activation_derivative(Zs[L-1]));
            
            // Backpropagation through hidden layers
            for (int l = L - 2; l >= 0; l--) {
                Matrix W_next_T = transpose(weights[l+1]);
                Vector dz = matVecMult(W_next_T, deltas[l+1]);
                deltas[l] = hadamardProduct(dz, layers[l].activation_derivative(Zs[l]));
            }
            
            // Compute gradients
            for (int l = 0; l < L; l++) {
                dW[l] = matMult(deltas[l], activations[l]); // dL/dW^(l) = δ^(l) * A^(l-1)^T
                dB[l] = deltas[l]; // dL/dB^(l) = δ^(l)
            }
            
            // Update weights and biases
            Optimizer::gradientDescent(weights, biases, dW, dB, learning_rate);
        }

        void train(const Vector& X, const Vector& y_true, double learning_rate = 0.01) {
            // Get weights and biases from layers
            std::vector<Matrix> weights(layers.size());
            std::vector<Vector> biases(layers.size());
            
            for (size_t i = 0; i < layers.size(); i++) {
                weights[i] = layers[i].weights;
                biases[i] = layers[i].bias;
            }
            
            // Forward pass
            Vector y_pred = forwardPass(weights, biases, X, activations, Zs);
            
            // Prepare gradient containers
            std::vector<Matrix> dW(layers.size());
            std::vector<Vector> dB(layers.size());
            
            // Backpropagation
            backpropagation(weights, biases, activations, Zs, y_true, dW, dB, learning_rate);
            
            // Update the layer weights and biases
            for (size_t i = 0; i < layers.size(); i++) {
                layers[i].weights = weights[i];
                layers[i].bias = biases[i];
            }
        }
        
        Vector predict(const Vector& X) {
            // Get weights and biases from layers
            std::vector<Matrix> weights(layers.size());
            std::vector<Vector> biases(layers.size());
            
            for (size_t i = 0; i < layers.size(); i++) {
                weights[i] = layers[i].weights;
                biases[i] = layers[i].bias;
            }
            
            // Forward pass
            return forwardPass(weights, biases, X, activations, Zs);
        }
        
        void fit(const std::vector<Vector>& X_train, const std::vector<Vector>& y_train, 
            int epochs, double learning_rate = 0.01) {
       if (X_train.size() != y_train.size()) {
           throw std::invalid_argument("X_train and y_train must have the same number of samples");
       }
       
       for (int epoch = 0; epoch < epochs; epoch++) {
           double total_loss = 0.0;
           
           for (size_t i = 0; i < X_train.size(); i++) {
               // Train on each sample
               train(X_train[i], y_train[i], learning_rate);
               
               // Compute loss
               Vector y_pred = predict(X_train[i]);
               total_loss += loss(y_train[i], y_pred);
           }
           
           // Calculate average loss
           double avg_loss = total_loss / X_train.size();
           std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << avg_loss << std::endl;
       }
   }
};


#endif // MPL_HPP