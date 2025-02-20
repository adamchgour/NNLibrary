#ifndef MPL_HPP
#define MPL_HPP
#include <vector>

class Module {
public:
    virtual std::vector<float> forward(const std::vector<float>& x) = 0;
};

class Layer : public Module {
    public:
        Layer(int in_features, int out_features);
        std::vector<float> forward(const std::vector<float>& x) override;
        std::vector<float> backward(const std::vector<float>& grad_output);
        void update(float lr);
        void zero_grad();
    
    private:
        int in_features;
        int out_features;
        std::vector<std::vector<float>> weights;
        std::vector<float> biases;
        std::vector<std::vector<float>> weights_jacobian;
        std::vector<float> bias_gradient;
};

class MLP : public Module {
    public:
        MLP(int input_size, int output_size);
        std::vector<float> forward(const std::vector<float>& x) override;
        void backward(const std::vector<float>& grad_output);
        void update(float lr);
        void zero_grad();
    
    private:
        std::vector<Layer> layers;
};


#endif // MPL_HPP