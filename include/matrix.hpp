#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <vector>
#include <memory>
#include <functional>

class Tensor {
    public:
    std::vector<std::vector<double>> data;
    std::vector<std::vector<double>> grad;
    std::vector<std::shared_ptr<Tensor>> parents; // list of pointers who manages memorry themselves to parent matrices
    std::function <void()> grad_function;
    bool requires_grad = false;
  
    bool is_grad = true;
    Tensor(int rows, int cols, bool requires_grad = false) : requires_grad(requires_grad) {
        data.resize(rows, std::vector<double>(cols, 0.0));
        grad.resize(rows, std::vector<double>(cols, 0.0));
    }
    Tensor(std::vector<std::vector<double>> values, bool requires_grad = false) 
        : data(values), requires_grad(requires_grad) {
        grad.resize(values.size(), std::vector<double>(values[0].size(), 0.0));
    }

};  
#endif // MATRIX_HPP
