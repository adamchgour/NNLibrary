#ifndef LOSS_HPP
#define LOSS_HPP

#include <vector>
#include <cmath>

namespace loss {
    double mse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        double sum = 0;
        for (int i = 0; i < y_true.size(); i++) {
            sum += pow(y_true[i] - y_pred[i], 2);
        }
        return sum / y_true.size();
    }

    double mae(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        double sum = 0;
        for (int i = 0; i < y_true.size(); i++) {
            sum += abs(y_true[i] - y_pred[i]);
        }
        return sum / y_true.size();
    }

    double rmse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        return sqrt(mse(y_true, y_pred));
    }
}

#endif