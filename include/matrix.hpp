#include <iostream>
#include <vector>
#include <cmath>



typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

Vector matVecMult(const Matrix& mat, const Vector& vec) {
    int rows = mat.size(), cols = mat[0].size();
    Vector result(rows, 0.0);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

Matrix transpose(const Matrix& mat) {
    int rows = mat.size(), cols = mat[0].size();
    Matrix trans(cols, Vector(rows, 0.0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            trans[j][i] = mat[i][j];
        }
    }
    return trans;
}

Vector hadamardProduct(const Vector& a, const Vector& b) {
    int size = a.size();
    Vector result(size);
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
    return result;
}

Matrix matMult(const Vector& a, const Vector& b) {
    int rows = a.size(), cols = b.size();
    Matrix result(rows, Vector(cols, 0.0));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

