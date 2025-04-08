#include "utils.h"
#include <iostream>
#include <cmath>

void printMatrix(const std::vector<std::vector<float>>& matrix, const std::string& name) {
    std::cout << name << ":\n";
    for (const auto& row : matrix) {
        std::cout << "  ";
        for (auto val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

bool compareMatrices(const std::vector<std::vector<float>>& mat1,
                     const std::vector<std::vector<float>>& mat2,
                     const std::string& label1,
                     const std::string& label2) {
    bool same = true;
    const float epsilon = 1e-3;
    if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size()) {
        std::cout << "Size mismatch between " << label1 << " and " << label2 << "\n";
        return false;
    }

    for (size_t i = 0; i < mat1.size(); ++i) {
        for (size_t j = 0; j < mat1[0].size(); ++j) {
            if (std::fabs(mat1[i][j] - mat2[i][j]) > epsilon) {
                same = false;
                std::cout << "Mismatch at (" << i << "," << j << "): "
                          << label1 << "=" << mat1[i][j]
                          << ", " << label2 << "=" << mat2[i][j] << "\n";
            }
        }
    }

    if (same) {
        std::cout << label1 << " and " << label2 << " match.\n";
    } else {
        std::cout << label1 << " and " << label2 << " do not match!\n";
    }

    return same;
}

void reportOperations(const std::string& acceleratorName, int totalOps) {
    std::cout << acceleratorName << " total operations: " << totalOps << "\n";
}
