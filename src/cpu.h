#ifndef CPU_H
#define CPU_H

#include <vector>

struct CSRMatrix {
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    int rows, cols;
    int conversionOverhead = 0;
};

struct CSCMatrix {
    std::vector<float> values;
    std::vector<int> row_indices;
    std::vector<int> col_ptr;
    int rows, cols;
    int conversionOverhead = 0;
};

class CPU {
public:
    CSRMatrix denseToCSR(const std::vector<std::vector<float>>& matrix);
    CSCMatrix denseToCSC(const std::vector<std::vector<float>>& matrix);
};

#endif // CPU_H
