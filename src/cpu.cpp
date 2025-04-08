#include "cpu.h"

CSRMatrix CPU::denseToCSR(const std::vector<std::vector<float>>& matrix) {
    CSRMatrix csr;
    csr.rows = matrix.size();
    csr.cols = matrix[0].size();
    csr.row_ptr.push_back(0);

    int nnz = 0; // non-zero counter

    for (int i = 0; i < csr.rows; i++) {
        for (int j = 0; j < csr.cols; j++) {
            float val = matrix[i][j];
            if (val != 0) {
                csr.values.push_back(val);
                csr.col_indices.push_back(j);
                nnz++;
            }
        }
        csr.row_ptr.push_back(nnz);
    }

    csr.conversionOverhead = nnz;
    return csr;
}

CSCMatrix CPU::denseToCSC(const std::vector<std::vector<float>>& matrix) {
    CSCMatrix csc;
    csc.rows = matrix.size();
    csc.cols = matrix[0].size();
    csc.col_ptr.push_back(0);

    int nnz = 0;

    for (int j = 0; j < csc.cols; j++) {
        for (int i = 0; i < csc.rows; i++) {
            float val = matrix[i][j];
            if (val != 0) {
                csc.values.push_back(val);
                csc.row_indices.push_back(i);
                nnz++;
            }
        }
        csc.col_ptr.push_back(nnz);
    }

    csc.conversionOverhead = nnz;
    return csc;
}
