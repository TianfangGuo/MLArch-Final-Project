#include "accelerator.h"

std::vector<std::vector<float>> GustavsonAccelerator::compute(
    const CSRMatrix& A, const std::vector<std::vector<float>>& B) {
    int m = A.rows;
    int n = B[0].size();
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0));

    totalOps = A.conversionOverhead;

    for (int i = 0; i < m; i++) {
        std::vector<float> row_acc(n, 0);
        for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; idx++) {
            int col = A.col_indices[idx];
            float val = A.values[idx];
            totalOps++;
            for (int j = 0; j < n; j++) {
                row_acc[j] += val * B[col][j];
                totalOps++;
            }
        }
        C[i] = row_acc;
    }
    return C;
}

std::vector<std::vector<float>> InnerProductAccelerator::compute(
    const CSRMatrix& A, const std::vector<std::vector<float>>& B) {

    int m = A.rows;
    int n = B[0].size();
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0));

    totalOps = A.conversionOverhead;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;

            // Read only non-zero elements of row i
            for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; idx++) {
                int k = A.col_indices[idx];
                float val = A.values[idx];
                totalOps++;
                sum += val * B[k][j];
                totalOps++;
            }

            C[i][j] = sum;
        }
    }

    return C;
}


std::vector<std::vector<float>> OuterProductAccelerator::compute(
    const CSRMatrix& A, const std::vector<std::vector<float>>& B) {
    int m = A.rows;
    int n = B[0].size();
    int p = A.cols;
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0));

    totalOps = A.conversionOverhead;

    for (int k = 0; k < p; k++) {
        for (int idx = A.row_ptr[0]; idx < A.row_ptr[A.rows]; idx++) {
            if (A.col_indices[idx] != k) continue;
            int i = 0;
            while (i < A.rows && !(A.row_ptr[i] <= idx && idx < A.row_ptr[i + 1])) i++;
            float val = A.values[idx];
            totalOps++;

            for (int j = 0; j < n; j++) {
                C[i][j] += val * B[k][j];
                totalOps++;
            }
        }
    }

    return C;
}
