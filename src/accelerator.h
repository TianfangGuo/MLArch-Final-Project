#ifndef ACCELERATOR_H
#define ACCELERATOR_H

#include "cpu.h"
#include <vector>

class Accelerator {
public:
    virtual std::vector<std::vector<float>> compute(
        const CSRMatrix& A, const std::vector<std::vector<float>>& B) = 0;

    int totalOps = 0;
    int conversionOverhead = 0;
};

class GustavsonAccelerator : public Accelerator {
public:
    std::vector<std::vector<float>> compute(
        const CSRMatrix& A, const std::vector<std::vector<float>>& B) override;
};

class InnerProductAccelerator : public Accelerator {
public:
    std::vector<std::vector<float>> compute(
        const CSRMatrix& A, const std::vector<std::vector<float>>& B) override;
};

class OuterProductAccelerator : public Accelerator {
public:
    std::vector<std::vector<float>> compute(
        const CSRMatrix& A, const std::vector<std::vector<float>>& B) override;
};

#endif // ACCELERATOR_H
