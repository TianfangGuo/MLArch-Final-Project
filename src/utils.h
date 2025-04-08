#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

void printMatrix(const std::vector<std::vector<float>>& matrix, const std::string& name);
bool compareMatrices(const std::vector<std::vector<float>>& mat1,
                     const std::vector<std::vector<float>>& mat2,
                     const std::string& label1,
                     const std::string& label2);
void reportOperations(const std::string& acceleratorName, int totalOps);

#endif // UTILS_H
