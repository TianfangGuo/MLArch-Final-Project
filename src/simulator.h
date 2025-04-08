#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "cpu.h"
#include "accelerator.h"
#include <vector>

class Simulator {
public:
    void run();

private:
    void runSingleTest(const std::vector<std::vector<float>>& A,
                       const std::vector<std::vector<float>>& B);
};

#endif // SIMULATOR_H
