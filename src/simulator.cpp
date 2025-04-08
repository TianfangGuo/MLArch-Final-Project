#include "simulator.h"
#include "utils.h"
#include <iostream>
#include <chrono>

void Simulator::run() {
    std::vector<std::vector<float>> A = {
        {1, 0, 0, 2},
        {0, 3, 0, 0},
        {4, 0, 5, 0},
        {0, 0, 0, 6}
    };

    std::vector<std::vector<float>> B = {
        {7, 0, 0, 0},
        {0, 8, 0, 0},
        {9, 0, 10, 0},
        {0, 0, 0, 11}
    };

    runSingleTest(A, B);
}

void Simulator::runSingleTest(const std::vector<std::vector<float>>& A,
                              const std::vector<std::vector<float>>& B) {
    CPU cpu;

    auto csrA = cpu.denseToCSR(A);
    auto csrB = cpu.denseToCSR(B);

    // For dense accelerators, we pass dense B
    auto denseB = B;

    // Golden Reference (Dense Inner Product)
    InnerProductAccelerator golden;
    auto start = std::chrono::high_resolution_clock::now();
    auto goldenResult = golden.compute(csrA, denseB);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Golden accelerator time: "
              << std::chrono::duration<double>(end - start).count() << " s\n";
    reportOperations("Golden Inner Product", golden.totalOps);

    // Gustavson Accelerator
    GustavsonAccelerator gustavson;
    start = std::chrono::high_resolution_clock::now();
    auto gustavsonResult = gustavson.compute(csrA, denseB);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Gustavson time: "
              << std::chrono::duration<double>(end - start).count() << " s\n";
    reportOperations("Gustavson", gustavson.totalOps);
    compareMatrices(goldenResult, gustavsonResult, "Golden", "Gustavson");

    // Inner Product Accelerator
    InnerProductAccelerator inner;
    start = std::chrono::high_resolution_clock::now();
    auto innerResult = inner.compute(csrA, denseB);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Inner Product time: "
              << std::chrono::duration<double>(end - start).count() << " s\n";
    reportOperations("Inner Product", inner.totalOps);
    compareMatrices(goldenResult, innerResult, "Golden", "Inner Product");

    // Outer Product Accelerator
    OuterProductAccelerator outer;
    start = std::chrono::high_resolution_clock::now();
    auto outerResult = outer.compute(csrA, denseB);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Outer Product time: "
              << std::chrono::duration<double>(end - start).count() << " s\n";
    reportOperations("Outer Product", outer.totalOps);
    compareMatrices(goldenResult, outerResult, "Golden", "Outer Product");
}
