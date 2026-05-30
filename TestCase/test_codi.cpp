/* CoDiPack AD Test: 2-Layer Neural Network Weight Sensitivities
 *
 * This test validates reverse-mode automatic differentiation (AD)
 * using CoDiPack on a small fully-connected neural network.
 *
 * Network structure:
 *
 *   Inputs:
 *       x1, x2
 *
 *   Hidden layer:
 *       h1 = tanh(w11*x1 + w12*x2)
 *       h2 = tanh(w21*x1 + w22*x2)
 *
 *   Output layer:
 *       y  = v1*h1 + v2*h2 + b2
 *
 *   Loss:
 *       L = y^2
 *
 * The test computes:
 *
 *   1. Reverse-mode AD gradients using CoDiPack
 *   2. Numerical gradients using centered finite differences
 *
 * The two sets of sensitivities are then compared to verify:
 *
 *   dL/dtheta (AD) ≈ dL/dtheta (Finite Difference)
 *
 * for all trainable parameters:
 *
 *   {w11, w12, w21, w22, v1, v2, b2}
 *
 * This serves as:
 *
 *   - a validation of CoDiPack integration,
 *   - a regression test for MLPCpp AD support,
 *   - and a minimal prototype for neural-network sensitivity
 *     analysis inside SU2-style adjoint workflows.
 *      
 *      How to run the test:
 *      from MLPCpp dir, run these commands:
 *          rm -rf build
 *          mkdir build 
 *          cd build
 *          cmake ..
 *          cmake --build . -j
 *          ./mlpcpp_ad_test
 * =================================================================== */
#include <iostream>
#include <vector>
#include <cmath>
#include "codi.hpp"


// Finite difference helper
double finite_difference(
    const std::vector<double>& p,
    int idx,
    double eps = 1e-6)
{
    std::vector<double> p1 = p;
    std::vector<double> p2 = p;

    p1[idx] += eps;
    p2[idx] -= eps;

    auto forward = [](const std::vector<double>& w) -> double {

        double x1 = 1.0;
        double x2 = 2.0;

        double w11 = w[0];
        double w12 = w[1];
        double w21 = w[2];
        double w22 = w[3];
        double v1  = w[4];
        double v2  = w[5];
        double b2  = w[6];

        // 2-layer NN (NONLINEAR = important!)
        double h1 = std::tanh(w11 * x1 + w12 * x2);
        double h2 = std::tanh(w21 * x1 + w22 * x2);

        double y = v1 * h1 + v2 * h2 + b2;

        double L = y * y;
        return L;
    };

    return (forward(p1) - forward(p2)) / (2.0 * eps);
}

int main()
{
    // Inputs
    codi::RealReverse x1 = 1.0;
    codi::RealReverse x2 = 2.0;

    codi::RealReverse w11 = 0.1, w12 = -0.2;
    codi::RealReverse w21 = 0.4, w22 = 0.3;

    codi::RealReverse v1  = 0.2, v2  = -0.5;
    codi::RealReverse b2  = 0.1;

    codi::RealReverse::Tape& tape = codi::RealReverse::getTape();

    // Tape setup
    tape.setActive();

    tape.registerInput(w11);
    tape.registerInput(w12);
    tape.registerInput(w21);
    tape.registerInput(w22);
    tape.registerInput(v1);
    tape.registerInput(v2);
    tape.registerInput(b2);

    // Forward pass (NN)
    codi::RealReverse h1 = codi::tanh(w11 * x1 + w12 * x2);
    codi::RealReverse h2 = codi::tanh(w21 * x1 + w22 * x2);

    codi::RealReverse y  = v1 * h1 + v2 * h2 + b2;

    codi::RealReverse L  = y * y;   // loss

    tape.registerOutput(L);

    // Reverse pass (AD)
    tape.setPassive();
    L.setGradient(1.0);
    tape.evaluate();

    // Print AD gradients
    std::cout << "=== AD Gradients ===\n";
    std::cout << "dL/dw11 = " << w11.getGradient() << "\n";
    std::cout << "dL/dw12 = " << w12.getGradient() << "\n";
    std::cout << "dL/dw21 = " << w21.getGradient() << "\n";
    std::cout << "dL/dw22 = " << w22.getGradient() << "\n";
    std::cout << "dL/dv1  = " << v1.getGradient()  << "\n";
    std::cout << "dL/dv2  = " << v2.getGradient()  << "\n";
    std::cout << "dL/db2  = " << b2.getGradient()  << "\n";

    // Finite difference check
    std::vector<double> p = {
        0.1, -0.2,
        0.4,  0.3,
        0.2, -0.5,
        0.1
    };

    std::cout << "\n=== Finite Difference ===\n";
    std::cout << "dw11 FD = " << finite_difference(p, 0) << "\n";
    std::cout << "dw12 FD = " << finite_difference(p, 1) << "\n";
    std::cout << "dw21 FD = " << finite_difference(p, 2) << "\n";
    std::cout << "dw22 FD = " << finite_difference(p, 3) << "\n";
    std::cout << "dv1  FD = " << finite_difference(p, 4) << "\n";
    std::cout << "dv2  FD = " << finite_difference(p, 5) << "\n";
    std::cout << "db2  FD = " << finite_difference(p, 6) << "\n";

    return 0;
}