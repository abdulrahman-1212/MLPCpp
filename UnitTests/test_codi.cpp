/*
 * ============================================================
 *  2-Layer Neural Network AD Verification Test (SU2 + FD)
 * ============================================================
 *
 *  This test validates the correctness of SU2's Algorithmic
 *  Differentiation (AD) wrapper by comparing computed gradients
 *  against finite difference (FD) approximations.
 *
 *  Network structure:
 *      - Input: (x1, x2)
 *      - Hidden layer (2 neurons):
 *            h1 = tanh(w11*x1 + w12*x2)
 *            h2 = tanh(w21*x1 + w22*x2)
 *      - Output:
 *            y = v1*h1 + v2*h2 + b2
 *      - Loss:
 *            L = y^2
 *
 *  Parameters:
 *      p = [w11, w12, w21, w22, v1, v2, b2]
 *
 *  Procedure:
 *      1. Record computation graph using SU2 AD
 *      2. Perform reverse-mode differentiation
 *      3. Compute analytical gradients via AD
 *      4. Compute numerical gradients using centered finite difference
 *      5. Compare both results for validation
 *
 *  Purpose:
 *      - Regression test for SU2 AD correctness
 *      - Validation of gradient flow through nonlinear activations
 *      - Baseline for future MLPCpp training / optimizer integration
 *  
 *  HOW TO RUN
 *  ----------
 *
 *  From MLPCpp directory:
 *
 *      rm -rf build
 *      mkdir build && cd build
 *      cmake ..
 *      cmake --build . -j
 *      ./mlpcpp_ad_test
 * ============================================================
 */


#include <iostream>
#include <vector>
#include <cmath>

#include "../../Common/include/basic_types/datatype_structure.hpp"

su2double loss_function(
    const su2double& x1,
    const su2double& x2,
    const std::vector<su2double>& p)
{
    // unpack parameters
    const su2double& w11 = p[0];
    const su2double& w12 = p[1];
    const su2double& w21 = p[2];
    const su2double& w22 = p[3];
    const su2double& v1  = p[4];
    const su2double& v2  = p[5];
    const su2double& b2  = p[6];

    // hidden layer
    su2double h1 = tanh(w11*x1 + w12*x2);
    su2double h2 = tanh(w21*x1 + w22*x2);

    // output
    su2double y = v1*h1 + v2*h2 + b2;

    // squared loss
    return y*y;
}

static su2double fd_grad(
    const std::vector<su2double>& p,
    std::size_t idx,
    su2double x1, su2double x2,
    su2double h = 1e-5)
{
    auto net = [&](const std::vector<su2double>& q) -> su2double {
        su2double h1 = std::tanh(q[0]*x1 + q[1]*x2);
        su2double h2 = std::tanh(q[2]*x1 + q[3]*x2);
        su2double y = q[4]*h1 + q[5]*h2 + q[6];

        return y * y;
    };

    std::vector<su2double> pp = p, pm = p;
    pp[idx] += h;
    pm[idx] -= h;

    return (net(pp) - net(pm)) / (2.0 * h);
}

int main() {

    const su2double x1 = 1.0;
    const su2double x2 = 2.0;

    std::vector<su2double> p = {
        0.1, -0.2,
        0.4,  0.3,
        0.2, -0.5,
        0.1
    };

    // START RECORDING
    AD::StartRecording();

    for (auto& param : p)
        AD::RegisterInput(param);

    // forward pass
    su2double L = loss_function(x1, x2, p);

    AD::RegisterOutput(L);

    AD::StopRecording();

    // BACKWARD PASS

    SU2_TYPE::SetDerivative(L, 1.0);

    AD::ComputeAdjoint();

    
    // PRINT GRADIENTS
    std::cout << "\n=== SU2 AD Gradients ===\n";

    std::cout << "dL/dw11 = " << SU2_TYPE::GetDerivative(p[0]) << "\n";
    std::cout << "dL/dw12 = " << SU2_TYPE::GetDerivative(p[1]) << "\n";
    std::cout << "dL/dw21 = " << SU2_TYPE::GetDerivative(p[2]) << "\n";
    std::cout << "dL/dw22 = " << SU2_TYPE::GetDerivative(p[3]) << "\n";
    std::cout << "dL/dv1  = " << SU2_TYPE::GetDerivative(p[4]) << "\n";
    std::cout << "dL/dv2  = " << SU2_TYPE::GetDerivative(p[5]) << "\n";
    std::cout << "dL/db2  = " << SU2_TYPE::GetDerivative(p[6]) << "\n";

    AD::ClearAdjoints();
    AD::Reset();


    std::vector<su2double> fd(7);
    for (std::size_t i = 0; i < 7; ++i)
        fd[i] = fd_grad(p, i, x1, x2);

    
    std::cout << "\n=== Finite Difference Gradients ===\n";
    std::cout << "dL/dw11 = " << fd[0] << "\n";
    std::cout << "dL/dw12 = " << fd[1] << "\n";
    std::cout << "dL/dw21 = " << fd[2] << "\n";
    std::cout << "dL/dw22 = " << fd[3] << "\n";
    std::cout << "dL/dv1  = " << fd[4] << "\n";
    std::cout << "dL/dv2  = " << fd[5] << "\n";
    std::cout << "dL/db2  = " << fd[6] << "\n";


    return 0;
}