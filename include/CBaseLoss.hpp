/*!
 * \file CBaseLoss.hpp
 * \brief Abstract base class for all PINN loss functions.
 *
 * Network convention (5 outputs):
 *   outputs[0] = u     (x-velocity)
 *   outputs[1] = v     (y-velocity)
 *   outputs[2] = p     (pressure)
 *   outputs[3] = k     (turbulent kinetic energy)
 *   outputs[4] = omega (specific dissipation rate)
 *
 * Network inputs (2 inputs):
 *   inputs[0] = x  (spatial x-coordinate)
 *   inputs[1] = y  (spatial y-coordinate)
 *
 * PredictionResult carries outputs, Jacobians, and Hessians
 * all computed by CNeuralNetwork::Predict(x, true, true).
 *   jacobians [iOutput][iInput]            = d(output_iOutput)/d(input_iInput)
 *   hessians  [iOutput][iInput1][iInput2]  = d²(output_iOutput)/(d_iInput1 d_iInput2)
 *
 * Jacobian index map:
 *   jacobians[0][0] = du/dx,  jacobians[0][1] = du/dy
 *   jacobians[1][0] = dv/dx,  jacobians[1][1] = dv/dy
 *   jacobians[2][0] = dp/dx,  jacobians[2][1] = dp/dy
 *   jacobians[3][0] = dk/dx,  jacobians[3][1] = dk/dy
 *   jacobians[4][0] = dω/dx,  jacobians[4][1] = dω/dy
 *
 * Hessian index map (symmetric):
 *   hessians[iOut][0][0] = d²f/dx²
 *   hessians[iOut][0][1] = d²f/dxdy
 *   hessians[iOut][1][1] = d²f/dy²
 */

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include "variable_def.hpp"

namespace MLPToolbox {   

// Output index aliases
static constexpr std::size_t IDX_U     = 0;
static constexpr std::size_t IDX_V     = 1;
static constexpr std::size_t IDX_P     = 2;
static constexpr std::size_t IDX_K     = 3;
static constexpr std::size_t IDX_OMEGA = 4;
static constexpr std::size_t N_OUTPUTS = 5;

// Input index aliases
static constexpr std::size_t IDX_X = 0;
static constexpr std::size_t IDX_Y = 1;
static constexpr std::size_t N_INPUTS = 2;

//  PredictionResult
//  Returned by the training loop for each collocation point.
//  Filled from CNeuralNetwork::Predict(inputs, true, true).
struct PredictionResult {
    std::vector<mlpdouble> outputs;                               // [N_OUTPUTS]
    std::vector<std::vector<mlpdouble>> jacobians;                // [N_OUTPUTS][N_INPUTS]
    std::vector<std::vector<std::vector<mlpdouble>>> hessians;    // [N_OUTPUTS][N_INPUTS][N_INPUTS]
};  

// When mlpdouble = double:             returns x directly
// When mlpdouble = codi::RealReverse:  returns x.getValue()
template<typename T>
inline double to_double(const T& x) { return static_cast<double>(x); }

#ifdef MLP_CUSTOM_TYPE
// Full specialisation for codi::RealReverse
template<>
inline double to_double<codi::RealReverse>(const codi::RealReverse& x) {
    return x.getValue();
}
#endif

//  CBaseLoss
class CBaseLoss {
public:
    explicit CBaseLoss(const std::string& name)
        : name_(name)
        , last_loss_value_(0.0)   // BUG 3 FIX: was last_loss_value (missing _)
    {}

    virtual ~CBaseLoss() = default;

    CBaseLoss(const CBaseLoss&)            = delete;
    CBaseLoss& operator=(const CBaseLoss&) = delete;

    /*!
     * \brief Evaluate the loss at a batch of collocation points.
     *
     * \param preds     - Network predictions at each point (outputs + Jacobians + Hessians)
     * \param ref_data  - Reference labels at each point (used by data loss, empty for physics)
     * \returns         - Scalar loss value (on the CoDi tape when mlpdouble=RealReverse)
     */
    virtual mlpdouble Evaluate(
        const std::vector<PredictionResult>&       preds,
        const std::vector<std::vector<mlpdouble>>& ref_data
    ) = 0;

    std::string GetName()          const { return name_; }
    double      GetLastLossValue() const { return last_loss_value_; }

protected:
    std::string name_;
    double      last_loss_value_;
};

}  // namespace MLPToolbox