/*!
 * \file CPhysicsLoss.hpp
 * \brief Generic physics-residual (PDE constraint) loss for PINN training.
 *
*/

#pragma once

#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include "CBaseLoss.hpp"

namespace MLPToolbox {

// =====================================================================
// PhysicsState - semantic accessor over one PredictionResult, using
// indices pre-resolved once at CPhysicsLoss construction time 
// =====================================================================
class PhysicsState {
public:
    PhysicsState(const PredictionResult& pred,
                 const size_t*            in_idx,
                 const size_t*            out_idx)
        : pred_(pred), in_indices_(in_idx), out_indices_(out_idx)
    {}

    mlpdouble In (size_t idx) const { return pred_.inputs [in_indices_ [idx]]; }
    mlpdouble Out(size_t idx) const { return pred_.outputs[out_indices_[idx]]; }

    mlpdouble Jac(size_t out_idx, size_t in_idx) const {
        return pred_.jacobians[out_indices_[out_idx]][in_indices_[in_idx]];
    }

    mlpdouble Hess(size_t out_idx, size_t in1_idx, size_t in2_idx) const {
        return pred_.hessians[out_indices_[out_idx]]
                             [in_indices_[in1_idx]]
                             [in_indices_[in2_idx]];
    }

private:
    const PredictionResult& pred_;
    const size_t*           in_indices_;
    const size_t*           out_indices_;
};

/*!
 * \brief One residual value per governing equation, evaluated at a
 *        single collocation point.
 *
 * Return the RAW residual for each equation - do NOT pre-square or
 * pre-combine them. CPhysicsLoss squares and sums every element
 *
 * Example (2-equation system: continuity + one momentum component):
 *   ResidualFunction fn = [=](const PhysicsState& s) -> std::vector<mlpdouble> {
 *       mlpdouble r_continuity = s.Jac(0,0) + s.Jac(1,1);          // du/dx+dv/dy
 *       mlpdouble r_momentum_x = s.Out(0)*s.Jac(0,0) - s.Jac(2,0); // u du/dx - dp/dx
 *       return { r_continuity, r_momentum_x };
 *   };
 */
using ResidualFunction =
    std::function<std::vector<mlpdouble>(const PhysicsState&)>;

// =====================================================================
// CPhysicsLoss
// =====================================================================
class CPhysicsLoss : public CBaseLoss {
public:
    /*!
     * \param name          Human-readable identifier (e.g. "continuity").
     * \param fn             Residual function; must return exactly
     *                       n_equations raw (unsquared) values per point.
     * \param n_equations    Expected size of fn's returned vector.
     *                       Validated on every Evaluate() call.
     * \param eq_in_names    Input variable names this equation reads
     *                       (e.g. {"x","y"}), in the order used inside fn.
     * \param eq_out_names   Output variable names this equation reads
     *                       (e.g. {"u","v","p"}).
     * \param net_in_names   Full network input name list (net.GetInputVars()).
     * \param net_out_names  Full network output name list (net.GetOutputVars()).
     * \param needs_jacobian Whether fn calls Jac(). Default true (physics
     *                       losses almost always need at least first
     *                       derivatives). Set false for purely algebraic
     *                       constraints (e.g. p >= 0) to skip Jacobian
     *                       computation for this loss entirely.
     * \param needs_hessian  Whether fn calls Hess(). Default false -
     *                       most physics losses only need first order.
     */
    CPhysicsLoss(const std::string&               name,
                 ResidualFunction                  fn,
                 std::size_t                       n_equations,
                 const std::vector<std::string>&   eq_in_names,
                 const std::vector<std::string>&   eq_out_names,
                 const std::vector<std::string>&   net_in_names,
                 const std::vector<std::string>&   net_out_names,
                 bool                               needs_jacobian = true,
                 bool                               needs_hessian  = false)
        : CBaseLoss(name)
        , residual_fn_(std::move(fn))
        , n_equations_(n_equations)
        , needs_jacobian_(needs_jacobian)
        , needs_hessian_(needs_hessian)
    {
        if (n_equations_ == 0)
            throw std::invalid_argument(
                "CPhysicsLoss [" + name + "]: n_equations must be >= 1");

        in_indices_.resize(eq_in_names.size());
        for (std::size_t i = 0; i < eq_in_names.size(); ++i)
            in_indices_[i] = resolve_name(eq_in_names[i], net_in_names, "input");

        out_indices_.resize(eq_out_names.size());
        for (std::size_t i = 0; i < eq_out_names.size(); ++i)
            out_indices_[i] = resolve_name(eq_out_names[i], net_out_names, "output");
    }

    /*!
     * \brief Whether this loss requires Jacobian evaluation.
     *        CMLPTrainer OR-combines this across all losses sharing a
     *        collocation set to decide whether to request derivatives
     *        from CNeuralNetwork::Predict() at all for that batch.
     */
    bool RequiresJacobian() const { return needs_jacobian_; }
    bool RequiresHessian()  const { return needs_hessian_;  }
    std::size_t NumEquations() const { return n_equations_; }

    /*!
     * \brief L = (1 / (N * n_equations)) * sum_i sum_j r_j(x_i)^2
     */
    mlpdouble Evaluate(
        const std::vector<PredictionResult>&       preds,
        const std::vector<std::vector<mlpdouble>>& /* ref_data — unused */
    ) override {

        const std::size_t N = preds.size();
        if (N == 0) { last_loss_value_ = 0.0; return mlpdouble(0.0); }

        mlpdouble total_loss = mlpdouble(0.0);

        for (const auto& pred : preds) {
            PhysicsState state(pred, in_indices_.data(), out_indices_.data());

            std::vector<mlpdouble> r = residual_fn_(state);
            if (r.size() != n_equations_)
                throw std::runtime_error(
                    "CPhysicsLoss [" + name_ + "]: residual_fn returned " +
                    std::to_string(r.size()) + " values, expected " +
                    std::to_string(n_equations_));

            for (const auto& rj : r)
                total_loss += rj * rj;
        }

        // Denominator built from doubles first (size_t overflow guard).
        const double denom_d = static_cast<double>(N) *
                               static_cast<double>(n_equations_);
        mlpdouble mse = total_loss / mlpdouble(denom_d);

        last_loss_value_ = to_double(mse);
        return mse;
    }

private:
    static size_t resolve_name(const std::string&               name,
                               const std::vector<std::string>& names,
                               const char*                      var_type) {
        auto it = std::find(names.begin(), names.end(), name);
        if (it == names.end())
            throw std::runtime_error(
                std::string("CPhysicsLoss: ") + var_type + " '" + name +
                "' not found among network variables.");
        return static_cast<size_t>(std::distance(names.begin(), it));
    }

    ResidualFunction     residual_fn_;
    std::size_t          n_equations_;
    bool                 needs_jacobian_;
    bool                 needs_hessian_;
    std::vector<size_t>  in_indices_;
    std::vector<size_t>  out_indices_;
};

}  // namespace MLPToolbox