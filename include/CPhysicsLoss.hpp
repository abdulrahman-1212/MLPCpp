/*!
 * \file CPhysicsLoss.hpp
 * \brief Generic physics-residual (PDE constraint) loss for PINN training.
 * 
 *  Optional RHS reference values 
 *
 * Three ways to specify the right-hand side of PDE residuals:
 *
 * 1. Homogeneous (RHS = 0 everywhere):
 *      physics_loss.Evaluate(preds, {});
 *      Inside fn: return { du_dx + dv_dy };  // residual = LHS - 0
 *
 * 2. Uniform RHS (same for all points):
 *      physics_loss.SetUniformRHS({0.0, source_term});
 *      Inside fn: return { r0 - s.Rhs(0), r1 - s.Rhs(1) };
 *
 * 3. Per-point RHS (e.g. body force varies with position):
 *      physics_loss.Evaluate(preds, rhs_data);
 *      // rhs_data[i] = {f_x(x_i), f_y(x_i)}
 *      Inside fn: return { r0 - s.Rhs(0), r1 - s.Rhs(1) };
 *
 *  Jacobian / Hessian conventions
 *
 *   output_Jacobian[iOutput][iInput]
 *     = d(output[iOutput]) / d(input[iInput])
 *
 *   PhysicsState::Jac(iOutput, iInput)   → same
 *   PhysicsState::Hess(iOutput, iIn1, iIn2)
 *     = d²(output[iOutput]) / (d(input[iIn1]) d(input[iIn2]))
 *
 *  Example: 2D incompressible continuity + x-momentum
 *
 *   ResidualFunction fn = [=](const PhysicsState& s) {
 *       // Jac(out, in):  Jac(0,0)=du/dx, Jac(0,1)=du/dy
 *       //                Jac(1,0)=dv/dx, Jac(2,0)=dp/dx
 *       mlpdouble continuity = s.Jac(0,0) + s.Jac(1,1);
 *       mlpdouble mom_x      = s.Out(0)*s.Jac(0,0)
 *                            + s.Out(1)*s.Jac(0,1)
 *                            + s.Jac(2,0);   // + (1/rho)*dp/dx
 *       return { continuity - s.Rhs(0),
 *                mom_x      - s.Rhs(1) };
 *   };
 *
 *   CPhysicsLoss loss("NS", fn, 2,
 *       {"x","y"}, {"u","v","p"},
 *       net.GetInputVars(), net.GetOutputVars());
 */

#pragma once

#include <vector>
#include <string>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include "CBaseLoss.hpp"

namespace MLPToolbox {

// ====================================
//  PhysicsState  — zero-overhead semantic accessor
// ==========================================
class PhysicsState {
public:
    PhysicsState(const PredictionResult&         pred,
                 const std::vector<std::size_t>& in_idx,
                 const std::vector<std::size_t>& out_idx,
                 const std::vector<mlpdouble>&   rhs)
        : pred_    (pred)
        , in_idx_  (in_idx)
        , out_idx_ (out_idx)
        , rhs_     (rhs)
    {}

    // Input value
    // in_idx: index into the equation's input name list (eq_in_names).
    // e.g. if eq_in_names = {"x","y"}, In(0)=x, In(1)=y.
    mlpdouble In(std::size_t eq_in_idx) const {
        return pred_.inputs[in_idx_[eq_in_idx]];
    }

    // Network output
    // out_idx: index into the equation's output name list (eq_out_names).
    // e.g. if eq_out_names = {"u","v","p"}, Out(0)=u, Out(1)=v, Out(2)=p.
    mlpdouble Out(std::size_t eq_out_idx) const {
        return pred_.outputs[out_idx_[eq_out_idx]];
    }

    // Jacobian
    // canonical layout output_Jacobian[iOutput][iInput].
    //   Jac(eq_out_idx, eq_in_idx)
    //   = d(eq_out_names[eq_out_idx]) / d(eq_in_names[eq_in_idx])
    //
    // Example: eq_out_names={"u","v","p"}, eq_in_names={"x","y"}
    //   Jac(0, 0) = du/dx    Jac(0, 1) = du/dy
    //   Jac(1, 0) = dv/dx    Jac(1, 1) = dv/dy
    //   Jac(2, 0) = dp/dx    Jac(2, 1) = dp/dy
    mlpdouble Jac(std::size_t eq_out_idx, std::size_t eq_in_idx) const {
        return pred_.output_Jacobian
            [out_idx_[eq_out_idx]]   // network output index
            [in_idx_ [eq_in_idx ]];  // network input  index
    }

    // Hessian
    // Hess(eq_out_idx, eq_in_idx1, eq_in_idx2)
    // = d^2(out) / (d(in1) d(in2))
    mlpdouble Hess(std::size_t eq_out_idx,
                   std::size_t eq_in_idx1,
                   std::size_t eq_in_idx2) const {
        return pred_.output_Hessian
            [out_idx_[eq_out_idx]]
            [in_idx_ [eq_in_idx1]]
            [in_idx_ [eq_in_idx2]];
    }

    // RHS reference value
    // Rhs(eq_idx) returns the right-hand side value for equation eq_idx
    // at this collocation point. Returns 0 if no reference was provided.
    //
    // Usage in residual function:
    //   return { (du_dx + dv_dy) - s.Rhs(0),
    //            (rho*u*du_dx + dp_dx) - s.Rhs(1) };
    mlpdouble Rhs(std::size_t eq_idx) const {
        if (eq_idx < rhs_.size()) return rhs_[eq_idx];
        return mlpdouble(0.0);  // default: homogeneous PDE
    }

    // Number of equations (for validation inside fn)
    std::size_t NumEquations() const { return rhs_.size(); }

private:
    const PredictionResult&         pred_;
    const std::vector<std::size_t>& in_idx_;
    const std::vector<std::size_t>& out_idx_;
    const std::vector<mlpdouble>&   rhs_;
};


// ResidualFunction signature
// Returns the RAW (unsquared) residual vector.
// CPhysicsLoss squares and averages.
// Size must equal n_equations passed to CPhysicsLoss constructor.
using ResidualFunction =
    std::function<std::vector<mlpdouble>(const PhysicsState&)>;


// ========================
//  CPhysicsLoss
// ==================
class CPhysicsLoss : public CBaseLoss {
public:

    /*!
     * \param name           Human-readable identifier (e.g. "continuity").
     * \param fn             Residual function. Returns n_equations raw
     *                       (unsquared) residuals per collocation point.
     *                       Access RHS via s.Rhs(eq_idx).
     * \param n_equations    Expected residual vector size. Validated on
     *                       every Evaluate() call.
     * \param eq_in_names    Input variable names the equation reads,
     *                       in the order used inside fn (e.g. {"x","y"}).
     * \param eq_out_names   Output variable names the equation reads,
     *                       in the order used inside fn (e.g. {"u","v","p"}).
     * \param net_in_names   Full network input name list (net.GetInputVars()).
     * \param net_out_names  Full network output name list (net.GetOutputVars()).
     * \param needs_jacobian Whether fn calls Jac(). Default true.
     * \param needs_hessian  Whether fn calls Hess(). Default false.
     */
    CPhysicsLoss(const std::string&             name,
                 ResidualFunction               fn,
                 std::size_t                    n_equations,
                 const std::vector<std::string>& eq_in_names,
                 const std::vector<std::string>& eq_out_names,
                 const std::vector<std::string>& net_in_names,
                 const std::vector<std::string>& net_out_names,
                 bool                            needs_jacobian = true,
                 bool                            needs_hessian  = false)
        : CBaseLoss(name)
        , residual_fn_    (std::move(fn))
        , n_equations_    (n_equations)
        , needs_jacobian_ (needs_jacobian)
        , needs_hessian_  (needs_hessian)
    {
        if (n_equations_ == 0)
            throw std::invalid_argument(
                "CPhysicsLoss [" + name + "]: n_equations must be >= 1");

        // BUG 3+7 FIX: vector<size_t> — no manual memory, no destructor,
        // no exception-safety issues, rule of zero applies.
        input_indices_.resize(eq_in_names.size());
        for (std::size_t i = 0; i < eq_in_names.size(); ++i)
            input_indices_[i] = resolve_name(
                eq_in_names[i], net_in_names, "input", name);

        output_indices_.resize(eq_out_names.size());
        for (std::size_t i = 0; i < eq_out_names.size(); ++i)
            output_indices_[i] = resolve_name(
                eq_out_names[i], net_out_names, "output", name);
    }


    // Accessors
    bool        RequiresJacobian() const noexcept { return needs_jacobian_; }
    bool        RequiresHessian()  const noexcept { return needs_hessian_;  }
    std::size_t NumEquations()     const noexcept { return n_equations_;    }

    // Uniform RHS setter (optional) 
    // Sets the same RHS for every collocation point.
    // Overridden per-point by non-empty ref_data passed to Evaluate().
    void SetUniformRHS(std::vector<mlpdouble> rhs) {
        if (!rhs.empty() && rhs.size() != n_equations_)
            throw std::invalid_argument(
                "CPhysicsLoss [" + name_ + "]: SetUniformRHS size "
                + std::to_string(rhs.size()) + " != n_equations "
                + std::to_string(n_equations_));
        uniform_rhs_ = std::move(rhs);
    }

    // Evaluate 
    /*!
     * \brief L = (1 / (N x n_eq)) sum_i(sum_j((f_j(x_i) - rhs_j(x_i))^2)) 
     *
     * ref_data usage 
     *   {} or empty     -> use SetUniformRHS() or zero for all points
     *   size N          -> ref_data[i] is the RHS at point i
     *     ref_data[i] = {}           -> zero RHS at point i
     *     ref_data[i].size()=n_eq    -> explicit per-equation RHS at point i
     *     ref_data[i].size()=1       -> same value for all equations at point i
     *   Any other size  -> runtime_error
     */
    mlpdouble Evaluate(
        const std::vector<PredictionResult>&       preds,
        const std::vector<std::vector<mlpdouble>>& ref_data
    ) override
    {
        const std::size_t N = preds.size();

        if (N == 0) {
            last_loss_value_ = 0.0;
            return mlpdouble(0.0);
        }

        const bool has_per_point_rhs = !ref_data.empty();
        if (has_per_point_rhs && ref_data.size() != N)
            throw std::runtime_error(
                "CPhysicsLoss [" + name_ + "]: ref_data has "
                + std::to_string(ref_data.size())
                + " rows but preds has " + std::to_string(N) + " rows.");

        // Static zero RHS fallback (avoids per-call allocation)
        static const std::vector<mlpdouble> zero_rhs;

        mlpdouble total_loss = mlpdouble(0.0);

        for (std::size_t i = 0; i < N; ++i) {

            // Validate pointers when needed
            if (needs_jacobian_ && preds[i].output_Jacobian == nullptr)
                throw std::runtime_error(
                    "CPhysicsLoss [" + name_
                    + "]: Jacobian requested but output_Jacobian is nullptr at point "
                    + std::to_string(i) + ". "
                    + "Ensure CNeuralNetwork::Predict() was called with eval_jac=true.");

            if (needs_hessian_ && preds[i].output_Hessian == nullptr)
                throw std::runtime_error(
                    "CPhysicsLoss [" + name_
                    + "]: Hessian requested but output_Hessian is nullptr at point "
                    + std::to_string(i) + ". "
                    + "Ensure CNeuralNetwork::Predict() was called with eval_hess=true.");

            // Resolve RHS for this point
            const std::vector<mlpdouble>& rhs_i = ResolveRHS(
                has_per_point_rhs ? &ref_data[i] : nullptr);

            // Build PhysicsState — passes RHS by const ref
            PhysicsState state(preds[i], input_indices_, output_indices_, rhs_i);

            // Evaluate residuals (user function)
            const std::vector<mlpdouble> residuals = residual_fn_(state);

            if (residuals.size() != n_equations_)
                throw std::runtime_error(
                    "CPhysicsLoss [" + name_ + "]: residual_fn returned "
                    + std::to_string(residuals.size())
                    + " values at point " + std::to_string(i)
                    + ", expected " + std::to_string(n_equations_) + ".");

            // User returns raw (unsquared) values. We square.
            for (std::size_t j = 0; j < n_equations_; ++j)
                total_loss = total_loss + residuals[j] * residuals[j];
        }

        mlpdouble mse = total_loss
                      / mlpdouble(static_cast<double>(N * n_equations_));
        last_loss_value_ = to_double(mse);
        return mse;
    }

private:

    ResidualFunction      residual_fn_;
    std::size_t           n_equations_;
    bool                  needs_jacobian_;
    bool                  needs_hessian_;
    std::vector<std::size_t> input_indices_;   // eq_in_names → net input indices
    std::vector<std::size_t> output_indices_;  // eq_out_names → net output indices
    std::vector<mlpdouble>   uniform_rhs_;     // optional uniform RHS (all points)

    // RHS resolution 
    const std::vector<mlpdouble>& ResolveRHS(
        const std::vector<mlpdouble>* rhs_i) const
    {
        // Per-point ref provided and non-empty
        if (rhs_i && !rhs_i->empty()) {
            if (rhs_i->size() != n_equations_ && rhs_i->size() != 1)
                throw std::runtime_error(
                    "CPhysicsLoss [" + name_
                    + "]: per-point ref_data row has "
                    + std::to_string(rhs_i->size())
                    + " values; expected 0, 1, or "
                    + std::to_string(n_equations_) + ".");
            return *rhs_i;
        }
        // Uniform RHS set via SetUniformRHS()
        if (!uniform_rhs_.empty()) return uniform_rhs_;
        // Homogeneous — return static empty vector; Rhs(j) returns 0.
        static const std::vector<mlpdouble> zero;
        return zero;
    }

    // Name resolver — called only at construction time (O(n) is fine)
    static std::size_t resolve_name(const std::string&             name,
                                    const std::vector<std::string>& names,
                                    const char*                     var_type,
                                    const std::string&              loss_name)
    {
        auto it = std::find(names.begin(), names.end(), name);
        if (it == names.end())
            throw std::runtime_error(
                "CPhysicsLoss [" + loss_name + "]: " + var_type
                + " '" + name + "' not found among network variables. "
                + "Available: " + [&]{
                    std::string s;
                    for (auto& n : names) s += "'" + n + "' ";
                    return s;
                }());
        return static_cast<std::size_t>(std::distance(names.begin(), it));
    }
};

}  // namespace MLPToolbox