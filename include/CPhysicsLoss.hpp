#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include "CBaseLoss.hpp"

namespace MLPToolbox {

// =====================================================================
// PhysicsState: semantic wrapper using pre-resolved indices
// =====================================================================
class PhysicsState {
private:
    const PredictionResult& pred_;
    const size_t* in_indices_;   
    const size_t* out_indices_;  

public:
    PhysicsState(const PredictionResult& pred,
                 const size_t* in_idx, 
                 const size_t* out_idx)
        : pred_(pred), in_indices_(in_idx), out_indices_(out_idx) {}

    mlpdouble In(size_t idx) const { return pred_.inputs[in_indices_[idx]]; }
    mlpdouble Out(size_t idx) const { return pred_.outputs[out_indices_[idx]]; }

    mlpdouble Jac(size_t out_idx, size_t in_idx) const {
        return pred_.jacobians[out_indices_[out_idx]][in_indices_[in_idx]];
    }

    mlpdouble Hess(size_t out_idx, size_t in1_idx, size_t in2_idx) const {
        return pred_.hessians[out_indices_[out_idx]]
                             [in_indices_[in1_idx]]
                             [in_indices_[in2_idx]];
    }
};

using ResidualFunction = std::function<mlpdouble(const PhysicsState&)>;

// =====================================================================
// CPhysicsLoss
// =====================================================================
class CPhysicsLoss : public CBaseLoss {
private:
    ResidualFunction residual_fn_;

    // Pre-resolved index arrays
    std::vector<size_t> in_indices_;
    std::vector<size_t> out_indices_;

    static size_t resolve_name(const std::string& name,
                               const std::vector<std::string>& names,
                               const char* var_type) {
        auto it = std::find(names.begin(), names.end(), name);
        if (it == names.end()) {
            throw std::runtime_error(
                std::string("CPhysicsLoss: ") + var_type + " '" + name + "' not found.");
        }
        return static_cast<size_t>(std::distance(names.begin(), it));
    }

public:
    CPhysicsLoss(const std::string& name, ResidualFunction fn,
                 const std::vector<std::string>& eq_in_names,
                 const std::vector<std::string>& eq_out_names,
                 const std::vector<std::string>& net_in_names,
                 const std::vector<std::string>& net_out_names)
        : CBaseLoss(name), residual_fn_(std::move(fn))
    {
        // Resolve ALL names to indices ONCE. 
        in_indices_.resize(eq_in_names.size());
        for (size_t i = 0; i < eq_in_names.size(); ++i)
            in_indices_[i] = resolve_name(eq_in_names[i], net_in_names, "Input");

        out_indices_.resize(eq_out_names.size());
        for (size_t i = 0; i < eq_out_names.size(); ++i)
            out_indices_[i] = resolve_name(eq_out_names[i], net_out_names, "Output");
    }

    mlpdouble Evaluate(
        const std::vector<PredictionResult>& preds,
        const std::vector<std::vector<mlpdouble>>& /*ref_data*/
    ) override {
        mlpdouble total_loss = 0.0;
        const size_t N = preds.size();

        if (N == 0) { last_loss_value_ = 0.0; return mlpdouble(0.0); }

        for (const auto& pred : preds) {
            PhysicsState state(pred, in_indices_.data(), out_indices_.data());

            mlpdouble res = residual_fn_(state);
            
            total_loss += res * res;
        }

        // Safe cast from size_t to mlpdouble
        mlpdouble mse = total_loss / mlpdouble(static_cast<double>(N));
        last_loss_value_ = to_double(mse);
        return mse;
    }
};

} // namespace MLPToolbox