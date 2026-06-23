#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "CBaseLoss.hpp"


namespace MLPToolbox {

struct KOmegaSSTConstants {
    mlpdouble beta_star    = mlpdouble(0.09);           // k-destruction coefficient
    mlpdouble alpha        = mlpdouble(5.0 / 9.0);      // omega-production coefficient
    mlpdouble beta         = mlpdouble(0.075);           // omega-destruction coefficient
    mlpdouble sigma_k      = mlpdouble(0.85);            // k diffusivity coefficient
    mlpdouble sigma_omega  = mlpdouble(0.5);             // omega diffusivity coefficient (inner)
    mlpdouble sigma_omega2 = mlpdouble(0.856);           // omega diffusivity coefficient (cross-diff)
    mlpdouble nu           = mlpdouble(1.0e-5);          // molecular kinematic viscosity
    mlpdouble omega_min    = mlpdouble(1.0e-10);         // floor to avoid division by zero
    mlpdouble k_min        = mlpdouble(1.0e-10);         // floor for production guard
};

struct FlowState {
    double x;   // spatial x-coordinate (plain double — not differentiated)
    double y;   // spatial y-coordinate (plain double — not differentiated)
    // nu is taken from KOmegaSSTConstants; override here if spatially varying:
    // double nu_local = -1.0;  // if >= 0, overrides KOmegaSSTConstants::nu
};


class CPhysicsLoss : public CBaseLoss {
    /*!
     * \param flow_states  - One FlowState per collocation point.
     * \param constants    - SST model constants (defaults: Menter 1994).
     */

public:
    explicit CPhysicsLoss(
        const std::vector<FLowState>& flow_states,
        const KOmegaSSTConstants& constants = {})
          : CBaseLoss("KOmegaSSTLoss"),
            flow_states_(flow_states),
            constants_(constants) {}
    /*!
     * \brief Compute total SST physics residual at all collocation points.
     *
     * \param predictions  - PredictionResult at each collocation point.
     *                       Must have outputs[0..4] = {u,v,p,k,omega},
     *                       jacobians[iOut][iIn], hessians[iOut][iIn1][iIn2].
     * \param ref_data     - Unused (physics loss needs no labels).
     */

    mlpdouble Evaluate(
        const std::vector<PredictionResult>& predictions,
        const std::vector<std::vector<mlpdouble>>&
    ) override {

        if (predictions.size() != flow_states_.size())
            throw std::runtime_error(
                "CPhysicsLoss: predictions.size() != flow_states_.size()");

        ValidatePredictions(predictions);
        
    }




private:
    std::vector<FlowState> flow_states_;
    KOmegaSSTConstants constants_;

    static void ValidatePredictions(const std::vector<PredictionResult>& preds)
    {
        for (std::size_t i = 0; i < preds.size(); ++i) {
            if (preds[i].outputs.size() != N_OUTPUTS)
                throw std::runtime_error(
                    "CPhysicsLoss: outputs must have 5 entries [u,v,p,k,omega]");
            if (preds[i].jacobians.size() != N_OUTPUTS)
                throw std::runtime_error(
                    "CPhysicsLoss: jacobians must have 5 rows");
            for (auto& row : preds[i].jacobians)
                if (row.size() != N_INPUTS)
                    throw std::runtime_error(
                        "CPhysicsLoss: each jacobian row must have 2 entries [dx,dy]");
            if (preds[i].hessians.size() != N_OUTPUTS)
                throw std::runtime_error(
                    "CPhysicsLoss: hessians must have 5 blocks");
        }
    }
};

} // namespace MLPToolbox