#pragma one
#include <stdexcept>
#include "CBaseLoss.hpp"

namespace MLPToolbox {


class CDataLoss : CBaseLoss {
public:
    CDataLoss() : CBaseLoss("data loss") {}

     /*!
     * \brief Evaluate MSE loss.
     * \param predictions - Network predictions (outputs[0] is y_pred)
     * \param ref_data - Reference data (ref_data[i][0] is y_true)
     */
    mlpdouble Evaluate(
        const std::vector<PredictionResult>& predictions,
        const std::vector<std::vector<mlpdouble>>& ref_data
    ) override {
        if (predictions.size() != ref_data.size()) {
            throw std::runtime_error("CDataLoss: predictions and ref_data size mismatch");
        }
        
        if (predictions.empty()) {
            last_loss_value_ = 0.0;
            return mlpdouble(0.0);
        }

        mlpdouble mse = 0.0;
        const size_t N = predictions.size();
        const size_t n_outputs = predictions[0].outputs.size();

        // Validate consistency across all points
        for (size_t i = 0; i < N; ++i) {
            if (predictions[i].outputs.size() != n_outputs) {
                throw std::runtime_error(
                    "CDataLoss: inconsistent number of outputs in predictions");
            }

            if (ref_data[i].size() != n_outputs) {
                throw std::runtime_error(
                    "CDataLoss: ref_data[i] size does not match number of outputs");
            }
        }

       for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < n_outputs; ++j) {
                mlpdouble y_pred = predictions[i].outputs[j];
                mlpdouble y_true = ref_data[i][j];
                mlpdouble diff = y_pred - y_true;
                mse += diff * diff;
            }
        }

        mlpdouble loss = mse / static_cast<mlpdouble>(N * n_outputs);
        last_loss_value_ = loss.getValue();
        return loss;
    }
    
};


}   // namespace MLPToolbox