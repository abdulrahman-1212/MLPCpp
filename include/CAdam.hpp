/*!
* \file CAdam.hpp
* \brief Adam optimizer implementation with learning rate annealing.
* \author Abdulrahman.M.Saad
*/


#pragma once
namespace MLPToolbox {

class CAdam {
public:
    // n_params: total number of trainable scalars (must equal GetWeightsBiases().size()).
    // Hyper-parameters default to the values from the original Adam paper.
    explicit CAdam(std::size_t n_params,
                 double      lr    = 3e-3,
                 double      beta1 = 0.9,
                 double      beta2 = 0.999,
                 double      eps   = 1e-8);

    // Apply one gradient-descent step.
    // weights: flat vector from GetWeightsBiases() — modified in-place.
    // grads:   d(L)/d(w_i) for each element, same ordering as weights.
    // Throws std::length_error if sizes do not match n_params.
    void Step(std::vector<mlpdouble>& weights,
                const std::vector<double>& grads);

    void Reset()

    double GetLR()     const { return lr_; }
    void SetLR(double lr)  { lr_ = lr; }



private:
    double      lr_;
    double      beta1_, beta2_, eps_;
    std::vector<double> m_;   // 1st moment  (same size as weight vector)
    std::vector<double> v_;   // 2nd moment
    std::size_t t_{0};        // step counter (used for bias correction)

  }
}