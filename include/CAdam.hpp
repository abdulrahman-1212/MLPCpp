/*!
* \file CAdam.hpp
* \brief Adam optimizer implementation with learning rate annealing.
* \author Abdulrahman.M.Saad
*/


#pragma once

#include "variable_def.hpp"
#include <iostream>

namespace MLPToolbox {

class CAdam {
public:
    // Hyper-parameters default to the values from the original Adam paper.
    explicit CAdam(std::size_t n_params, mlpdouble lr = 1e-3, mlpdouble beta1 = 0.9, mlpdouble beta2 = 0.999, mlpdouble eps = 1e-8):
        lr_(lr),
        beta1_(beta1), beta2_(beta2), eps_(eps),
        m_(n_params, mlpdouble(0)),
        v_(n_params, mlpdouble(0)),
        t_(0)
    {
        if (n_params == 0) 
            throw std::invalid_argument("CAdam: number of parameters must be greater than zero.");
        
        if (lr <= mlpdouble(0))
            throw std::invalid_argument("CAdam: learning rate must be positive.");

        if (beta1 <= mlpdouble(0) || beta1 >= mlpdouble(1))
            throw std::invalid_argument("CAdam: beta1 must be in the interval (0, 1).");

         if (beta2 <= mlpdouble(0) || beta2 >= mlpdouble(1))
            throw std::invalid_argument("CAdam: beta2 must be in the interval (0, 1).");
    
        if (eps <= mlpdouble(0))
            throw std::invalid_argument("CAdam: epsilon must be positive.");

    };

    void Step(std::vector<mlpdouble>& weights, const std::vector<mlpdouble>& grads);

    void Reset() {
        std::fill(m_.begin(), m_.end(), mlpdouble(0));
        std::fill(v_.begin(), v_.end(), mlpdouble(0));
        t_ = 0;
    }
    
    std::size_t GetStep() const noexcept { return t_; }

    // Hyperparameters Getters
    mlpdouble GetLR() const noexcept { return lr_; }
    mlpdouble GetBeta1() const noexcept { return beta1_; }
    mlpdouble GetBeta2() const noexcept { return beta2_; }
    mlpdouble GetEps()   const noexcept { return eps_; }

    // Hyperparameters Setters
    void SetLR(mlpdouble lr) noexcept { lr_ = lr; }



private:
    mlpdouble lr_;
    mlpdouble beta1_, beta2_, eps_;
    std::vector<mlpdouble> m_;   // 1st moment  (same size as weight vector)
    std::vector<mlpdouble> v_;   // 2nd moment

    // mlpdouble weight_decay = 0.0 // just for AdamW
    std::size_t t_{0};        // step counter (used for bias correction)

  };
}