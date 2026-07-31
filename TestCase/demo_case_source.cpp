/*!
 * \file demo_case_source.cpp
 * \brief Simple PINN demo with data loss + physics residual that includes
 *        a source term supplied via PhysicsData.
 *
 * Manufactured solution
 * ---------------------
 *   Domain      : (u,v) ∈ [-1,1] × [-1,1]
 *   Solution    : y(u,v) = u^2 + 0.5·v
 *   PDE         : dy/du - s(u,v) = 0
 *   Source      : s(u,v) = 2u
 *
 * Training setup
 * --------------
 *   - 200 labelled points for the data (MSE) loss
 *   - 300 collocation points for the physics residual
 *   - Source values are stored per collocation point and passed
 *     through PhysicsData (name "s")
 *   - Fixed lambda = 1 (annealer off) for clarity
 *   - Network: 2 -> 32 -> 32 -> 1, tanh, min-max scaling
 */

#define MLP_CUSTOM_TYPE codi::RealReverse
#include "codi.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "CNeuralNetwork.hpp"
#include "CAdam.hpp"
#include "CGradientAnnealer.hpp"
#include "CMLPTrainer.hpp"
#include "CPhysicsLoss.hpp"
#include "CDataLoss.hpp"
#include "variable_def.hpp"

using namespace MLPToolbox;

// ============================================================================
// Manufactured solution and source
// ============================================================================
static double exact_y(double u, double v) {
    return u * u + 0.5 * v;
}

static double source_s(double u, double /*v*/) {
    return 2.0 * u;                     // dy/du of the exact solution
}

// ============================================================================
// Main
// ============================================================================
int main() {
    // -------------------------------------------------------------------------
    // 1. Generate synthetic labelled data
    // -------------------------------------------------------------------------
    const std::size_t N_data = 200;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    std::vector<std::vector<mlpdouble>> X(N_data, std::vector<mlpdouble>(2));
    std::vector<std::vector<mlpdouble>> Y(N_data, std::vector<mlpdouble>(1));

    double u_min = 1e9, u_max = -1e9;
    double v_min = 1e9, v_max = -1e9;
    double y_min = 1e9, y_max = -1e9;

    for (std::size_t i = 0; i < N_data; ++i) {
        const double u = uni(rng);
        const double v = uni(rng);
        const double y = exact_y(u, v);

        X[i][0] = mlpdouble(u);
        X[i][1] = mlpdouble(v);
        Y[i][0] = mlpdouble(y);

        u_min = std::min(u_min, u); u_max = std::max(u_max, u);
        v_min = std::min(v_min, v); v_max = std::max(v_max, v);
        y_min = std::min(y_min, y); y_max = std::max(y_max, y);
    }

    std::cout << "Generated " << N_data << " labelled points.\n"
              << "  u ∈ [" << u_min << ", " << u_max << "]\n"
              << "  v ∈ [" << v_min << ", " << v_max << "]\n"
              << "  y ∈ [" << y_min << ", " << y_max << "]\n";

    // -------------------------------------------------------------------------
    // 2. Build network
    // -------------------------------------------------------------------------
    std::vector<std::size_t> architecture = {2, 32, 32, 1};
    CNeuralNetwork net(architecture);
    net.SetActivationFunction("tanh");

    net.SetInputRegularization("minmax");
    net.SetInputNorm(0, static_cast<mlpdouble>(u_min), static_cast<mlpdouble>(u_max));
    net.SetInputNorm(1, static_cast<mlpdouble>(v_min), static_cast<mlpdouble>(v_max));

    net.SetOutputRegularization("minmax");
    net.SetOutputNorm(0, static_cast<mlpdouble>(y_min), static_cast<mlpdouble>(y_max));

    net.SetInputName(0, "u");
    net.SetInputName(1, "v");
    net.SetOutputName(0, "y");

    net.RandomWeights();
    net.DisplayNetwork();

    // -------------------------------------------------------------------------
    // 3. Physics residual with source term
    //
    //    residual = dy/du - s
    //    s is supplied per collocation point through PhysicsData ("s")
    // -------------------------------------------------------------------------
    CPhysicsEquation eq;
    eq.name               = "dy_du_minus_s";
    eq.input_names        = {"u"};
    eq.output_names       = {"y"};
    eq.weight             = 1.0;
    eq.requires_jacobian  = true;
    eq.requires_hessian   = false;

    eq.residual = [](const PhysicsState& state, const PhysicsData& data) -> mlpdouble {
        // dy/du  (equation-local indices: input 0 = "u", output 0 = "y")
        const mlpdouble dydu = state.EquationJac(0, 0);
        // source term stored under the name "s"
        const mlpdouble s = data.At("s");
        return dydu - s;
    };

    auto physics_loss = std::make_shared<CPhysicsLoss>(
        "poisson_like",
        net.GetInputVars(),
        net.GetOutputVars(),
        std::vector<std::string>{"s"},          // one physics variable: the source
        std::vector<CPhysicsEquation>{eq}
    );

    // -------------------------------------------------------------------------
    // 4. Optimizer & trainer configuration
    // -------------------------------------------------------------------------
    CAdam optimizer(1e-3, 0.9, 0.999, 1e-8);

    AnnealerConfig anneal_cfg;                  // unused (annealer off)
    anneal_cfg.n_data_terms = 1;
    anneal_cfg.lambda_init  = 1.0;
    anneal_cfg.lambda_max   = 10;

    TrainerConfig trainer_cfg;
    trainer_cfg.max_epochs         = 300;
    trainer_cfg.batch_size         = 64;
    trainer_cfg.physics_batch_size = 0;         // full collocation batch
    trainer_cfg.use_annealer       = false;     // fixed lambda = 1
    trainer_cfg.verbose            = true;
    trainer_cfg.log_every          = 20;
    trainer_cfg.shuffle_per_epoch  = true;

    CMLPTrainer trainer(net, std::move(optimizer), anneal_cfg, trainer_cfg);
    trainer.SetTrainingData(X, Y);

    // -------------------------------------------------------------------------
    // 5. Collocation points + per-point source values
    // -------------------------------------------------------------------------
    const int Ncoll = 300;
    std::vector<std::vector<mlpdouble>> physics_points(Ncoll, std::vector<mlpdouble>(2));
    std::vector<std::vector<mlpdouble>> source_data(Ncoll, std::vector<mlpdouble>(1));

    for (int i = 0; i < Ncoll; ++i) {
        // uniform random points inside the data box
        const double u = uni(rng);
        const double v = uni(rng);
        physics_points[i][0] = mlpdouble(u);
        physics_points[i][1] = mlpdouble(v);
        source_data[i][0]    = mlpdouble(source_s(u, v));   // s = 2u
    }

    trainer.SetCollocationPoints("colloc", physics_points);

    // Register the loss and attach the source values for every collocation point
    trainer.AddPhysicsLoss(physics_loss, "colloc", source_data);

    trainer.Build();

    // -------------------------------------------------------------------------
    // 6. Train
    // -------------------------------------------------------------------------
    std::cout << "\nStarting training (" << trainer_cfg.max_epochs << " epochs)...\n"
              << "Physics residual : dy/du - s   with s = 2u\n"
              << "Collocation pts  : " << Ncoll << "\n"
              << "Annealer         : off\n\n";

    std::cout << std::left
              << std::setw(8)  << "Epoch"
              << std::setw(16) << "L_data"
              << std::setw(16) << "L_phys"
              << std::setw(12) << "lambda"
              << std::setw(16) << "L_total"
              << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (std::size_t epoch = 0; epoch < trainer_cfg.max_epochs; ++epoch) {
        trainer.TrainEpoch();
        const TrainStepResult& result = trainer.GetLastResult();

        if (epoch % 20 == 0 || epoch == trainer_cfg.max_epochs - 1) {
            const double lambda = result.lambdas.empty() ? 1.0 : result.lambdas[0];
            std::cout << std::left
                      << std::setw(8)  << (epoch + 1)
                      << std::setw(16) << std::scientific << std::setprecision(4) << result.loss_ref
                      << std::setw(16) << result.loss_phys[0]
                      << std::setw(12) << std::fixed << std::setprecision(4) << lambda
                      << std::setw(16) << std::scientific << std::setprecision(4) << result.loss_total
                      << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // 7. Quick diagnostic on a few test locations
    // -------------------------------------------------------------------------
    const TrainStepResult& final = trainer.GetLastResult();
    std::cout << "\nTraining finished.\n"
              << "Final L_data  = " << std::scientific << final.loss_ref  << "\n"
              << "Final L_phys  = " << final.loss_phys[0] << "\n"
              << "Final L_total = " << final.loss_total << "\n\n";

    std::cout << "Point-wise check (network vs exact):\n";
    std::cout << std::setw(10) << "u" << std::setw(10) << "v"
              << std::setw(14) << "y_net" << std::setw(14) << "y_exact"
              << std::setw(14) << "dy/du" << std::setw(14) << "s=2u" << "\n";

    const std::vector<std::pair<double,double>> probes = {
        {0.0, 0.0}, {0.5, -0.3}, {-0.7, 0.8}, {0.9, 0.9}, {-0.4, -0.6}
    };

    for (const auto& [u, v] : probes) {
        std::vector<mlpdouble> x = {mlpdouble(u), mlpdouble(v)};
        net.Predict(x, true, false);                // request Jacobian
        const double y_net  = static_cast<double>(net.GetOutput(0).getValue());
        const double dydu   = static_cast<double>(net.GetJacobian(0, 0).getValue());
        const double y_ex   = exact_y(u, v);
        const double s_ex   = source_s(u, v);

        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(10) << u << std::setw(10) << v
                  << std::setw(14) << y_net << std::setw(14) << y_ex
                  << std::setw(14) << dydu  << std::setw(14) << s_ex << "\n";
    }

    net.WriteNeuralNetwork("trained_model_source.mlp");
    std::cout << "\nModel written to trained_model_source.mlp\n";
    return 0;
}
