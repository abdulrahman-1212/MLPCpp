#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>

#include "CPhysicsLoss.hpp"
#include "CBaseLoss.hpp"

using namespace MLPToolbox;

// ============================================================
//  Custom Macro
// ============================================================
#define REQUIRE_EQUAL_TOL(a, b, tol) \
    REQUIRE(static_cast<double>(a) == Approx(static_cast<double>(b)).margin(tol))

// ============================================================
//  Unit Tests
// ============================================================

TEST_CASE("CPhysicsLoss n_equations == 0 throws", "[CPhysicsLoss]") {
    std::vector<std::string> net_in = {"x", "t"};
    std::vector<std::string> net_out = {"u", "p"};
    
    ResidualFunction dummy_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{1.0};
    };

    REQUIRE_THROWS_AS(
        CPhysicsLoss("test", dummy_fn, 0, {"x"}, {"u"}, net_in, net_out),
        std::invalid_argument
    );
}

TEST_CASE("CPhysicsLoss sparsity flags and indices", "[CPhysicsLoss]") {
    std::vector<std::string> net_in = {"x", "t"};
    std::vector<std::string> net_out = {"u", "p"};
    
    ResidualFunction dummy_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{1.0};
    };

    // Test default flags
    CPhysicsLoss loss_default("default", dummy_fn, 1, {"t", "x"}, {"p", "u"}, net_in, net_out);
    REQUIRE(loss_default.NumEquations() == 1);
    REQUIRE(loss_default.RequiresJacobian() == true);
    REQUIRE(loss_default.RequiresHessian() == false);

    // Test custom sparsity flags (e.g., algebraic constraint)
    CPhysicsLoss loss_algebraic("algebraic", dummy_fn, 1, {"x"}, {"u"}, net_in, net_out, false, false);
    REQUIRE(loss_algebraic.RequiresJacobian() == false);
    REQUIRE(loss_algebraic.RequiresHessian() == false);

    // Test Hessian requirement (e.g., viscous terms)
    CPhysicsLoss loss_viscous("viscous", dummy_fn, 1, {"x"}, {"u"}, net_in, net_out, true, true);
    REQUIRE(loss_viscous.RequiresJacobian() == true);
    REQUIRE(loss_viscous.RequiresHessian() == true);
}

TEST_CASE("CPhysicsLoss missing variable throws", "[CPhysicsLoss]") {
    std::vector<std::string> net_in = {"x", "t"};
    std::vector<std::string> net_out = {"u", "p"};
    
    ResidualFunction dummy_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{1.0};
    };

    // Missing input variable
    REQUIRE_THROWS_AS(
        CPhysicsLoss("test", dummy_fn, 1, {"z"}, {"u"}, net_in, net_out),
        std::runtime_error
    );

    // Missing output variable
    REQUIRE_THROWS_AS(
        CPhysicsLoss("test", dummy_fn, 1, {"x"}, {"v"}, net_in, net_out),
        std::runtime_error
    );
}

TEST_CASE("CPhysicsLoss residual size mismatch throws", "[CPhysicsLoss]") {
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};
    
    // Function returns 2 residuals, but we declare n_equations = 1
    ResidualFunction bad_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{1.0, 2.0};
    };

    CPhysicsLoss loss("bad_size", bad_fn, 1, {"x"}, {"u"}, net_in, net_out);

    PredictionResult pred;
    pred.inputs = {1.0};
    pred.outputs = {1.0};
    
    std::vector<PredictionResult> preds = {pred};
    std::vector<std::vector<mlpdouble>> ref_data;

    REQUIRE_THROWS_AS(loss.Evaluate(preds, ref_data), std::runtime_error);
}

TEST_CASE("CPhysicsLoss correct MSE computation", "[CPhysicsLoss]") {
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};
    
    int call_count = 0;
    ResidualFunction math_fn = [&call_count](const PhysicsState& s) {
        if (call_count == 0) {
            call_count++;
            return std::vector<mlpdouble>{2.0, 3.0}; // 2^2 + 3^2 = 13
        } else {
            call_count++;
            return std::vector<mlpdouble>{1.0, 4.0}; // 1^2 + 4^2 = 17
        }
    };

    CPhysicsLoss loss("math_test", math_fn, 2, {"x"}, {"u"}, net_in, net_out);

    PredictionResult pred1; pred1.inputs = {1.0}; pred1.outputs = {1.0};
    PredictionResult pred2; pred2.inputs = {2.0}; pred2.outputs = {2.0};

    std::vector<PredictionResult> preds = {pred1, pred2};
    std::vector<std::vector<mlpdouble>> ref_data;

    mlpdouble result = loss.Evaluate(preds, ref_data);
    
    // Expected: (13 + 17) / (2 points * 2 equations) = 30 / 4 = 7.5
    REQUIRE_EQUAL_TOL(result, 7.5, 1e-12);
    REQUIRE_EQUAL_TOL(loss.GetLastLossValue(), 7.5, 1e-12);
}

TEST_CASE("CPhysicsLoss empty predictions", "[CPhysicsLoss]") {
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};
    
    ResidualFunction dummy_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{1.0};
    };

    CPhysicsLoss loss("empty_test", dummy_fn, 1, {"x"}, {"u"}, net_in, net_out);

    std::vector<PredictionResult> preds;
    std::vector<std::vector<mlpdouble>> ref_data;

    mlpdouble result = loss.Evaluate(preds, ref_data);
    REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);
}

TEST_CASE("CPhysicsState accessor mapping", "[CPhysicsLoss]") {
    std::vector<std::string> net_in = {"x", "y", "t"};
    std::vector<std::string> net_out = {"u", "v", "p"};
    
    std::vector<mlpdouble> captured_in;
    std::vector<mlpdouble> captured_out;
    std::vector<std::vector<mlpdouble>> captured_jac;
    std::vector<std::vector<std::vector<mlpdouble>>> captured_hess;

    ResidualFunction capture_fn = [&](const PhysicsState& s) {
        // eq_in_names is {"t", "x", "y"} -> maps to network indices {2, 0, 1}
        captured_in.push_back(s.In(0));  // t -> 30.0
        captured_in.push_back(s.In(1));  // x -> 10.0
        captured_in.push_back(s.In(2));  // y -> 20.0
        
        // eq_out_names is {"p", "u"} -> maps to network indices {2, 0}
        captured_out.push_back(s.Out(0)); // p -> 300.0
        captured_out.push_back(s.Out(1)); // u -> 100.0
        
        // Jac(out_idx, in_idx)
        // s.Jac(0, 2) -> out="p"(idx 2), in="y"(idx 1) -> pred.jacobians[2][1]
        // s.Jac(1, 0) -> out="u"(idx 0), in="t"(idx 2) -> pred.jacobians[0][2]
        captured_jac.push_back({s.Jac(0, 2), s.Jac(1, 0)}); 
        
        // Hess(out_idx, in1_idx, in2_idx)
        // s.Hess(0, 2, 0) -> out="p"(idx 2), in1="y"(idx 1), in2="t"(idx 2) -> pred.hessians[2][1][2]
        captured_hess.push_back({{s.Hess(0, 2, 0)}}); 
        
        return std::vector<mlpdouble>{0.0};
    };

    // Note: eq_in_names now includes "y" so that index 2 is valid
    CPhysicsLoss loss("accessor_test", capture_fn, 1, {"t", "x", "y"}, {"p", "u"}, net_in, net_out);

    // Setup mock PredictionResult with distinct values for easy verification
    PredictionResult pred;
    pred.inputs = {10.0, 20.0, 30.0};       // x=10, y=20, t=30
    pred.outputs = {100.0, 200.0, 300.0};   // u=100, v=200, p=300
    
    pred.jacobians.resize(3, std::vector<mlpdouble>(3, 0.0));
    pred.jacobians[0][0] = 1.0; // du/dx (unused in test, but good for completeness)
    pred.jacobians[0][2] = 5.0; // du/dt
    pred.jacobians[2][1] = 3.0; // dp/dy

    pred.hessians.resize(3, std::vector<std::vector<mlpdouble>>(3, std::vector<mlpdouble>(3, 0.0)));
    pred.hessians[2][1][2] = 4.0; // d2p/dydt

    std::vector<PredictionResult> preds = {pred};
    std::vector<std::vector<mlpdouble>> ref_data;

    loss.Evaluate(preds, ref_data);

    // Verify Inputs (t=30, x=10, y=20)
    REQUIRE_EQUAL_TOL(captured_in[0], 30.0, 1e-12);
    REQUIRE_EQUAL_TOL(captured_in[1], 10.0, 1e-12);
    REQUIRE_EQUAL_TOL(captured_in[2], 20.0, 1e-12);
    
    // Verify Outputs (p=300, u=100)
    REQUIRE_EQUAL_TOL(captured_out[0], 300.0, 1e-12);
    REQUIRE_EQUAL_TOL(captured_out[1], 100.0, 1e-12);
    
    // Verify Jacobians (dp/dy=3.0, du/dt=5.0)
    REQUIRE_EQUAL_TOL(captured_jac[0][0], 3.0, 1e-12);
    REQUIRE_EQUAL_TOL(captured_jac[0][1], 5.0, 1e-12);
    
    // Verify Hessians (d2p/dydt=4.0)
    REQUIRE_EQUAL_TOL(captured_hess[0][0][0], 4.0, 1e-12);
}