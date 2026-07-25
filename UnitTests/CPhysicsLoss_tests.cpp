#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>
#include <memory>
#include "CPhysicsLoss.hpp"
#include "CBaseLoss.hpp"

using namespace MLPToolbox;

// ============================================================
// Custom Macro
// ============================================================
#define REQUIRE_EQUAL_TOL(a, b, tol) \
    REQUIRE(static_cast<double>(a) == Approx(static_cast<double>(b)).margin(tol))

// ============================================================
// Unit Tests – updated for new CPhysicsLoss implementation
// ============================================================

// -----------------------------------------------------------------
// 1. n_equations == 0 throws
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss n_equations == 0 throws", "[CPhysicsLoss]")
{
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

// -----------------------------------------------------------------
// 2. sparsity flags and indices
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss sparsity flags and indices", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x", "t"};
    std::vector<std::string> net_out = {"u", "p"};
    ResidualFunction dummy_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{1.0};
    };

    // Default flags (jac=true, hess=false)
    CPhysicsLoss loss_default("default", dummy_fn, 1, {"t", "x"}, {"p", "u"},
                              net_in, net_out);
    REQUIRE(loss_default.NumEquations() == 1);
    REQUIRE(loss_default.RequiresJacobian() == true);
    REQUIRE(loss_default.RequiresHessian() == false);

    // Algebraic (no derivatives)
    CPhysicsLoss loss_algebraic("algebraic", dummy_fn, 1, {"x"}, {"u"},
                                net_in, net_out, false, false);
    REQUIRE(loss_algebraic.RequiresJacobian() == false);
    REQUIRE(loss_algebraic.RequiresHessian() == false);

    // Viscous (needs Hessian)
    CPhysicsLoss loss_viscous("viscous", dummy_fn, 1, {"x"}, {"u"},
                              net_in, net_out, true, true);
    REQUIRE(loss_viscous.RequiresJacobian() == true);
    REQUIRE(loss_viscous.RequiresHessian() == true);
}

// -----------------------------------------------------------------
// 3. missing variable throws
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss missing variable throws", "[CPhysicsLoss]")
{
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

// -----------------------------------------------------------------
// 4. residual size mismatch throws
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss residual size mismatch throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};

    // Function returns 2 residuals, but we declare n_equations = 1
    ResidualFunction bad_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{1.0, 2.0};
    };

    // No Jacobian needed, so we can set needs_jacobian=false to avoid
    // nullptr checks on output_Jacobian.
    CPhysicsLoss loss("bad_size", bad_fn, 1, {"x"}, {"u"},
                      net_in, net_out, false, false);

    PredictionResult pred;
    pred.inputs = {1.0};
    pred.outputs = {1.0};
    std::vector<PredictionResult> preds = {pred};
    std::vector<std::vector<mlpdouble>> ref_data;

    REQUIRE_THROWS_AS(loss.Evaluate(preds, ref_data), std::runtime_error);
}

// -----------------------------------------------------------------
// 5. correct MSE computation (no RHS)
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss correct MSE computation", "[CPhysicsLoss]")
{
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

    // No derivatives used -> set flags false
    CPhysicsLoss loss("math_test", math_fn, 2, {"x"}, {"u"},
                      net_in, net_out, false, false);

    PredictionResult pred1;
    pred1.inputs = {1.0};
    pred1.outputs = {1.0};

    PredictionResult pred2;
    pred2.inputs = {2.0};
    pred2.outputs = {2.0};

    std::vector<PredictionResult> preds = {pred1, pred2};
    std::vector<std::vector<mlpdouble>> ref_data;

    mlpdouble result = loss.Evaluate(preds, ref_data);
    // Expected: (13 + 17) / (2 points * 2 equations) = 30 / 4 = 7.5
    REQUIRE_EQUAL_TOL(result, 7.5, 1e-12);
    REQUIRE_EQUAL_TOL(loss.GetLastLossValue(), 7.5, 1e-12);
}

// -----------------------------------------------------------------
// 6. empty predictions returns zero
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss empty predictions", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};
    ResidualFunction dummy_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{1.0};
    };

    CPhysicsLoss loss("empty_test", dummy_fn, 1, {"x"}, {"u"},
                      net_in, net_out, false, false);

    std::vector<PredictionResult> preds;
    std::vector<std::vector<mlpdouble>> ref_data;
    mlpdouble result = loss.Evaluate(preds, ref_data);
    REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);
}

// -----------------------------------------------------------------
// 7. PhysicsState accessor mapping (uses Jacobian and Hessian)
// -----------------------------------------------------------------
TEST_CASE("CPhysicsState accessor mapping", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x", "y", "t"};
    std::vector<std::string> net_out = {"u", "v", "p"};

    std::vector<mlpdouble> captured_in;
    std::vector<mlpdouble> captured_out;
    std::vector<std::vector<mlpdouble>> captured_jac;
    std::vector<std::vector<std::vector<mlpdouble>>> captured_hess;

    ResidualFunction capture_fn = [&](const PhysicsState& s) {
        // eq_in_names = {"t", "x", "y"} -> indices {2, 0, 1}
        captured_in.push_back(s.In(0)); // t -> 30.0
        captured_in.push_back(s.In(1)); // x -> 10.0
        captured_in.push_back(s.In(2)); // y -> 20.0

        // eq_out_names = {"p", "u"} -> indices {2, 0}
        captured_out.push_back(s.Out(0)); // p -> 300.0
        captured_out.push_back(s.Out(1)); // u -> 100.0

        // Jac(out_idx, in_idx)
        // s.Jac(0, 2) -> out="p"(idx 2), in="y"(idx 1) -> pred.output_Jacobian[2][1]
        // s.Jac(1, 0) -> out="u"(idx 0), in="t"(idx 2) -> pred.output_Jacobian[0][2]
        captured_jac.push_back({s.Jac(0, 2), s.Jac(1, 0)});

        // Hess(out_idx, in1_idx, in2_idx)
        // s.Hess(0, 2, 0) -> out="p"(idx 2), in1="y"(idx 1), in2="t"(idx 2)
        captured_hess.push_back({{s.Hess(0, 2, 0)}});

        return std::vector<mlpdouble>{0.0};
    };

    // Needs Jacobian and Hessian (default flags true)
    CPhysicsLoss loss("accessor_test", capture_fn, 1,
                      {"t", "x", "y"}, {"p", "u"},
                      net_in, net_out);

    // ------------------------------------------------------------
    // Build raw Jacobian (3x3) and Hessian (3x3x3) structures
    // ------------------------------------------------------------
    // 1) Jacobian: 2D data stored in vector, then build row pointers
    std::vector<std::vector<mlpdouble>> jac_data(3, std::vector<mlpdouble>(3, 0.0));
    jac_data[0][0] = 1.0;   // du/dx
    jac_data[0][2] = 5.0;   // du/dt
    jac_data[2][1] = 3.0;   // dp/dy

    std::vector<mlpdouble*> jac_ptrs(3);
    for (int i = 0; i < 3; ++i)
        jac_ptrs[i] = jac_data[i].data();

    // 2) Hessian: 3D data stored in nested vectors, then build pointer hierarchy
    std::vector<std::vector<std::vector<mlpdouble>>> hess_data(
        3, std::vector<std::vector<mlpdouble>>(3, std::vector<mlpdouble>(3, 0.0)));
    hess_data[2][1][2] = 4.0;  // d²p/dydt

    // Build intermediate pointers: hess_ptrs2[i][j] points to row j of hess_data[i]
    std::vector<std::vector<mlpdouble*>> hess_rows(3);
    for (int i = 0; i < 3; ++i) {
        hess_rows[i].resize(3);
        for (int j = 0; j < 3; ++j)
            hess_rows[i][j] = hess_data[i][j].data();
    }
    std::vector<mlpdouble**> hess_ptrs2(3);
    for (int i = 0; i < 3; ++i)
        hess_ptrs2[i] = hess_rows[i].data();

    // Now assign to PredictionResult
    PredictionResult pred;
    pred.inputs = {10.0, 20.0, 30.0};   // x=10, y=20, t=30
    pred.outputs = {100.0, 200.0, 300.0}; // u=100, v=200, p=300
    pred.output_Jacobian = jac_ptrs.data();    // double**
    pred.output_Hessian  = hess_ptrs2.data();  // double***

    std::vector<PredictionResult> preds = {pred};
    std::vector<std::vector<mlpdouble>> ref_data;

    loss.Evaluate(preds, ref_data);

    // Verify inputs (t=30, x=10, y=20)
    REQUIRE_EQUAL_TOL(captured_in[0], 30.0, 1e-12);
    REQUIRE_EQUAL_TOL(captured_in[1], 10.0, 1e-12);
    REQUIRE_EQUAL_TOL(captured_in[2], 20.0, 1e-12);

    // Verify outputs (p=300, u=100)
    REQUIRE_EQUAL_TOL(captured_out[0], 300.0, 1e-12);
    REQUIRE_EQUAL_TOL(captured_out[1], 100.0, 1e-12);

    // Verify Jacobians (dp/dy=3.0, du/dt=5.0)
    REQUIRE_EQUAL_TOL(captured_jac[0][0], 3.0, 1e-12);
    REQUIRE_EQUAL_TOL(captured_jac[0][1], 5.0, 1e-12);

    // Verify Hessians (d²p/dydt=4.0)
    REQUIRE_EQUAL_TOL(captured_hess[0][0][0], 4.0, 1e-12);
}

// ============================================================
// New tests for RHS handling and null pointer checks
// ============================================================

// -----------------------------------------------------------------
// 8. uniform RHS via SetUniformRHS
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss uniform RHS via SetUniformRHS", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};

    // Residual = u - Rhs(0)  (so loss should be zero if u == Rhs)
    ResidualFunction rhs_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{s.Out(0) - s.Rhs(0)};
    };

    CPhysicsLoss loss("uniform_rhs", rhs_fn, 1, {"x"}, {"u"},
                      net_in, net_out, false, false);

    // Set uniform RHS = 3.0
    loss.SetUniformRHS({3.0});

    PredictionResult pred;
    pred.inputs = {1.0};
    pred.outputs = {3.0};   // exactly matches RHS -> residual = 0

    std::vector<PredictionResult> preds = {pred};
    std::vector<std::vector<mlpdouble>> ref_data;  // empty -> uses uniform RHS

    mlpdouble result = loss.Evaluate(preds, ref_data);
    REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);

    // Now set output to 5.0, residual = 2.0, squared loss = 4.0 / (1*1) = 4.0
    pred.outputs = {5.0};
    preds = {pred};
    result = loss.Evaluate(preds, ref_data);
    REQUIRE_EQUAL_TOL(result, 4.0, 1e-12);
}

// -----------------------------------------------------------------
// 9. per-point RHS via ref_data
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss per-point RHS via ref_data", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};

    ResidualFunction rhs_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{s.Out(0) - s.Rhs(0)};
    };

    CPhysicsLoss loss("per_point_rhs", rhs_fn, 1, {"x"}, {"u"},
                      net_in, net_out, false, false);

    // Two points
    PredictionResult pred1; pred1.inputs = {1.0}; pred1.outputs = {2.0};
    PredictionResult pred2; pred2.inputs = {2.0}; pred2.outputs = {5.0};
    std::vector<PredictionResult> preds = {pred1, pred2};

    // ref_data: per-point RHS: [1.0] for first point, [3.0] for second
    std::vector<std::vector<mlpdouble>> ref_data = {{1.0}, {3.0}};

    // Residuals: (2-1)=1, (5-3)=2 -> squares sum = 1+4=5 -> loss = 5 / (2*1) = 2.5
    mlpdouble result = loss.Evaluate(preds, ref_data);
    REQUIRE_EQUAL_TOL(result, 2.5, 1e-12);

    // Also test that uniform RHS is ignored when ref_data is non-empty
    loss.SetUniformRHS({0.0});
    result = loss.Evaluate(preds, ref_data);
    REQUIRE_EQUAL_TOL(result, 2.5, 1e-12);  // still uses per-point
}

// -----------------------------------------------------------------
// 10. RHS size mismatch throws
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss RHS size mismatch throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};

    // FIX: return correct number of residuals (2) so that RHS size checks are reached.
    ResidualFunction dummy_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{0.0, 0.0};
    };

    CPhysicsLoss loss("rhs_size", dummy_fn, 2, {"x"}, {"u"},
                      net_in, net_out, false, false);

    PredictionResult pred; pred.inputs = {1.0}; pred.outputs = {1.0};
    std::vector<PredictionResult> preds = {pred};

    // ref_data row size = 3, expected 0, 1, or 2 -> throws
    std::vector<std::vector<mlpdouble>> ref_data = {{1.0, 2.0, 3.0}};
    REQUIRE_THROWS_AS(loss.Evaluate(preds, ref_data), std::runtime_error);

    // row size = 1 is allowed (applied to all equations)
    ref_data = {{5.0}};
    REQUIRE_NOTHROW(loss.Evaluate(preds, ref_data));

    // row size = 2 is allowed
    ref_data = {{5.0, 6.0}};
    REQUIRE_NOTHROW(loss.Evaluate(preds, ref_data));
}

// -----------------------------------------------------------------
// 11. default RHS is zero
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss default RHS zero", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};

    ResidualFunction check_rhs = [](const PhysicsState& s) {
        // Rhs returns 0 by default
        return std::vector<mlpdouble>{s.Rhs(0)};
    };

    CPhysicsLoss loss("default_zero", check_rhs, 1, {"x"}, {"u"},
                      net_in, net_out, false, false);

    PredictionResult pred; pred.inputs = {1.0}; pred.outputs = {1.0};
    std::vector<PredictionResult> preds = {pred};

    // No uniform RHS set, ref_data empty -> Rhs(0) should be 0
    std::vector<std::vector<mlpdouble>> ref_data;
    mlpdouble result = loss.Evaluate(preds, ref_data);
    // residual = 0, loss = 0
    REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);
}

// -----------------------------------------------------------------
// 12. Jacobian nullptr throws
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss Jacobian nullptr throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};

    // Residual uses Jac (needs_jacobian=true by default)
    ResidualFunction jac_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{s.Jac(0,0)};
    };

    CPhysicsLoss loss("jac_null", jac_fn, 1, {"x"}, {"u"},
                      net_in, net_out);  // default needs_jacobian=true

    PredictionResult pred;
    pred.inputs = {1.0};
    pred.outputs = {1.0};
    pred.output_Jacobian = nullptr;   // explicitly null

    std::vector<PredictionResult> preds = {pred};
    std::vector<std::vector<mlpdouble>> ref_data;

    REQUIRE_THROWS_AS(loss.Evaluate(preds, ref_data), std::runtime_error);
}

// -----------------------------------------------------------------
// 13. Hessian nullptr throws
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss Hessian nullptr throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};

    // Residual uses Hess (needs_hessian=true)
    ResidualFunction hess_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{s.Hess(0,0,0)};
    };

    CPhysicsLoss loss("hess_null", hess_fn, 1, {"x"}, {"u"},
                      net_in, net_out, true, true);  // jac=true, hess=true

    // Allocate a dummy Jacobian (needs_jacobian is true, so it must not be null)
    std::vector<std::vector<mlpdouble>> jac_data(1, std::vector<mlpdouble>(1, 0.0));
    std::vector<mlpdouble*> jac_ptrs(1, jac_data[0].data());

    PredictionResult pred;
    pred.inputs = {1.0};
    pred.outputs = {1.0};
    pred.output_Jacobian = jac_ptrs.data();   // valid pointer
    pred.output_Hessian = nullptr;            // explicitly null

    std::vector<PredictionResult> preds = {pred};
    std::vector<std::vector<mlpdouble>> ref_data;

    REQUIRE_THROWS_AS(loss.Evaluate(preds, ref_data), std::runtime_error);
}

// -----------------------------------------------------------------
// 14. SetUniformRHS size mismatch throws
// -----------------------------------------------------------------
TEST_CASE("CPhysicsLoss SetUniformRHS size mismatch throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in = {"x"};
    std::vector<std::string> net_out = {"u"};
    ResidualFunction dummy_fn = [](const PhysicsState& s) {
        return std::vector<mlpdouble>{0.0};
    };

    CPhysicsLoss loss("uniform_size", dummy_fn, 2, {"x"}, {"u"},
                      net_in, net_out, false, false);

    // n_equations = 2, so setting RHS of size 1 should throw
    REQUIRE_THROWS_AS(loss.SetUniformRHS({1.0}), std::invalid_argument);
    // size 2 is fine
    REQUIRE_NOTHROW(loss.SetUniformRHS({1.0, 2.0}));
}