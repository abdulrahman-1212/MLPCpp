#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#include "CPhysicsLoss.hpp"
#include "CBaseLoss.hpp"

using namespace MLPToolbox;

// ============================================================
// Helpers
// ============================================================

#define REQUIRE_EQUAL_TOL(a, b, tol) \
    REQUIRE(static_cast<double>(a) == Approx(static_cast<double>(b)).margin(tol))

// Layout: jacobian[input][output]
static mlpdouble** AllocateJacobian(std::size_t n_inputs, std::size_t n_outputs)
{
    mlpdouble** jac = new mlpdouble*[n_inputs];
    for (std::size_t i = 0; i < n_inputs; ++i) {
        jac[i] = new mlpdouble[n_outputs];
        for (std::size_t o = 0; o < n_outputs; ++o) {
            jac[i][o] = 0.0;
        }
    }
    return jac;
}

// Layout: hessian[input_i][input_j][output]
static mlpdouble*** AllocateHessian(std::size_t n_inputs, std::size_t n_outputs)
{
    mlpdouble*** hess = new mlpdouble**[n_inputs];
    for (std::size_t i = 0; i < n_inputs; ++i) {
        hess[i] = new mlpdouble*[n_inputs];
        for (std::size_t j = 0; j < n_inputs; ++j) {
            hess[i][j] = new mlpdouble[n_outputs];
            for (std::size_t o = 0; o < n_outputs; ++o) {
                hess[i][j][o] = 0.0;
            }
        }
    }
    return hess;
}

static void FreeJacobian(mlpdouble** jac, std::size_t n_inputs)
{
    if (!jac) return;
    for (std::size_t i = 0; i < n_inputs; ++i) delete[] jac[i];
    delete[] jac;
}

static void FreeHessian(mlpdouble*** hess, std::size_t n_inputs)
{
    if (!hess) return;
    for (std::size_t i = 0; i < n_inputs; ++i) {
        for (std::size_t j = 0; j < n_inputs; ++j) delete[] hess[i][j];
        delete[] hess[i];
    }
    delete[] hess;
}

// PredictionResult now owns its input/output vectors.
// Jacobian/Hessian pointers use the new field names and input-major layout.
static PredictionResult MakePrediction(
    std::vector<mlpdouble> inputs,
    std::vector<mlpdouble> outputs,
    mlpdouble** jac = nullptr,
    mlpdouble*** hess = nullptr)
{
    PredictionResult pred;
    pred.inputs  = std::move(inputs);
    pred.outputs = std::move(outputs);
    pred.jacobian = jac;
    pred.hessian  = hess;
    return pred;
}

// ============================================================
// 1. Empty equations vector throws
// ============================================================

TEST_CASE("CPhysicsLoss empty equations throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x", "t"};
    std::vector<std::string> net_out = {"u", "p"};

    REQUIRE_THROWS_AS(
        CPhysicsLoss("test", net_in, net_out, {}, {}),
        std::invalid_argument);
}

// ============================================================
// 2. Sparsity flags per equation
// ============================================================

TEST_CASE("CPhysicsLoss sparsity flags", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x", "t"};
    std::vector<std::string> net_out = {"u", "p"};

    auto dummy_residual = [](const PhysicsState&, const PhysicsData&) {
        return mlpdouble(1.0);
    };

    // Algebraic: no derivatives needed
    CPhysicsEquation eq_algebraic;
    eq_algebraic.name = "alg";
    eq_algebraic.input_names  = {"x"};
    eq_algebraic.output_names = {"u"};
    eq_algebraic.requires_jacobian = false;
    eq_algebraic.requires_hessian  = false;
    eq_algebraic.residual = dummy_residual;

    CPhysicsLoss loss_algebraic(
        "algebraic", net_in, net_out, {},
        {eq_algebraic});

    REQUIRE(loss_algebraic.RequiresJacobian() == false);
    REQUIRE(loss_algebraic.RequiresHessian()  == false);

    // First-order: Jacobian needed
    CPhysicsEquation eq_jac;
    eq_jac.name = "jac_eq";
    eq_jac.input_names  = {"x"};
    eq_jac.output_names = {"u"};
    eq_jac.requires_jacobian = true;
    eq_jac.requires_hessian  = false;
    eq_jac.residual = [](const PhysicsState& s, const PhysicsData&) {
        return s.EquationJac(0, 0);  // uses Jacobian
    };

    CPhysicsLoss loss_jac(
        "jac_only", net_in, net_out, {},
        {eq_jac});

    REQUIRE(loss_jac.RequiresJacobian() == true);
    REQUIRE(loss_jac.RequiresHessian()  == false);

    // Second-order: both needed
    CPhysicsEquation eq_hess;
    eq_hess.name = "hess_eq";
    eq_hess.input_names  = {"x"};
    eq_hess.output_names = {"u"};
    eq_hess.requires_jacobian = true;
    eq_hess.requires_hessian  = true;
    eq_hess.residual = [](const PhysicsState& s, const PhysicsData&) {
        return s.EquationHess(0, 0, 0);  // uses Hessian
    };

    CPhysicsLoss loss_hess(
        "hess_full", net_in, net_out, {},
        {eq_hess});

    REQUIRE(loss_hess.RequiresJacobian() == true);
    REQUIRE(loss_hess.RequiresHessian()  == true);

    // Mixed: one equation needs jac, another needs hess
    CPhysicsLoss loss_mixed(
        "mixed", net_in, net_out, {},
        {eq_jac, eq_hess});

    REQUIRE(loss_mixed.RequiresJacobian() == true);
    REQUIRE(loss_mixed.RequiresHessian()  == true);
}

// ============================================================
// 3. Missing network variable throws
// ============================================================

TEST_CASE("CPhysicsLoss missing network variable throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x", "t"};
    std::vector<std::string> net_out = {"u", "p"};

    CPhysicsEquation eq;
    eq.name = "bad_input";
    eq.input_names  = {"z"};   // not in net_in
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    REQUIRE_THROWS_AS(
        CPhysicsLoss("test", net_in, net_out, {}, {eq}),
        std::invalid_argument);

    eq.name = "bad_output";
    eq.input_names  = {"x"};
    eq.output_names = {"v"};   // not in net_out

    REQUIRE_THROWS_AS(
        CPhysicsLoss("test", net_in, net_out, {}, {eq}),
        std::invalid_argument);
}

// ============================================================
// 4. Duplicate physics variable names throws
// ============================================================

TEST_CASE("CPhysicsLoss duplicate physics names throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "dup_phys";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    REQUIRE_THROWS_AS(
        CPhysicsLoss("dup_physics", net_in, net_out, {"fx", "fx"}, {eq}),
        std::invalid_argument);
}

// ============================================================
// 5. Correct MSE computation (per-equation residuals)
// ============================================================

TEST_CASE("CPhysicsLoss correct MSE with per-equation residuals", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    int eq1_calls = 0;
    int eq2_calls = 0;

    CPhysicsEquation eq1;
    eq1.name = "eq1";
    eq1.input_names  = {"x"};
    eq1.output_names = {"u"};
    eq1.residual = [&eq1_calls](const PhysicsState&, const PhysicsData&) {
        ++eq1_calls;
        return eq1_calls == 1 ? mlpdouble(2.0) : mlpdouble(1.0);
    };

    CPhysicsEquation eq2;
    eq2.name = "eq2";
    eq2.input_names  = {"x"};
    eq2.output_names = {"u"};
    eq2.residual = [&eq2_calls](const PhysicsState&, const PhysicsData&) {
        ++eq2_calls;
        return eq2_calls == 1 ? mlpdouble(3.0) : mlpdouble(4.0);
    };

    CPhysicsLoss loss(
        "math_test", net_in, net_out, {},
        {eq1, eq2});

    PredictionResult pred1 = MakePrediction({1.0}, {1.0});
    PredictionResult pred2 = MakePrediction({2.0}, {2.0});

    mlpdouble result = loss.Evaluate({pred1, pred2}, {});

    // Point 1: eq1=2, eq2=3  -> 4 + 9 = 13
    // Point 2: eq1=1, eq2=4  -> 1 + 16 = 17
    // Normalized: (13 + 17) / (2 points * 2 equations) = 30 / 4 = 7.5
    REQUIRE_EQUAL_TOL(result, 7.5, 1e-12);
    REQUIRE_EQUAL_TOL(loss.GetLastLossValue(), 7.5, 1e-12);
}

// ============================================================
// 6. Empty predictions
// ============================================================

TEST_CASE("CPhysicsLoss empty predictions", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "empty_test";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(1.0); };

    CPhysicsLoss loss("empty_test", net_in, net_out, {}, {eq});

    mlpdouble result = loss.Evaluate({}, {});

    REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);
    REQUIRE_EQUAL_TOL(loss.GetLastLossValue(), 0.0, 1e-12);
}

// ============================================================
// 7. PhysicsState accessor mapping (index-based, input-major)
// ============================================================

TEST_CASE("PhysicsState accessor mapping", "[CPhysicsLoss]")
{
    // Network: inputs {x, y, t}, outputs {u, v, p}
    // Equation uses subset: inputs {t, x, y}, outputs {p, u}
    std::vector<std::string> net_in  = {"x", "y", "t"};
    std::vector<std::string> net_out = {"u", "v", "p"};

    std::vector<mlpdouble> captured_in;
    std::vector<mlpdouble> captured_out;
    std::vector<mlpdouble> captured_jac;
    std::vector<mlpdouble> captured_hess;

    CPhysicsEquation eq;
    eq.name = "accessor_test";
    eq.input_names  = {"t", "x", "y"};   // eq_in: t->net[2], x->net[0], y->net[1]
    eq.output_names = {"p", "u"};         // eq_out: p->net[2], u->net[0]
    eq.requires_jacobian = true;
    eq.requires_hessian  = true;
    eq.residual = [&](const PhysicsState& s, const PhysicsData&) {
        // EquationIn: local indices map to correct network values
        captured_in.push_back(s.EquationIn(0));   // t -> net_in[2] -> 30
        captured_in.push_back(s.EquationIn(1));   // x -> net_in[0] -> 10
        captured_in.push_back(s.EquationIn(2));   // y -> net_in[1] -> 20

        // EquationOut: local indices map to correct network values
        captured_out.push_back(s.EquationOut(0));  // p -> net_out[2] -> 300
        captured_out.push_back(s.EquationOut(1));  // u -> net_out[0] -> 100

        // EquationJac(eq_input, eq_output) -> input-major layout
        // dp/dy: eq_input=2(y), eq_output=0(p) -> Jac(net_in[1], net_out[2]) = jac[1][2]
        captured_jac.push_back(s.EquationJac(2, 0));
        // du/dt: eq_input=0(t), eq_output=1(u) -> Jac(net_in[2], net_out[0]) = jac[2][0]
        captured_jac.push_back(s.EquationJac(0, 1));

        // EquationHess(eq_input_i, eq_input_j, eq_output)
        // d²p/dydt: eq_in_i=2(y), eq_in_j=0(t), eq_out=0(p)
        // -> Hess(net_in[1], net_in[2], net_out[2]) = hess[1][2][2]
        captured_hess.push_back(s.EquationHess(2, 0, 0));

        return mlpdouble(0.0);
    };

    CPhysicsLoss loss("accessor_test", net_in, net_out, {}, {eq});

    // Network inputs: x=10, y=20, t=30
    // Network outputs: u=100, v=200, p=300
    mlpdouble** jac  = AllocateJacobian(3, 3);
    mlpdouble*** hess = AllocateHessian(3, 3);

    // Jacobian layout: jacobian[input][output]
    jac[1][2] = 3.0;   // dp/dy  (input=1=y, output=2=p)
    jac[2][0] = 5.0;   // du/dt  (input=2=t, output=0=u)

    // Hessian layout: hessian[input_i][input_j][output]
    hess[1][2][2] = 4.0;  // d²p/dydt (y=1, t=2, p=2)

    PredictionResult pred = MakePrediction({10.0, 20.0, 30.0}, {100.0, 200.0, 300.0}, jac, hess);

    loss.Evaluate({pred}, {});

    REQUIRE_EQUAL_TOL(captured_in[0], 30.0, 1e-12);  // t
    REQUIRE_EQUAL_TOL(captured_in[1], 10.0, 1e-12);  // x
    REQUIRE_EQUAL_TOL(captured_in[2], 20.0, 1e-12);  // y

    REQUIRE_EQUAL_TOL(captured_out[0], 300.0, 1e-12);  // p
    REQUIRE_EQUAL_TOL(captured_out[1], 100.0, 1e-12);  // u

    REQUIRE_EQUAL_TOL(captured_jac[0], 3.0, 1e-12);   // dp/dy
    REQUIRE_EQUAL_TOL(captured_jac[1], 5.0, 1e-12);   // du/dt

    REQUIRE_EQUAL_TOL(captured_hess[0], 4.0, 1e-12);  // d²p/dydt

    FreeJacobian(jac, 3);
    FreeHessian(hess, 3);
}

// ============================================================
// 8. Per-point physics/source terms
// ============================================================

TEST_CASE("CPhysicsLoss passes per-point physics data", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x", "t"};
    std::vector<std::string> net_out = {"u"};

    std::vector<mlpdouble> captured_source;

    CPhysicsEquation eq;
    eq.name = "rhs";
    eq.input_names  = {"x", "t"};
    eq.output_names = {"u"};
    eq.requires_jacobian = true;
    eq.residual = [&](const PhysicsState& s, const PhysicsData& data) {
        captured_source.push_back(data.Ref("source"));
        // du/dt - source = 0
        // EquationJac: eq_input[1]=t(net_in[1]), eq_output[0]=u(net_out[0])
        mlpdouble residual = s.EquationJac(1, 0) - data.Ref("source");
        return residual;
    };

    CPhysicsLoss loss("rhs_test", net_in, net_out, {"source"}, {eq});

    mlpdouble** jac1 = AllocateJacobian(2, 1);
    jac1[1][0] = 10.0;  // du/dt (input=1=t, output=0=u)
    PredictionResult pred1 = MakePrediction({1.0, 0.0}, {2.0}, jac1);

    mlpdouble** jac2 = AllocateJacobian(2, 1);
    jac2[1][0] = 20.0;  // du/dt
    PredictionResult pred2 = MakePrediction({2.0, 1.0}, {3.0}, jac2);

    std::vector<std::vector<mlpdouble>> physics_data = {{10.0}, {20.0}};

    mlpdouble result = loss.Evaluate({pred1, pred2}, physics_data);

    REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);
    REQUIRE(captured_source.size() == 2);
    REQUIRE_EQUAL_TOL(captured_source[0], 10.0, 1e-12);
    REQUIRE_EQUAL_TOL(captured_source[1], 20.0, 1e-12);

    FreeJacobian(jac1, 2);
    FreeJacobian(jac2, 2);
}

// ============================================================
// 9. Physics data row count mismatch
// ============================================================

TEST_CASE("CPhysicsLoss rejects physics data row count mismatch", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "rows";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    CPhysicsLoss loss("physics_rows", net_in, net_out, {"source"}, {eq});

    PredictionResult p1 = MakePrediction({0.0}, {0.0});
    PredictionResult p2 = MakePrediction({1.0}, {1.0});

    // 1 row for 2 predictions
    REQUIRE_THROWS_AS(
        loss.Evaluate({p1, p2}, {{0.0}}),
        std::invalid_argument);
}

// ============================================================
// 10. Physics data row size mismatch
// ============================================================

TEST_CASE("CPhysicsLoss rejects physics data row size mismatch", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "size";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    CPhysicsLoss loss("physics_size", net_in, net_out, {"fx", "fy"}, {eq});

    PredictionResult pred = MakePrediction({1.0}, {1.0});

    // 1 value per row, but loss expects 2
    REQUIRE_THROWS_AS(
        loss.Evaluate({pred}, {{1.0}}),
        std::invalid_argument);
}

// ============================================================
// 11. Missing Jacobian throws
// ============================================================

TEST_CASE("CPhysicsLoss Jacobian nullptr throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "jac_null";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.requires_jacobian = true;
    eq.residual = [](const PhysicsState& s, const PhysicsData&) {
        return s.EquationJac(0, 0);
    };

    CPhysicsLoss loss("jac_null", net_in, net_out, {}, {eq});

    PredictionResult pred = MakePrediction({1.0}, {1.0}, nullptr, nullptr);

    REQUIRE_THROWS_AS(
        loss.Evaluate({pred}, {}),
        std::invalid_argument);
}

// ============================================================
// 12. Missing Hessian throws
// ============================================================

TEST_CASE("CPhysicsLoss Hessian nullptr throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "hess_null";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.requires_jacobian = true;
    eq.requires_hessian  = true;
    eq.residual = [](const PhysicsState& s, const PhysicsData&) {
        return s.EquationHess(0, 0, 0);
    };

    CPhysicsLoss loss("hess_null", net_in, net_out, {}, {eq});

    mlpdouble** jac = AllocateJacobian(1, 1);
    PredictionResult pred = MakePrediction({1.0}, {1.0}, jac, nullptr);

    REQUIRE_THROWS_AS(
        loss.Evaluate({pred}, {}),
        std::invalid_argument);

    FreeJacobian(jac, 1);
}

// ============================================================
// 13. Multiple physics source terms (separate equations)
// ============================================================

TEST_CASE("CPhysicsLoss multiple source terms via separate equations", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x", "y", "t"};
    std::vector<std::string> net_out = {"u", "v"};

    CPhysicsEquation eq1;
    eq1.name = "u_rhs";
    eq1.input_names  = {"x", "y", "t"};
    eq1.output_names = {"u"};
    eq1.requires_jacobian = true;
    eq1.residual = [](const PhysicsState& s, const PhysicsData& data) {
        // du/dt - fx = 0
        // eq_input[2]=t, eq_output[0]=u
        return s.EquationJac(2, 0) - data.Ref("fx");
    };

    CPhysicsEquation eq2;
    eq2.name = "v_rhs";
    eq2.input_names  = {"x", "y", "t"};
    eq2.output_names = {"v"};
    eq2.requires_jacobian = true;
    eq2.residual = [](const PhysicsState& s, const PhysicsData& data) {
        // dv/dt - fy = 0
        // eq_input[2]=t, eq_output[0]=v
        return s.EquationJac(2, 0) - data.Ref("fy");
    };

    CPhysicsLoss loss("multiple_rhs", net_in, net_out, {"fx", "fy"}, {eq1, eq2});

    mlpdouble** jac = AllocateJacobian(3, 2);
    jac[2][0] = 5.0;  // du/dt (input=2=t, output=0=u)
    jac[2][1] = 7.0;  // dv/dt (input=2=t, output=1=v)

    PredictionResult pred = MakePrediction({1.0, 2.0, 3.0}, {10.0, 20.0}, jac);

    std::vector<std::vector<mlpdouble>> physics_data = {{5.0, 7.0}};

    mlpdouble result = loss.Evaluate({pred}, physics_data);
    REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);

    FreeJacobian(jac, 3);
}

// ============================================================
// 14. PhysicsData accessor mapping
// ============================================================

TEST_CASE("PhysicsData accessor mapping", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    std::vector<mlpdouble> captured;

    CPhysicsEquation eq;
    eq.name = "data_accessor";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [&](const PhysicsState&, const PhysicsData& data) {
        captured.push_back(data.Ref("rho"));
        captured.push_back(data.Ref("mu"));
        return mlpdouble(0.0);
    };

    CPhysicsLoss loss("physics_data_accessor", net_in, net_out, {"rho", "mu"}, {eq});

    PredictionResult pred = MakePrediction({1.0}, {2.0});

    loss.Evaluate({pred}, {{1000.0, 0.001}});

    REQUIRE(captured.size() == 2);
    REQUIRE_EQUAL_TOL(captured[0], 1000.0, 1e-12);
    REQUIRE_EQUAL_TOL(captured[1], 0.001, 1e-12);
}

// ============================================================
// 15. Missing physics variable throws (out_of_range)
// ============================================================

TEST_CASE("PhysicsData missing variable throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "missing_var";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData& data) {
        return data.Ref("does_not_exist");
    };

    CPhysicsLoss loss("missing_physics", net_in, net_out, {}, {eq});

    PredictionResult pred = MakePrediction({1.0}, {1.0});

    REQUIRE_THROWS_AS(
        loss.Evaluate({pred}, {}),
        std::out_of_range);
}

// ============================================================
// 16. PhysicsData Has()
// ============================================================

TEST_CASE("PhysicsData Has detects configured variables", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "has_test";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData& data) {
        REQUIRE(data.Has("fx"));
        REQUIRE(data.Has("fy"));
        REQUIRE_FALSE(data.Has("fz"));
        return mlpdouble(0.0);
    };

    CPhysicsLoss loss("has_test", net_in, net_out, {"fx", "fy"}, {eq});

    PredictionResult pred = MakePrediction({1.0}, {1.0});
    loss.Evaluate({pred}, {{1.0, 2.0}});
}

// ============================================================
// 17. Index-based PhysicsState access
// ============================================================

TEST_CASE("PhysicsState index-based access", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x", "y"};
    std::vector<std::string> net_out = {"u", "v"};

    CPhysicsEquation eq;
    eq.name = "index_access";
    eq.input_names  = {"x", "y"};
    eq.output_names = {"u", "v"};
    eq.residual = [](const PhysicsState& s, const PhysicsData&) {
        REQUIRE_EQUAL_TOL(s.In(0), 10.0, 1e-12);
        REQUIRE_EQUAL_TOL(s.In(1), 20.0, 1e-12);
        REQUIRE_EQUAL_TOL(s.Out(0), 100.0, 1e-12);
        REQUIRE_EQUAL_TOL(s.Out(1), 200.0, 1e-12);
        return mlpdouble(0.0);
    };

    CPhysicsLoss loss("index_access", net_in, net_out, {}, {eq});

    PredictionResult pred = MakePrediction({10.0, 20.0}, {100.0, 200.0});
    loss.Evaluate({pred}, {});
}

// ============================================================
// 18. Bounds checking
// ============================================================

TEST_CASE("PhysicsState bounds checking", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "bounds";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState& s, const PhysicsData&) {
        REQUIRE_THROWS_AS(s.In(1),  std::out_of_range);
        REQUIRE_THROWS_AS(s.Out(1), std::out_of_range);
        return mlpdouble(0.0);
    };

    CPhysicsLoss loss("bounds_test", net_in, net_out, {}, {eq});

    PredictionResult pred = MakePrediction({1.0}, {2.0});
    loss.Evaluate({pred}, {});
}

// ============================================================
// 19. EvaluateOne returns raw unnormalized loss
// ============================================================

TEST_CASE("CPhysicsLoss EvaluateOne returns raw loss", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq1;
    eq1.name = "r1";
    eq1.input_names  = {"x"};
    eq1.output_names = {"u"};
    eq1.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(3.0); };

    CPhysicsEquation eq2;
    eq2.name = "r2";
    eq2.input_names  = {"x"};
    eq2.output_names = {"u"};
    eq2.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(4.0); };

    CPhysicsLoss loss("raw_test", net_in, net_out, {}, {eq1, eq2});

    PredictionResult pred = MakePrediction({1.0}, {1.0});

    // EvaluateOne returns sum weight * r^2 = 1*9 + 1*16 = 25
    mlpdouble raw = loss.EvaluateOne(pred);
    REQUIRE_EQUAL_TOL(raw, 25.0, 1e-12);

    // Evaluate normalizes: 25 / (1 * 2) = 12.5
    mlpdouble normalized = loss.Evaluate({pred}, {});
    REQUIRE_EQUAL_TOL(normalized, 12.5, 1e-12);
}

// ============================================================
// 20. Per-equation weights
// ============================================================

TEST_CASE("CPhysicsLoss per-equation weights", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq1;
    eq1.name = "weighted";
    eq1.input_names  = {"x"};
    eq1.output_names = {"u"};
    eq1.weight = 2.0;  // weight * r^2 = 2 * 3^2 = 18
    eq1.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(3.0); };

    CPhysicsEquation eq2;
    eq2.name = "unweighted";
    eq2.input_names  = {"x"};
    eq2.output_names = {"u"};
    eq2.weight = 1.0;  // weight * r^2 = 1 * 4^2 = 16
    eq2.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(4.0); };

    CPhysicsLoss loss("weight_test", net_in, net_out, {}, {eq1, eq2});

    PredictionResult pred = MakePrediction({1.0}, {1.0});

    // EvaluateOne: 2*9 + 1*16 = 18 + 16 = 34
    mlpdouble raw = loss.EvaluateOne(pred);
    REQUIRE_EQUAL_TOL(raw, 34.0, 1e-12);

    // Evaluate: 34 / (1 * 2) = 17.0
    mlpdouble normalized = loss.Evaluate({pred}, {});
    REQUIRE_EQUAL_TOL(normalized, 17.0, 1e-12);
}

// ============================================================
// 21. Negative weight throws
// ============================================================

TEST_CASE("CPhysicsLoss negative weight throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "neg_weight";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.weight = -1.0;
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    REQUIRE_THROWS_AS(
        CPhysicsLoss("neg", net_in, net_out, {}, {eq}),
        std::invalid_argument);
}

// ============================================================
// 22. Duplicate network input names throws
// ============================================================

TEST_CASE("CPhysicsLoss duplicate network input names throws", "[CPhysicsLoss]")
{
    // CPhysicsLoss validates network name uniqueness at construction
    std::vector<std::string> dup_in  = {"x", "x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "dup_net";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    REQUIRE_THROWS_AS(
        CPhysicsLoss("dup_net_in", dup_in, net_out, {}, {eq}),
        std::invalid_argument);
}

// ============================================================
// 23. Duplicate network output names throws
// ============================================================

TEST_CASE("CPhysicsLoss duplicate network output names throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> dup_out = {"u", "u"};

    CPhysicsEquation eq;
    eq.name = "dup_net";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    REQUIRE_THROWS_AS(
        CPhysicsLoss("dup_net_out", net_in, dup_out, {}, {eq}),
        std::invalid_argument);
}

// ============================================================
// 24. Empty equation name throws
// ============================================================

TEST_CASE("CPhysicsLoss empty equation name throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "";    // empty
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    REQUIRE_THROWS_AS(
        CPhysicsLoss("empty_eq_name", net_in, net_out, {}, {eq}),
        std::invalid_argument);
}

// ============================================================
// 25. Duplicate equation input/output names throws
// ============================================================

TEST_CASE("CPhysicsEquation duplicate input names throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x", "y"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "dup_eq_in";
    eq.input_names  = {"x", "x"};   // duplicate within equation
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    REQUIRE_THROWS_AS(
        CPhysicsLoss("dup_eq_in", net_in, net_out, {}, {eq}),
        std::invalid_argument);
}

TEST_CASE("CPhysicsEquation duplicate output names throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u", "v"};

    CPhysicsEquation eq;
    eq.name = "dup_eq_out";
    eq.input_names  = {"x"};
    eq.output_names = {"u", "u"};   // duplicate within equation
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    REQUIRE_THROWS_AS(
        CPhysicsLoss("dup_eq_out", net_in, net_out, {}, {eq}),
        std::invalid_argument);
}

// ============================================================
// 26. Multiple equations with different input/output subsets
// ============================================================

TEST_CASE("CPhysicsLoss equations with different subsets", "[CPhysicsLoss]")
{
    // Network: {x, y, t} -> {u, v}
    // Eq1 uses {x, t} -> {u}   (du/dt = 0)
    // Eq2 uses {y, t} -> {v}   (dv/dt = 0)
    std::vector<std::string> net_in  = {"x", "y", "t"};
    std::vector<std::string> net_out = {"u", "v"};

    CPhysicsEquation eq1;
    eq1.name = "u_eq";
    eq1.input_names  = {"x", "t"};    // eq_in: x->0, t->2
    eq1.output_names = {"u"};          // eq_out: u->0
    eq1.requires_jacobian = true;
    eq1.residual = [](const PhysicsState& s, const PhysicsData&) {
        // EquationJac(eq_input_idx, eq_output_idx)
        // eq_in[1]=t(net[2]), eq_out[0]=u(net[0])
        return s.EquationJac(1, 0) - mlpdouble(5.0);
    };

    CPhysicsEquation eq2;
    eq2.name = "v_eq";
    eq2.input_names  = {"y", "t"};    // eq_in: y->1, t->2
    eq2.output_names = {"v"};          // eq_out: v->1
    eq2.requires_jacobian = true;
    eq2.residual = [](const PhysicsState& s, const PhysicsData&) {
        // eq_in[1]=t(net[2]), eq_out[0]=v(net[1])
        return s.EquationJac(1, 0) - mlpdouble(7.0);
    };

    CPhysicsLoss loss("subset_test", net_in, net_out, {}, {eq1, eq2});

    mlpdouble** jac = AllocateJacobian(3, 2);
    jac[2][0] = 5.0;   // du/dt (input=2=t, output=0=u)
    jac[2][1] = 7.0;   // dv/dt (input=2=t, output=1=v)

    PredictionResult pred = MakePrediction({1.0, 2.0, 3.0}, {10.0, 20.0}, jac);

    mlpdouble result = loss.Evaluate({pred}, {});
    REQUIRE_EQUAL_TOL(result, 0.0, 1e-12);

    FreeJacobian(jac, 3);
}

// ============================================================
// 27. Empty physics variable name throws
// ============================================================

TEST_CASE("CPhysicsLoss empty physics variable name throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "empty_phys";
    eq.input_names  = {"x"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    REQUIRE_THROWS_AS(
        CPhysicsLoss("empty_phys", net_in, net_out, {"", "fx"}, {eq}),
        std::invalid_argument);
}

// ============================================================
// 28. PredictionResult dimension mismatch throws
// ============================================================

TEST_CASE("CPhysicsLoss PredictionResult dimension mismatch throws", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x", "y"};
    std::vector<std::string> net_out = {"u"};

    CPhysicsEquation eq;
    eq.name = "dim";
    eq.input_names  = {"x", "y"};
    eq.output_names = {"u"};
    eq.residual = [](const PhysicsState&, const PhysicsData&) { return mlpdouble(0.0); };

    CPhysicsLoss loss("dim_test", net_in, net_out, {}, {eq});

    // Wrong input dimension
    PredictionResult bad_in = MakePrediction({1.0}, {1.0});
    REQUIRE_THROWS_AS(
        loss.EvaluateOne(bad_in),
        std::invalid_argument);

    // Wrong output dimension
    PredictionResult bad_out = MakePrediction({1.0, 2.0}, {1.0, 2.0});
    REQUIRE_THROWS_AS(
        loss.EvaluateOne(bad_out),
        std::invalid_argument);
}

// ============================================================
// 29. Jac/Hess runtime null-pointer throws (undeclared usage)
// ============================================================

TEST_CASE("Jac/Hess throws when pointer is null regardless of flags", "[CPhysicsLoss]")
{
    std::vector<std::string> net_in  = {"x"};
    std::vector<std::string> net_out = {"u"};

    // Equation declares requires_jacobian=false but residual
    // still calls EquationJac.  This is a user contract violation,
    // but the runtime null-pointer check catches it.
    CPhysicsEquation eq_bad;
    eq_bad.name = "undeclared_jac";
    eq_bad.input_names  = {"x"};
    eq_bad.output_names = {"u"};
    eq_bad.requires_jacobian = false;  // WRONG — residual uses Jac
    eq_bad.residual = [](const PhysicsState& s, const PhysicsData&) {
        return s.EquationJac(0, 0);  // calls Jac despite flag=false
    };

    CPhysicsLoss loss("undeclared", net_in, net_out, {}, {eq_bad});

    // No Jacobian provided (flag is false, so trainer wouldn't allocate one)
    PredictionResult pred = MakePrediction({1.0}, {1.0});

    REQUIRE_THROWS_AS(
        loss.EvaluateOne(pred),
        std::runtime_error);  // null pointer in Jac()
}