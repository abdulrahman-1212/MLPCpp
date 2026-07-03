#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#include "CBaseLoss.hpp"
#include "CPhysicsLoss.hpp"
#include "variable_def.hpp"

#define REQUIRE_EQUAL_TOL(a, b, tol) \
    REQUIRE(static_cast<double>(a) == Approx(static_cast<double>(b)).margin(tol))

// Helper to construct PredictionResult with inputs for testing
static MLPToolbox::PredictionResult make_pred(
    std::vector<mlpdouble> in,
    std::vector<mlpdouble> out,
    std::vector<std::vector<mlpdouble>> jac,
    std::vector<std::vector<std::vector<mlpdouble>>> hes)
{
    MLPToolbox::PredictionResult p;
    p.inputs = std::move(in);
    p.outputs = std::move(out);
    p.jacobians = std::move(jac);
    p.hessians = std::move(hes);
    return p;
}


TEST_CASE("CPhysicsLoss basic evaluation", "[CPhysicsLoss]") {
    // Single equation: u_x + v_y
    auto res_fn = [](const MLPToolbox::PhysicsState& s) -> mlpdouble {
        return s.Jac(0, 0) + s.Jac(1, 1); // u_x + v_y
    };

    MLPToolbox::CPhysicsLoss loss("continuity", res_fn,
        {"x", "y"}, {"u", "v"},
        {"x", "y"}, {"u", "v", "p"});

    auto pred = make_pred(
        {0.0, 0.0},               // inputs
        {0.0, 0.0, 0.0},          // outputs (u, v, p)
        {{1.0, 2.0}, {3.0, 4.0}, {0.0, 0.0}}, // jacobians
        {{{0,0},{0,0}}, {{0,0},{0,0}}, {{0,0},{0,0}}} // hessians
    );

    // u_x=1, v_y=4 -> res=5, MSE = 25/1 = 25
    mlpdouble val = loss.Evaluate({pred}, {});
    REQUIRE_EQUAL_TOL(val, 25.0, 1e-12);
}

TEST_CASE("CPhysicsLoss empty predictions", "[CPhysicsLoss]") {
    auto res_fn = [](const MLPToolbox::PhysicsState&) -> mlpdouble { return 0.0; };
    MLPToolbox::CPhysicsLoss loss("empty", res_fn,
        {"x"}, {"u"}, {"x"}, {"u"});

    std::vector<MLPToolbox::PredictionResult> preds;
    mlpdouble val = loss.Evaluate(preds, {});
    REQUIRE_EQUAL_TOL(val, 0.0, 1e-12);
}

TEST_CASE("CPhysicsLoss missing input name throws at construction", "[CPhysicsLoss]") {
    auto res_fn = [](const MLPToolbox::PhysicsState&) -> mlpdouble { return 0.0; };
    REQUIRE_THROWS_AS(
        MLPToolbox::CPhysicsLoss("bad", res_fn,
            {"z"}, {"u"},   // "z" not in net inputs
            {"x", "y"}, {"u", "v"}),
        std::runtime_error);
}

TEST_CASE("CPhysicsLoss missing output name throws at construction", "[CPhysicsLoss]") {
    auto res_fn = [](const MLPToolbox::PhysicsState&) -> mlpdouble { return 0.0; };
    REQUIRE_THROWS_AS(
        MLPToolbox::CPhysicsLoss("bad", res_fn,
            {"x"}, {"w"},   // "w" not in net outputs
            {"x", "y"}, {"u", "v"}),
        std::runtime_error);
}

TEST_CASE("CPhysicsLoss ignores ref_data", "[CPhysicsLoss]") {
    auto res_fn = [](const MLPToolbox::PhysicsState& s) -> mlpdouble {
        return s.Out(0); // u
    };
    MLPToolbox::CPhysicsLoss loss("ignore_ref", res_fn,
        {"x"}, {"u"}, {"x"}, {"u"});

    auto pred = make_pred({0.0}, {2.0}, {{0.0}}, {{{0.0}}});

    // Pass garbage ref_data; must be ignored
    std::vector<std::vector<mlpdouble>> garbage = {{999.0}};
    mlpdouble val = loss.Evaluate({pred}, garbage);
    REQUIRE_EQUAL_TOL(val, 4.0, 1e-12); // 2^2 / 1
}

TEST_CASE("CPhysicsLoss uses pre-resolved indices correctly", "[CPhysicsLoss]") {
    // Equation uses only "v" and "y" from a larger network
    // net outputs: {u, v, p}, net inputs: {x, y}
    // eq maps: out{"v"}->net idx 1, in{"y"}->net idx 1
    auto res_fn = [](const MLPToolbox::PhysicsState& s) -> mlpdouble {
        return s.Jac(0, 0); // local idx 0 = v_y in net coords
    };

    MLPToolbox::CPhysicsLoss loss("v_y_only", res_fn,
        {"y"}, {"v"},              // equation-local names
        {"x", "y"}, {"u", "v", "p"} // full network names
    );

    // jacobians[1][1] = dv/dy = 7.0
    auto pred = make_pred(
        {0.0, 0.0},
        {0.0, 0.0, 0.0},
        {{1.0, 2.0}, {3.0, 7.0}, {5.0, 6.0}},
        {{{0,0},{0,0}}, {{0,0},{0,0}}, {{0,0},{0,0}}}
    );

    // res = 7.0, MSE = 49/1
    mlpdouble val = loss.Evaluate({pred}, {});
    REQUIRE_EQUAL_TOL(val, 49.0, 1e-12);
}

TEST_CASE("CPhysicsLoss Hessian access", "[CPhysicsLoss]") {
    // Single equation: u_xx
    auto res_fn = [](const MLPToolbox::PhysicsState& s) -> mlpdouble {
        return s.Hess(0, 0, 0); // u_xx
    };

    MLPToolbox::CPhysicsLoss loss("laplacian", res_fn,
        {"x"}, {"u"}, {"x", "y"}, {"u", "v"});

    // hessians[0][0][0] = d²u/dx² = 3.0
    auto pred = make_pred(
        {0.0, 0.0},
        {0.0, 0.0},
        {{0.0, 0.0}, {0.0, 0.0}},
        {{{3.0, 1.0}, {1.0, 5.0}}, {{0,0},{0,0}}}
    );

    mlpdouble val = loss.Evaluate({pred}, {});
    REQUIRE_EQUAL_TOL(val, 9.0, 1e-12); // 3^2 / 1
}

TEST_CASE("CPhysicsLoss multiple points averaging", "[CPhysicsLoss]") {
    auto res_fn = [](const MLPToolbox::PhysicsState& s) -> mlpdouble {
        return s.Out(0); // u
    };
    MLPToolbox::CPhysicsLoss loss("multi", res_fn,
        {"x"}, {"u"}, {"x"}, {"u"});

    auto p1 = make_pred({0.0}, {2.0}, {{0.0}}, {{{0.0}}});
    auto p2 = make_pred({1.0}, {4.0}, {{0.0}}, {{{0.0}}});

    // res: 2, 4 -> sq: 4, 16 -> sum: 20, MSE = 20/2 = 10
    mlpdouble val = loss.Evaluate({p1, p2}, {});
    REQUIRE_EQUAL_TOL(val, 10.0, 1e-12);
}

TEST_CASE("CPhysicsLoss In() accesses pred.inputs", "[CPhysicsLoss]") {
    auto res_fn = [](const MLPToolbox::PhysicsState& s) -> mlpdouble {
        return s.In(0); // x coordinate
    };
    MLPToolbox::CPhysicsLoss loss("input_access", res_fn,
        {"x"}, {"u"}, {"x"}, {"u"});

    auto pred = make_pred({3.0}, {0.0}, {{0.0}}, {{{0.0}}});

    // res = 3.0, MSE = 9/1
    mlpdouble val = loss.Evaluate({pred}, {});
    REQUIRE_EQUAL_TOL(val, 9.0, 1e-12);
}