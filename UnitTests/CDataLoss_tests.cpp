#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#include "CBaseLoss.hpp"
#include "CDataLoss.hpp"
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

// ============================================================
//  CDataLoss Unit Tests
// ============================================================

TEST_CASE("CDataLoss empty predictions", "[CDataLoss]") {
    MLPToolbox::CDataLoss loss;
    std::vector<MLPToolbox::PredictionResult> preds;
    std::vector<std::vector<mlpdouble>> ref;

    mlpdouble val = loss.Evaluate(preds, ref);
    REQUIRE_EQUAL_TOL(val, 0.0, 1e-12);
}

TEST_CASE("CDataLoss size mismatch throws", "[CDataLoss]") {
    MLPToolbox::CDataLoss loss;
    std::vector<MLPToolbox::PredictionResult> preds(1);
    preds[0].outputs = {1.0};
    std::vector<std::vector<mlpdouble>> ref(2, {1.0});

    REQUIRE_THROWS_AS(loss.Evaluate(preds, ref), std::runtime_error);
}

TEST_CASE("CDataLoss inconsistent outputs throws", "[CDataLoss]") {
    MLPToolbox::CDataLoss loss;
    std::vector<MLPToolbox::PredictionResult> preds(2);
    preds[0].outputs = {1.0, 2.0};
    preds[1].outputs = {1.0};
    std::vector<std::vector<mlpdouble>> ref = {{1.0, 2.0}, {1.0, 2.0}};

    REQUIRE_THROWS_AS(loss.Evaluate(preds, ref), std::runtime_error);
}

TEST_CASE("CDataLoss ref_data size mismatch throws", "[CDataLoss]") {
    MLPToolbox::CDataLoss loss;
    std::vector<MLPToolbox::PredictionResult> preds(1);
    preds[0].outputs = {1.0, 2.0};
    std::vector<std::vector<mlpdouble>> ref = {{1.0}};

    REQUIRE_THROWS_AS(loss.Evaluate(preds, ref), std::runtime_error);
}

TEST_CASE("CDataLoss simple MSE calculation", "[CDataLoss]") {
    MLPToolbox::CDataLoss loss;
    std::vector<MLPToolbox::PredictionResult> preds(1);
    preds[0].outputs = {3.0, 4.0};
    std::vector<std::vector<mlpdouble>> ref = {{1.0, 2.0}};

    // diff = {2, 2}, sq = {4, 4}, sum = 8, MSE = 8 / (1*2) = 4
    mlpdouble val = loss.Evaluate(preds, ref);
    REQUIRE_EQUAL_TOL(val, 4.0, 1e-12);
}

TEST_CASE("CDataLoss multiple points MSE", "[CDataLoss]") {
    MLPToolbox::CDataLoss loss;
    std::vector<MLPToolbox::PredictionResult> preds(2);
    preds[0].outputs = {2.0};
    preds[1].outputs = {4.0};
    std::vector<std::vector<mlpdouble>> ref = {{1.0}, {2.0}};

    // diffs: 1, 2 -> sq: 1, 4 -> sum: 5, MSE = 5 / (2*1) = 2.5
    mlpdouble val = loss.Evaluate(preds, ref);
    REQUIRE_EQUAL_TOL(val, 2.5, 1e-12);
}

