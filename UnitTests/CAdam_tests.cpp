/*!
 * \file adam_optimizer_test.cpp
 * \brief Unit tests for the Adam optimizer.
 * \author Abdulrahman M. Saad
 */


#include "../include/CAdam.hpp"

using namespace MLPToolbox;

TEST_CASE("Adam Constructor and Hyperparameter Tests",
          "[Optimization][Adam]")
{
  SECTION("Default construction")
  {
    CAdam adam(100);

    CHECK(adam.GetLR()    == Approx(1e-3));
    CHECK(adam.GetBeta1() == Approx(0.9));
    CHECK(adam.GetBeta2() == Approx(0.999));
    CHECK(adam.GetEps()   == Approx(1e-8));

    CHECK(adam.GetStep() == 0);
  }

  SECTION("Custom hyperparameters")
  {
    CAdam adam(
      50,
      1e-2,
      0.8,
      0.95,
      1e-6);

    CHECK(adam.GetLR()    == Approx(1e-2));
    CHECK(adam.GetBeta1() == Approx(0.8));
    CHECK(adam.GetBeta2() == Approx(0.95));
    CHECK(adam.GetEps()   == Approx(1e-6));

    CHECK(adam.GetStep() == 0);
  }

  SECTION("Zero parameter count throws")
  {
    CHECK_THROWS_AS(
      CAdam(0),
      std::invalid_argument);
  }

  SECTION("Zero learning rate throws")
  {
    CHECK_THROWS_AS(
      CAdam(10, 0.0),
      std::invalid_argument);
  }

  SECTION("Negative learning rate throws")
  {
    CHECK_THROWS_AS(
      CAdam(10, -1e-3),
      std::invalid_argument);
  }

  SECTION("Invalid beta1 lower bound")
  {
    CHECK_THROWS_AS(
      CAdam(10, 1e-3, 0.0),
      std::invalid_argument);
  }

  SECTION("Invalid beta1 upper bound")
  {
    CHECK_THROWS_AS(
      CAdam(10, 1e-3, 1.0),
      std::invalid_argument);
  }

  SECTION("Invalid beta2 lower bound")
  {
    CHECK_THROWS_AS(
      CAdam(10, 1e-3, 0.9, 0.0),
      std::invalid_argument);
  }

  SECTION("Invalid beta2 upper bound")
  {
    CHECK_THROWS_AS(
      CAdam(10, 1e-3, 0.9, 1.0),
      std::invalid_argument);
  }

  SECTION("Zero epsilon throws")
  {
    CHECK_THROWS_AS(
      CAdam(10, 1e-3, 0.9, 0.999, 0.0),
      std::invalid_argument);
  }
}

TEST_CASE("Adam Setter Tests",
          "[Optimization][Adam]")
{
  CAdam adam(10);

  SECTION("Learning rate setter")
  {
    adam.SetLR(5e-4);

    CHECK(adam.GetLR() == Approx(5e-4));
  }

  SECTION("Multiple learning rate updates")
  {
    adam.SetLR(1e-2);
    CHECK(adam.GetLR() == Approx(1e-2));

    adam.SetLR(1e-4);
    CHECK(adam.GetLR() == Approx(1e-4));
  }
}

TEST_CASE("Adam Reset Test",
          "[Optimization][Adam]")
{
  CAdam adam(10);

  SECTION("Reset immediately after construction")
  {
    adam.Reset();

    CHECK(adam.GetStep() == 0);
  }
}