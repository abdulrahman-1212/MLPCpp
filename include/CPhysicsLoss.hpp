#pragma once
#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "CBaseLoss.hpp"
#include "variable_def.hpp"

namespace MLPToolbox {

// Forward declarations
class PhysicsData;
class PhysicsState;

// Now safe to define
using ResidualFunction = std::function<mlpdouble(const PhysicsState&, const PhysicsData&)>;

// ============================================================================
// PhysicsData
// ============================================================================
class PhysicsData {
public:
    PhysicsData(std::vector<mlpdouble> values, std::vector<std::string> names)
        : values_(std::move(values)), names_(std::move(names)) {
        if (names_.size() != values_.size()) {
            throw std::invalid_argument(
                "PhysicsData: names and values must have identical sizes. " +
                std::string("names=") + std::to_string(names_.size()) +
                ", values=" + std::to_string(values_.size()));
        }
        for (std::size_t i = 0; i < names_.size(); ++i) {
            if (names_[i].empty()) {
                throw std::invalid_argument("PhysicsData: physics variable name cannot be empty.");
            }
            if (name_to_index_.find(names_[i]) != name_to_index_.end()) {
                throw std::invalid_argument("PhysicsData: duplicate variable name '" + names_[i] + "'.");
            }
            name_to_index_.emplace(names_[i], i);
        }
    }

    bool Empty() const noexcept { return values_.empty(); }
    std::size_t Size() const noexcept { return values_.size(); }
    bool Has(const std::string& name) const noexcept { return name_to_index_.find(name) != name_to_index_.end(); }

    mlpdouble At(std::size_t index) const {
        if (index >= values_.size()) {
            throw std::out_of_range("PhysicsData: index " + std::to_string(index) + " is out of range.");
        }
        return values_[index];
    }

    mlpdouble At(const std::string& name) const {
        const auto it = name_to_index_.find(name);
        if (it == name_to_index_.end()) {
            throw std::out_of_range("PhysicsData: variable '" + name + "' not found.");
        }
        return values_[it->second];
    }

    mlpdouble Ref(std::size_t index) const { return At(index); }
    mlpdouble Ref(const std::string& name) const { return At(name); }

    const std::vector<std::string>& Names() const noexcept { return names_; }
    const std::vector<mlpdouble>& Values() const noexcept { return values_; }

private:
    std::vector<mlpdouble> values_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, std::size_t> name_to_index_;
};

// ============================================================================
// PhysicsState
// ============================================================================
class PhysicsState {
public:
    PhysicsState(const PredictionResult& pred,
                 const std::vector<std::size_t>& equation_input_indices,
                 const std::vector<std::size_t>& equation_output_indices,
                 std::size_t n_network_inputs,
                 std::size_t n_network_outputs)
        : pred_(pred), equation_input_indices_(equation_input_indices),
          equation_output_indices_(equation_output_indices),
          n_network_inputs_(n_network_inputs),
          n_network_outputs_(n_network_outputs) {
        if (pred_.inputs.size() != n_network_inputs_) {
            throw std::invalid_argument(
                "PhysicsState: PredictionResult input dimension mismatch. " +
                std::string("expected=") + std::to_string(n_network_inputs_) +
                ", got=" + std::to_string(pred_.inputs.size()));
        }
        if (pred_.outputs.size() != n_network_outputs_) {
            throw std::invalid_argument(
                "PhysicsState: PredictionResult output dimension mismatch. " +
                std::string("expected=") + std::to_string(n_network_outputs_) +
                ", got=" + std::to_string(pred_.outputs.size()));
        }
    }

    mlpdouble In(std::size_t network_input_index) const {
        if (network_input_index >= pred_.inputs.size()) {
            throw std::out_of_range("PhysicsState::In: input index out of range.");
        }
        return pred_.inputs[network_input_index];
    }

    mlpdouble Out(std::size_t network_output_index) const {
        if (network_output_index >= pred_.outputs.size()) {
            throw std::out_of_range("PhysicsState::Out: output index out of range.");
        }
        return pred_.outputs[network_output_index];
    }

    mlpdouble Jac(std::size_t input_index, std::size_t output_index) const {
        if (input_index >= pred_.inputs.size())  throw std::out_of_range("PhysicsState::Jac: input index out of range.");
        if (output_index >= pred_.outputs.size()) throw std::out_of_range("PhysicsState::Jac: output index out of range.");
        if (pred_.jacobian == nullptr)            throw std::runtime_error("PhysicsState::Jac: Jacobian was not provided by PredictionResult.");
        return pred_.jacobian[input_index][output_index];
    }

    mlpdouble Hess(std::size_t input_i, std::size_t input_j, std::size_t output_index) const {
        if (input_i >= pred_.inputs.size() || input_j >= pred_.inputs.size()) throw std::out_of_range("PhysicsState::Hess: input index out of range.");
        if (output_index >= pred_.outputs.size())                             throw std::out_of_range("PhysicsState::Hess: output index out of range.");
        if (pred_.hessian == nullptr)                                         throw std::runtime_error("PhysicsState::Hess: Hessian was not provided by PredictionResult.");
        return pred_.hessian[input_i][input_j][output_index];
    }

    mlpdouble EquationIn(std::size_t equation_input_index) const {
        if (equation_input_index >= equation_input_indices_.size()) throw std::out_of_range("PhysicsState::EquationIn: index out of range.");
        return In(equation_input_indices_[equation_input_index]);
    }

    mlpdouble EquationOut(std::size_t equation_output_index) const {
        if (equation_output_index >= equation_output_indices_.size()) throw std::out_of_range("PhysicsState::EquationOut: index out of range.");
        return Out(equation_output_indices_[equation_output_index]);
    }

    mlpdouble EquationJac(std::size_t equation_input_index, std::size_t equation_output_index) const {
        if (equation_input_index >= equation_input_indices_.size())  throw std::out_of_range("PhysicsState::EquationJac: input index out of range.");
        if (equation_output_index >= equation_output_indices_.size()) throw std::out_of_range("PhysicsState::EquationJac: output index out of range.");
        return Jac(equation_input_indices_[equation_input_index], equation_output_indices_[equation_output_index]);
    }

    mlpdouble EquationHess(std::size_t equation_input_i, std::size_t equation_input_j, std::size_t equation_output_index) const {
        if (equation_input_i >= equation_input_indices_.size() || equation_input_j >= equation_input_indices_.size()) throw std::out_of_range("PhysicsState::EquationHess: input index out of range.");
        if (equation_output_index >= equation_output_indices_.size()) throw std::out_of_range("PhysicsState::EquationHess: output index out of range.");
        return Hess(equation_input_indices_[equation_input_i], equation_input_indices_[equation_input_j], equation_output_indices_[equation_output_index]);
    }

    std::size_t NumEquationInputs()  const noexcept { return equation_input_indices_.size(); }
    std::size_t NumEquationOutputs() const noexcept { return equation_output_indices_.size(); }
    std::size_t NumNetworkInputs()   const noexcept { return n_network_inputs_; }
    std::size_t NumNetworkOutputs()  const noexcept { return n_network_outputs_; }

private:
    const PredictionResult& pred_;
    const std::vector<std::size_t>& equation_input_indices_;
    const std::vector<std::size_t>& equation_output_indices_;
    std::size_t n_network_inputs_;
    std::size_t n_network_outputs_;
};

// ============================================================================
// CPhysicsEquation
// ============================================================================
struct CPhysicsEquation {
    std::string name;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    double weight{1.0};                  
    bool requires_jacobian{false};       
    bool requires_hessian{false};        
    ResidualFunction residual;
};

// ============================================================================
// CPhysicsLoss
// ============================================================================
class CPhysicsLoss : public CBaseLoss {
public:
    CPhysicsLoss(std::string name,
                 std::vector<std::string> network_input_names,
                 std::vector<std::string> network_output_names,
                 std::vector<std::string> physics_variable_names,
                 std::vector<CPhysicsEquation> equations)
        : CBaseLoss(name),
          network_input_names_(std::move(network_input_names)),
          network_output_names_(std::move(network_output_names)),
          physics_variable_names_(std::move(physics_variable_names)),
          equations_(std::move(equations)) {
        if (name_.empty()) throw std::invalid_argument("CPhysicsLoss: loss name cannot be empty.");
        if (equations_.empty()) throw std::invalid_argument("CPhysicsLoss: at least one equation is required.");

        ValidateNetworkNames();
        ValidatePhysicsVariableNames();
        ResolveEquationIndices();
        ValidateEquationCallbacks();
        ValidateEquationWeights();
        DetermineDerivativeRequirements();
    }

    std::size_t NumEquations() const noexcept { return equations_.size(); }
    std::size_t NumPhysicsVariables() const noexcept { return physics_variable_names_.size(); }
    bool RequiresJacobian() const noexcept { return requires_jacobian_; }
    bool RequiresHessian() const noexcept { return requires_hessian_; }

    const std::vector<std::string>& GetNetworkInputNames() const noexcept { return network_input_names_; }
    const std::vector<std::string>& GetNetworkOutputNames() const noexcept { return network_output_names_; }
    const std::vector<std::string>& GetPhysicsVariableNames() const noexcept { return physics_variable_names_; }

    const CPhysicsEquation& GetEquation(std::size_t index) const {
        if (index >= equations_.size()) throw std::out_of_range("CPhysicsLoss::GetEquation: index out of range.");
        return equations_[index];
    }

    mlpdouble EvaluateOne(const PredictionResult& pred,
                           const std::vector<mlpdouble>& physics_data = {}) const {
        ValidatePrediction(pred);
        if (physics_data.size() != physics_variable_names_.size()) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': physics data size mismatch. Expected " +
                std::to_string(physics_variable_names_.size()) + ", got " +
                std::to_string(physics_data.size()));
        }

        PhysicsData data(physics_data, physics_variable_names_);
        mlpdouble raw_loss = mlpdouble(0.0);
        for (std::size_t e = 0; e < equations_.size(); ++e) {
            PhysicsState state(pred,
                               equation_input_indices_[e],
                               equation_output_indices_[e],
                               network_input_names_.size(),
                               network_output_names_.size());
            const mlpdouble residual = equations_[e].residual(state, data);
            const double w = equations_[e].weight;
            raw_loss += mlpdouble(w) * residual * residual;
        }
        return raw_loss;
    }

    mlpdouble Evaluate(const std::vector<PredictionResult>& preds,
                       const std::vector<std::vector<mlpdouble>>& physics_data) override {
        const std::size_t N = preds.size();
        if (N == 0) {
            last_loss_value_ = 0.0;
            return mlpdouble(0.0);
        }
        if (!physics_variable_names_.empty() && physics_data.size() != N) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': physics_data must contain exactly one row per prediction.");
        }
        if (physics_variable_names_.empty() && !physics_data.empty() && physics_data.size() != N) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': supplied physics_data has wrong number of rows.");
        }

        mlpdouble raw_total = mlpdouble(0.0);
        for (std::size_t p = 0; p < N; ++p) {
            const std::vector<mlpdouble> empty_data;
            const auto& point_data = physics_data.empty() ? empty_data : physics_data[p];
            raw_total += EvaluateOne(preds[p], point_data);
        }

        const double denominator = static_cast<double>(N) * static_cast<double>(equations_.size());
        const mlpdouble normalized = raw_total / mlpdouble(denominator);
        last_loss_value_ = to_double(normalized);
        return normalized;
    }

private:
    void ValidateNetworkNames() {
        ValidateUniqueNames(network_input_names_, "network input");
        ValidateUniqueNames(network_output_names_, "network output");
    }

    void ValidateUniqueNames(const std::vector<std::string>& names, const std::string& context) const {
        std::unordered_set<std::string> seen;
        for (const auto& name : names) {
            if (name.empty()) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': " + context + " name cannot be empty.");
            }
            if (seen.find(name) != seen.end()) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': duplicate " + context + " name '" + name + "'.");
            }
            seen.insert(name);
        }
    }

    void ValidatePhysicsVariableNames() const {
        std::unordered_set<std::string> seen;
        for (const auto& name : physics_variable_names_) {
            if (name.empty()) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': physics variable name cannot be empty.");
            }
            if (seen.find(name) != seen.end()) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': duplicate physics variable name '" + name + "'.");
            }
            seen.insert(name);
        }
    }

    void ResolveEquationIndices() {
        equation_input_indices_.resize(equations_.size());
        equation_output_indices_.resize(equations_.size());

        for (std::size_t e = 0; e < equations_.size(); ++e) {
            const auto& eq = equations_[e];
            if (eq.name.empty()) {
                throw std::invalid_argument("CPhysicsLoss '" + name_ + "': equation name cannot be empty.");
            }

            ValidateUniqueNames(eq.input_names, "input for equation '" + eq.name + "'");
            ValidateUniqueNames(eq.output_names, "output for equation '" + eq.name + "'");

            std::vector<std::size_t> input_indices;
            input_indices.reserve(eq.input_names.size());
            for (const auto& input_name : eq.input_names) {
                auto it = std::find(network_input_names_.begin(), network_input_names_.end(), input_name);
                if (it == network_input_names_.end()) {
                    throw std::invalid_argument(
                        "CPhysicsLoss '" + name_ + "': equation '" + eq.name +
                        "' references unknown network input '" + input_name + "'.");
                }
                input_indices.push_back(static_cast<std::size_t>(std::distance(network_input_names_.begin(), it)));
            }

            std::vector<std::size_t> output_indices;
            output_indices.reserve(eq.output_names.size());
            for (const auto& output_name : eq.output_names) {
                auto it = std::find(network_output_names_.begin(), network_output_names_.end(), output_name);
                if (it == network_output_names_.end()) {
                    throw std::invalid_argument(
                        "CPhysicsLoss '" + name_ + "': equation '" + eq.name +
                        "' references unknown network output '" + output_name + "'.");
                }
                output_indices.push_back(static_cast<std::size_t>(std::distance(network_output_names_.begin(), it)));
            }

            equation_input_indices_[e] = std::move(input_indices);
            equation_output_indices_[e] = std::move(output_indices);
        }
    }

    void ValidateEquationCallbacks() const {
        for (const auto& eq : equations_) {
            if (!eq.residual) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': equation '" + eq.name + "' has no residual callback.");
            }
        }
    }

    void ValidateEquationWeights() const {
        for (const auto& eq : equations_) {
            if (eq.weight < 0.0) {
                throw std::invalid_argument(
                    "CPhysicsLoss '" + name_ + "': equation '" + eq.name +
                    "' has negative weight (" + std::to_string(eq.weight) + ").");
            }
        }
    }

    void DetermineDerivativeRequirements() {
        requires_jacobian_ = false;
        requires_hessian_ = false;
        for (const auto& eq : equations_) {
            requires_jacobian_ = requires_jacobian_ || eq.requires_jacobian;
            requires_hessian_ = requires_hessian_ || eq.requires_hessian;
        }
    }

    void ValidatePrediction(const PredictionResult& pred) const {
        if (pred.inputs.size() != network_input_names_.size()) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': PredictionResult input dimension mismatch. Expected " +
                std::to_string(network_input_names_.size()) + ", got " + std::to_string(pred.inputs.size()));
        }
        if (pred.outputs.size() != network_output_names_.size()) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': PredictionResult output dimension mismatch. Expected " +
                std::to_string(network_output_names_.size()) + ", got " + std::to_string(pred.outputs.size()));
        }
        if (requires_jacobian_ && pred.jacobian == nullptr) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': Jacobian is required but PredictionResult does not contain one.");
        }
        if (requires_hessian_ && pred.hessian == nullptr) {
            throw std::invalid_argument(
                "CPhysicsLoss '" + name_ + "': Hessian is required but PredictionResult does not contain one.");
        }
    }

    // --- Members ---
    // NOTE: name_ is inherited from CBaseLoss and does not need to be redeclared here.
    std::vector<std::string> network_input_names_;
    std::vector<std::string> network_output_names_;
    std::vector<std::string> physics_variable_names_;
    std::vector<CPhysicsEquation> equations_;
    std::vector<std::vector<std::size_t>> equation_input_indices_;
    std::vector<std::vector<std::size_t>> equation_output_indices_;
    bool requires_jacobian_{false};
    bool requires_hessian_{false};
};

} // namespace MLPToolbox