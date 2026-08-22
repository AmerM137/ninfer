#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"
#include "targets/qwen3_6_35b_a3b/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

LoadPlan::LoadPlan(std::unique_ptr<ArtifactLoadPlan> artifact) noexcept
    : artifact(std::move(artifact)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (artifact == nullptr) { throw std::logic_error("target load plan is empty"); }
    return artifact->materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<LoadedModelData> data) noexcept : data(std::move(data)) {}

LoadedModel::LoadedModel(LoadedModel&&) noexcept = default;

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail

namespace ninfer::targets::qwen3_6_35b_a3b {
namespace {

constexpr ModelSamplingDefaults kQwen3_6_35BA3BDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kQwen3_6_35BA3BDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::GroupwiseInt;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    return LoadPlan(std::make_unique<detail::ArtifactLoadPlan>(
        detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(options))));
}

Package::LoadedModel
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.artifact == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto artifact = std::move(plan.artifact);
    auto data     = std::make_unique<detail::LoadedModelData>(
        artifact->weights_profile, std::move(artifact->bindings), std::move(materialized));
    return LoadedModel(std::move(data));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.data == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(model.data->frontend,
                                  qwen3_6::FrontendOptions{
                                      .vision_enabled    = model.data->runtime.features.vision,
                                      .max_context       = options.max_context,
                                      .media_cache_bytes = options.media_cache_bytes,
                                      .media_live_bytes  = options.media_live_bytes,
                                      .media_preprocess_threads = options.media_preprocess_threads,
                                  });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.data == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::create_program<detail::Variant>(
        model.data->runtime, model.data->weights_profile, std::move(plan), device);
}

} // namespace ninfer::targets::qwen3_6_35b_a3b
