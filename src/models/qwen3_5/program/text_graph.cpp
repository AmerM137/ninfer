#include "models/qwen3_5/program/text_graph.h"
#include "models/common/moe/plan.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

std::size_t moe_graph_index(ops::ExpertSubsetKind kind) {
    switch (kind) {
    case ops::ExpertSubsetKind::All:
        return 0;
    case ops::ExpertSubsetKind::None:
        return 1;
    case ops::ExpertSubsetKind::MaskOnes:
        return 2;
    default:
        throw std::logic_error("invalid GPU subset for a MoE graph");
    }
}

void instantiate(DecodeGraphExecutable& executable, const DecodeGraphDefinition& definition) {
    if (definition.ready()) executable.instantiate(definition);
}

void update(DecodeGraphExecutable& executable, const DecodeGraphDefinition& definition) {
    if (executable.ready() != definition.ready())
        throw std::logic_error("text graph segment topology changed");
    if (definition.ready()) executable.update(definition);
}

void upload(DecodeGraphExecutable& executable, cudaStream_t stream) {
    if (executable.ready()) executable.upload(stream);
}
} // namespace

TextGraphDefinition::TextGraphDefinition(std::size_t count, const TextGraphDefinition* shared)
    : layers(count) {
    for (std::size_t i = 0; i < count; ++i)
        layers[i].moe = shared ? shared->layers.at(i).moe : std::make_shared<MoeGraphDefinitions>();
}

struct TextGraphExecutable::Target {
    struct Layer {
        DecodeGraphExecutable mixer, steady, feature;
        std::array<DecodeGraphExecutable, 3> moe;
    };

    std::vector<Layer> layers;
    const TextGraphDefinition* installed;

    explicit Target(const TextGraphDefinition& definition)
        : layers(definition.layers.size()), installed(&definition) {
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& source = definition.layers[i];
            auto& target       = layers[i];
            instantiate(target.mixer, source.mixer);
            instantiate(target.steady, source.steady);
            instantiate(target.feature, source.feature);
            for (std::size_t kind = 0; kind < target.moe.size(); ++kind)
                instantiate(target.moe[kind], source.moe->subsets[kind]);
        }
    }
};

TextGraphExecutable::TextGraphExecutable(const TextGraphDefinition& definition,
                                         const TextGraphExecutable* shared)
    : target_(shared ? shared->target_ : std::make_shared<Target>(definition)) {
    prefix_.instantiate(definition.prefix);
    suffix_.instantiate(definition.suffix);
}

void TextGraphExecutable::update(const TextGraphDefinition& definition) {
    prefix_.update(definition.prefix);
    suffix_.update(definition.suffix);
    activate(definition);
}

void TextGraphExecutable::activate(const TextGraphDefinition& definition) {
    if (target_->installed == &definition) return;
    if (target_->layers.size() != definition.layers.size())
        throw std::logic_error("text graph layer count changed");
    for (std::size_t i = 0; i < target_->layers.size(); ++i) {
        auto& target       = target_->layers[i];
        const auto& source = definition.layers[i];
        execution::update(target.mixer, source.mixer);
        execution::update(target.steady, source.steady);
        execution::update(target.feature, source.feature);
    }
    target_->installed = &definition;
}

void TextGraphExecutable::upload(cudaStream_t stream) {
    prefix_.upload(stream);
    suffix_.upload(stream);
    for (auto& layer : target_->layers) {
        execution::upload(layer.mixer, stream);
        execution::upload(layer.steady, stream);
        execution::upload(layer.feature, stream);
        for (auto& subset : layer.moe) execution::upload(subset, stream);
    }
}

TextGraphRun::TextGraphRun(TextGraphDefinition& definition, CudaExecutionView cuda)
    : definition_(&definition), cuda_(cuda) {}

TextGraphRun::TextGraphRun(TextGraphExecutable& executable, CudaExecutionView cuda,
                           const std::function<void()>* after_first_submit)
    : executable_(&executable), cuda_(cuda), after_first_submit_(after_first_submit) {}

void TextGraphRun::begin() {
    if (recording()) capture_.emplace(cuda_.stream());
}

void TextGraphRun::begin_layers() {
    if (traversed_) throw std::logic_error("text graph has more than one target traversal");
    traversed_ = true;
    if (recording()) {
        definition_->prefix = capture_->finish();
        capture_.reset();
    } else {
        executable_->prefix_.launch(cuda_.stream());
    }
}

void TextGraphRun::end_layers() {
    if (recording()) capture_.emplace(cuda_.stream());
}

void TextGraphRun::finish() {
    if (!traversed_) throw std::logic_error("text graph did not execute its target traversal");
    if (recording()) {
        definition_->suffix = capture_->finish();
        capture_.reset();
    } else {
        executable_->suffix_.launch(cuda_.stream());
    }
}

void TextGraphRun::first_submitted() {
    if (!notified_) {
        notified_ = true;
        if (after_first_submit_) (*after_first_submit_)();
    }
}

void TextGraphRun::capture_layer(std::size_t layer, moe::Resources& resources, ExpertBankId bank,
                                 std::int32_t tokens, const std::function<void()>& mixer,
                                 const std::function<void(const moe::PreparedExecution&)>& gpu_moe,
                                 const std::function<void()>& feature) {
    auto& definition   = definition_->layers.at(layer);
    const auto& plan   = resources.plan().layer(bank);
    const auto experts = std::uint32_t(plan.spec.profile.routed.experts);
    definition.mixer.capture(cuda_.stream(), mixer);
    using Kind = ops::ExpertSubsetKind;
    std::array<bool, 3> reachable{};
    if (plan.requires_full_bank(tokens)) {
        reachable[moe_graph_index(Kind::All)] = true;
    } else {
        reachable[moe_graph_index(Kind::All)]  = plan.cache_experts == experts;
        reachable[moe_graph_index(Kind::None)] = plan.fixed_experts == 0;
        reachable[moe_graph_index(Kind::MaskOnes)] =
            plan.cache_experts > 0 && plan.fixed_experts < experts;
    }
    for (auto kind : {Kind::All, Kind::None, Kind::MaskOnes}) {
        auto& part = definition.moe->subsets[moe_graph_index(kind)];
        if (!reachable[moe_graph_index(kind)] || part.ready()) continue;
        const auto call = resources.describe_layer(bank, tokens, kind);
        part.capture(cuda_.stream(), [&] { gpu_moe(call); });
    }
    if (!plan.requires_full_bank(tokens) || plan.stream_layout.empty()) {
        const auto kind = plan.cache_experts == 0         ? Kind::None
                          : plan.cache_experts == experts ? Kind::All
                                                          : Kind::MaskOnes;
        const auto call = resources.describe_layer(bank, tokens, kind);
        definition.steady.capture(cuda_.stream(), [&] {
            mixer();
            gpu_moe(call);
        });
    }
    if (feature) definition.feature.capture(cuda_.stream(), feature);
}

void TextGraphRun::replay_layer(std::size_t layer, moe::Resources& resources, ExpertBankId bank,
                                Tensor& output) {
    resources.before_layer(bank);
    auto& stages        = executable_->target_->layers.at(layer);
    const bool combined = resources.settled() && stages.steady.ready();
    if (!combined) stages.mixer.launch(cuda_.stream());
    const auto call = resources.prepare_layer(bank);
    try {
        if (combined)
            stages.steady.launch(cuda_.stream());
        else
            stages.moe[moe_graph_index(call.partition.gpu_kind)].launch(cuda_.stream());
        resources.gpu_submitted(bank);
        first_submitted();
        resources.finish_layer(moe::complete(call, output));
    } catch (...) {
        resources.abort_layer(bank);
        throw;
    }
    if (stages.feature.ready()) stages.feature.launch(cuda_.stream());
}

} // namespace ninfer::models::qwen3_5::execution
