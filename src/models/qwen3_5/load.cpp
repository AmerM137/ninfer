#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "models/qwen3_5/load/bindings.h"

#include <utility>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    const artifact::Reader* reader = nullptr;
    artifact::PreparedResources prepared_resources;
    std::vector<ExpertBankDescription> expert_banks;
    FrontendResources resources;
    InstanceInfo info;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

std::span<const ExpertBankDescription> LoadPlan::expert_banks() const {
    return impl_->expert_banks;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->reader  = &reader;
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, options);
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(bindings, *out->config.vision, text);
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    out->pending            = std::move(bindings.weights);
    out->expert_banks       = describe_expert_weights(reader, out->weights, out->pending);
    out->prepared_resources = std::move(binder).finish();
    out->info.name          = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    return LoadPlan(std::move(out));
}

ResolvedLoadPlan::ResolvedLoadPlan(LoadPlan&& logical,
                                   artifact::MaterializationPlan&& materialization,
                                   std::vector<Device>&& devices,
                                   std::vector<Device>&& bank_devices)
    : materialization_(std::move(materialization)), logical_(std::move(logical)),
      devices_(std::move(devices)), bank_devices_(std::move(bank_devices)) {}

ResolvedLoadPlan resolve_load(LoadPlan&& plan, Device cuda, std::optional<Device> main_experts) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    const auto expert_device = main_experts.value_or(cuda);
    if (cuda.type() != DeviceType::CUDA ||
        (expert_device != cuda && expert_device != Device::cpu())) {
        throw artifact::ArtifactError("weight placement requires the selected GPU or CPU");
    }
    auto& data = *plan.impl_;
    std::vector<Device> devices(data.pending.size(), cuda);
    std::vector<Device> bank_devices;
    for (const auto& bank : data.expert_banks) {
        bank_devices.push_back(bank.mtp ? cuda : expert_device);
    }
    const auto place_experts = [&](const BlockWeights& layer) {
        if (const auto* moe = std::get_if<MoeWeights>(&layer.ffn)) {
            for (const auto& expert : moe->experts) {
                for (const auto id : {expert.gate, expert.up, expert.down}) {
                    devices[id.index] = bank_devices[moe->bank.index];
                }
            }
        }
    };
    for (const auto& layer : data.weights.text.layers) { place_experts(layer); }
    if (data.weights.mtp) { place_experts(data.weights.mtp->layer); }
    std::vector<artifact::PlacementRequest> requests;
    for (std::size_t i = 0; i < data.pending.size(); ++i) {
        for (const auto& part : data.pending[i].reference.binding.parts) {
            requests.push_back({part.object, devices[i],
                                devices[i] == Device::cpu() ? artifact::HostAccess::CudaTransfer
                                                            : artifact::HostAccess::CpuOnly});
        }
    }
    auto placement =
        artifact::plan_materialization(*data.reader, requests, std::move(data.prepared_resources));
    return ResolvedLoadPlan(std::move(plan), std::move(placement), std::move(devices),
                            std::move(bank_devices));
}

std::unique_ptr<Model> materialize_model(ResolvedLoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.logical_.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.logical_.impl_);
    auto backing = artifact::materialize(*data->reader, std::move(plan.materialization_),
                                         device.transfer_view(), observer);
    auto bound   = loading::resolve_weights(std::move(data->pending), backing, plan.devices_);
    auto banks   = bind_expert_weights(data->expert_banks, backing, plan.bank_devices_);
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(banks), std::move(data->resources), std::move(data->info), std::move(backing)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(resolve_load(plan_load(reader, options), device.device()), device,
                             observer);
}

} // namespace ninfer::models::qwen3_5
