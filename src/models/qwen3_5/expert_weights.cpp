#include "models/qwen3_5/expert_weights.h"
#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5 {
namespace {

artifact::Part region(const artifact::ParameterReference& reference) {
    if (reference.binding.parts.empty()) { throw artifact::ArtifactError("empty expert Binding"); }
    auto out = reference.binding.parts.front();
    for (std::size_t i = 1; i < reference.binding.parts.size(); ++i) {
        const auto& next = reference.binding.parts[i];
        if (next.object != out.object || next.begin != out.end) {
            throw artifact::ArtifactError("expert matrix requires contiguous parent rows");
        }
        out.end = next.end;
    }
    return out;
}

} // namespace

std::vector<ExpertBankDescription>
describe_expert_weights(const artifact::Reader& reader, ModelWeights& weights,
                        std::span<const loading::PendingWeight> pending) {
    std::vector<ExpertBankDescription> out;
    const auto describe = [&](MoeWeights& moe, bool mtp) {
        if (moe.experts.empty()) { throw artifact::ArtifactError("empty expert bank"); }
        const auto& first      = moe.experts.front();
        const auto& gate       = pending[first.gate.index].reference;
        const auto& down       = pending[first.down.index].reference;
        const auto gu_origin   = region(gate);
        const auto down_origin = region(down);
        const auto n = gate.shape.at(0), k = gate.shape.at(1);
        const auto count = static_cast<std::uint32_t>(moe.experts.size());
        for (std::size_t e = 0; e < moe.experts.size(); ++e) {
            const auto& ids  = moe.experts[e];
            const auto check = [&](WeightId id, artifact::ObjectHandle object, std::uint64_t begin,
                                   std::uint64_t rows, std::uint64_t cols) {
                const auto& reference = pending[id.index].reference;
                const auto part       = region(reference);
                if (reference.shape != artifact::Shape{rows, cols} || part.object != object ||
                    part.begin != begin || part.end != begin + rows * cols) {
                    throw artifact::ArtifactError("expert Bindings do not form an affine bank");
                }
            };
            check(ids.gate, gu_origin.object, gu_origin.begin + e * 2 * n * k, n, k);
            check(ids.up, gu_origin.object, gu_origin.begin + (e * 2 + 1) * n * k, n, k);
            check(ids.down, down_origin.object, down_origin.begin + e * n * k, k, n);
        }
        if (gu_origin.begin % k || down_origin.begin % n) {
            throw artifact::ArtifactError("expert Binding splits a row");
        }
        const auto gu = describe_weight_bank(reader.geometry(gu_origin.object),
                                             {gu_origin.begin / k, count, 2 * n, 2 * n});
        const auto dn = describe_weight_bank(reader.geometry(down_origin.object),
                                             {down_origin.begin / n, count, k, k});
        moe.bank      = ExpertBankId{static_cast<std::uint32_t>(out.size())};
        out.push_back({{gu_origin.object, gu}, {down_origin.object, dn}, mtp});
    };
    for (auto& layer : weights.text.layers) {
        if (auto* moe = std::get_if<MoeWeights>(&layer.ffn)) { describe(*moe, false); }
    }
    if (weights.mtp) {
        if (auto* moe = std::get_if<MoeWeights>(&weights.mtp->layer.ffn)) { describe(*moe, true); }
    }
    return out;
}

std::vector<ExpertBankSource> bind_expert_weights(std::span<const ExpertBankDescription> banks,
                                                  const artifact::MaterializedArtifact& backing,
                                                  std::span<const Device> devices) {
    std::vector<ExpertBankSource> out;
    out.reserve(banks.size());
    if (devices.size() != banks.size()) {
        throw artifact::ArtifactError("expert placement count differs");
    }
    for (std::size_t i = 0; i < banks.size(); ++i) {
        const auto& bank  = banks[i];
        const auto device = devices[i];
        const auto bind   = [&](const ExpertMatrixSource& source) {
            const auto& parent = device.type() == DeviceType::CPU
                                       ? backing.host_parent(source.object)
                                       : backing.device_parent(source.object);
            if (parent.device != device) {
                throw artifact::ArtifactError("expert source device differs");
            }
            return bind_weight_bank(parent, source.layout);
        };
        const auto upload = [&](const ExpertMatrixSource& source) {
            return device == Device::cpu() ? backing.host_transfer_view(source.object)
                                           : CudaHostView{};
        };
        out.push_back(
            {bind(bank.gate_up), bind(bank.down), upload(bank.gate_up), upload(bank.down)});
    }
    return out;
}

} // namespace ninfer::models::qwen3_5
