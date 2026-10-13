#include "models/common/moe/weights.h"

namespace ninfer::models::moe {
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

BankDescription describe_bank(const artifact::Reader& reader,
                              std::span<const ExpertBinding> experts) {
    if (experts.empty()) throw artifact::ArtifactError("empty expert bank");
    const auto& gate     = experts.front().gate;
    const auto& down     = experts.front().down;
    const auto gu_origin = region(gate), down_origin = region(down);
    const auto n = gate.shape.at(0), k = gate.shape.at(1);
    const auto count = static_cast<std::uint32_t>(experts.size());
    const auto check = [&](const artifact::ParameterReference& reference,
                           artifact::ObjectHandle object, std::uint64_t begin, std::uint64_t rows,
                           std::uint64_t columns) {
        const auto part = region(reference);
        if (reference.shape != artifact::Shape{rows, columns} || part.object != object ||
            part.begin != begin || part.end != begin + rows * columns)
            throw artifact::ArtifactError("expert Bindings do not form an affine bank");
    };
    for (std::size_t e = 0; e < experts.size(); ++e) {
        check(experts[e].gate, gu_origin.object, gu_origin.begin + e * 2 * n * k, n, k);
        check(experts[e].up, gu_origin.object, gu_origin.begin + (e * 2 + 1) * n * k, n, k);
        check(experts[e].down, down_origin.object, down_origin.begin + e * n * k, k, n);
    }
    if (gu_origin.begin % k || down_origin.begin % n)
        throw artifact::ArtifactError("expert Binding splits a row");
    return {{gu_origin.object, describe_weight_bank(reader.geometry(gu_origin.object),
                                                    {gu_origin.begin / k, count, 2 * n, 2 * n})},
            {down_origin.object, describe_weight_bank(reader.geometry(down_origin.object),
                                                      {down_origin.begin / n, count, k, k})}};
}

std::vector<ExpertBankSource> bind_banks(std::span<const BankDescription> banks,
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
        const auto bind   = [&](const MatrixSource& source) {
            const auto& parent = device.type() == DeviceType::CPU
                                       ? backing.host_parent(source.object)
                                       : backing.device_parent(source.object);
            if (parent.device != device) {
                throw artifact::ArtifactError("expert source device differs");
            }
            return bind_weight_bank(parent, source.layout);
        };
        const auto upload = [&](const MatrixSource& source) {
            return device == Device::cpu() ? backing.host_transfer_view(source.object)
                                           : CudaHostView{};
        };
        out.push_back(
            {bind(bank.gate_up), bind(bank.down), upload(bank.gate_up), upload(bank.down)});
    }
    return out;
}

} // namespace ninfer::models::moe
