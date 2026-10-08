#include "models/qwen3_5/execution/gdn.h"
#include "ninfer/ops/gdn_control.h"
#include "ninfer/ops/gdn_input_proj.h"

namespace ninfer::models::qwen3_5::execution {

std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters, std::int32_t first,
                                           std::int32_t last) {
    return ops::gdn_input_proj_workspace_capacity_bytes(parameters.projection, first, last);
}

std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters, std::int32_t batch,
                                         std::int32_t first_width, std::int32_t last_width) {
    return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(parameters.projection, batch,
                                                                      first_width, last_width);
}

std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters, std::int32_t batch,
                                       std::int32_t first_width, std::int32_t last_width) {
    return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(parameters.projection, batch,
                                                                    first_width, last_width);
}

void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    ops::gdn_input_proj(hidden, parameters.projection, {qkv, z}, workspace, stream);
}

void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      WorkspaceArena& workspace, DeviceExecutionView execution) {
    ops::gdn_norm_control_proj(residual, norm, epsilon, parameters.control, parameters.a_log,
                               parameters.dt_bias, {hidden, {g, beta}}, workspace, execution);
}

void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slots, const Tensor& destination_slots,
                             Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::gdn_input_proj_conv_snapshot(hidden, parameters.projection, parameters.convolution,
                                      {conv_states, initial_slots, destination_slots},
                                      valid_columns, {query, key, value, z}, workspace, stream);
}

void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slots, Tensor& conv_record, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& workspace,
                           cudaStream_t stream) {
    ops::gdn_input_proj_conv_record(hidden, parameters.projection, parameters.convolution,
                                    {conv_states, initial_slots}, valid_columns,
                                    {query, key, value, z}, conv_record, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
