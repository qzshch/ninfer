#pragma once
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
void speculative_collect_support_frontier_launch(
    const Tensor& logits, const Tensor& verify_ids, const Tensor& drafts, const Tensor& candidate_ids,
    const Tensor& proposal_q, std::int32_t token_domain, const SamplingConfig* configs,
    const Tensor& packets, Tensor& support_frontiers, cudaStream_t stream);
} // namespace ninfer::ops::detail
