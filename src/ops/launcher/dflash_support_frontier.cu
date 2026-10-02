#include "ops/launcher/dflash_support_frontier.h"
#include "ops/kernel/dflash_support_frontier.cuh"
#include "core/device.h"
namespace ninfer::ops::detail {
void speculative_collect_support_frontier_launch(
    const Tensor& logits, const Tensor& verify_ids, const Tensor& drafts, const Tensor& candidate_ids,
    const Tensor& proposal_q, std::int32_t token_domain, const SamplingConfig* configs,
    const Tensor& packets, Tensor& support_frontiers, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(drafts.ne[1]), 2U);
    dflash_collect_support_frontier_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), logits.ne[0], token_domain,
        static_cast<const int*>(verify_ids.data), static_cast<const int*>(drafts.data), static_cast<const int*>(candidate_ids.data),
        static_cast<const float*>(proposal_q.data), configs,
        static_cast<const ninfer::SpeculativeProposalDiagnostic*>(packets.data),
        static_cast<ninfer::DFlashSupportFrontier*>(support_frontiers.data), drafts.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
