#pragma once

// Read-only observer, explicitly invoked only by the optional diagnostic path.
// No target sampling, selector, probability, history or RNG writes.
#include "ninfer/dflash_support_frontier.h"
#include "ninfer/speculative_diagnostics.h"
#include "ops/kernel/sampling_device.cuh"

namespace ninfer::ops {
inline constexpr int kDFlashSupportFrontierCandidates = 16;

// Call on the accept stream before any committed-count update. Sparse acceptance
// leaves token_counts unchanged; the observer reads and never modifies it.
// Only sampled packets with an actual proposal are scanned. Padding is excluded.
__global__ void dflash_collect_support_frontier_kernel(
    const __nv_bfloat16* logits, int physical_rows, int token_domain,
    const int* verify_ids, const int* drafts, const int* candidate_ids, const float* proposal_q,
    const SamplingConfig* configs, const ninfer::SpeculativeProposalDiagnostic* packets,
    ninfer::DFlashSupportFrontier* output, int k) {
    const int row = static_cast<int>(blockIdx.x);
    const int which = static_cast<int>(blockIdx.y);
    const auto packet = packets[row * 2 + which];
    if (threadIdx.x == 0) {
        output[row * 2 + which] = {};
        if (packet.valid != 0)
            output[row * 2 + which].round_anchor_id = verify_ids[row * (k + 1)];
    }
    if (packet.valid == 0 || packet.position < 0) return;
    const int position = packet.position;
    const int proposal = packet.proposal_id;
    if (position >= k || proposal < 0 || proposal >= token_domain) return;
    const auto cfg = configs[row];
    const int* overlay = drafts + row * k;
    const auto base = (static_cast<std::int64_t>(row) * (k + 1) + position) * physical_rows;
    const float raw = __bfloat162float(logits[base + proposal]);
    const float adjusted = sampling_adjusted_logit(raw, proposal, cfg, overlay, position);
    unsigned raw_better = 0, adjusted_better = 0;
    for (int token = threadIdx.x; token < token_domain; token += blockDim.x) {
        const float value = __bfloat162float(logits[base + token]);
        raw_better += sampling_better(value, token, raw, proposal);
        const float changed = sampling_adjusted_logit(value, token, cfg, overlay, position);
        adjusted_better += sampling_better(changed, token, adjusted, proposal);
    }
    __shared__ unsigned ranks[2][256];
    ranks[0][threadIdx.x] = raw_better;
    ranks[1][threadIdx.x] = adjusted_better;
    __syncthreads();
    // Candidate launch contract: exactly256threads; no shared atomics or config writes.
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            ranks[0][threadIdx.x] += ranks[0][threadIdx.x + stride];
            ranks[1][threadIdx.x] += ranks[1][threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    ninfer::DFlashSupportFrontier result;
    result.valid = 1;
    result.round_anchor_id = verify_ids[row * (k + 1)];
    result.raw_rank = static_cast<int>(ranks[0][0]) + 1;
    result.adjusted_rank = static_cast<int>(ranks[1][0]) + 1;
    result.effective_top_k = cfg.temperature > 0 ? sampling_candidate_cap(cfg, token_domain) : 1;
    result.committed_count = cfg.token_counts ? cfg.token_counts[proposal] : 0;
    for (int j = 0; j < position; ++j) result.overlay_count += overlay[j] == proposal;
    result.raw_logit = raw;
    result.adjusted_logit = adjusted;
    const int q_at = (row * k + position) * kDFlashSupportFrontierCandidates;
    for (int rank = 0; rank < kDFlashSupportFrontierCandidates; ++rank) {
        const int id = candidate_ids[q_at + rank];
        if (id == proposal) {
            result.proposal_in_raw_candidates = 1;
            result.proposal_q = proposal_q[q_at + rank];
        }
        if (id == packet.target_top1) {
            result.target_top1_in_raw_candidates = 1;
            result.target_top1_candidate_rank = rank;
            result.target_top1_q = proposal_q[q_at + rank];
        }
    }
    result.penalty_crossed_top_k = result.raw_rank <= result.effective_top_k &&
                                  result.adjusted_rank > result.effective_top_k;
    result.stage = ninfer::classify_dflash_support_frontier(
        packet.stochastic != 0, result.adjusted_rank, result.effective_top_k,
        packet.draft_in_target_support != 0, cfg.min_p, packet.pd, packet.qd);
    output[row * 2 + which] = result;
}

} // namespace ninfer::ops
