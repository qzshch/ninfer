#pragma once

#include "ninfer/speculative_diagnostics.h"
#include "ops/kernel/speculative_round.cuh"

namespace ninfer::ops {

// Observes the original kernel's retained p support, q, and accepted count.
// Sampling RNG is stateless: the exact accept-domain counter is recomputed without
// consuming or changing RNG state. All selection and correction kernels stay unchanged.
__global__ void speculative_collect_sparse_diagnostics_kernel(
    const int* target_tokens, const int* drafts, const int* candidate_ids, const float* proposal_q,
    const int* current_extents, const int* round_lengths, const int* accepted,
    const SamplingConfig* configs, bool raw_greedy, SamplingWorkspace workspace,
    std::size_t workspace_row_stride, const int* mask,
    ninfer::SpeculativeProposalDiagnostic* packets, int k) {
    const int row = static_cast<int>(blockIdx.x);
    const int which = static_cast<int>(threadIdx.x);
    if (which >= 2) { return; }
    auto& output = packets[row * 2 + which];
    output = ninfer::SpeculativeProposalDiagnostic{};
    if (mask[row] == 0) { return; }
    const int p = min(k, max(0, current_extents[row]));
    const int a = accepted[row];
    if (which == 1 && a == 0) { return; }
    const int position = which == 0 ? (a < p ? a : -1) : (mask[row] - 1) % a;
    const int column = position < 0 ? p : position;
    const auto cfg = configs[row];
    const bool stochastic = cfg.temperature > 0.0f;
    const bool penalties = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    const bool direct_top1 = raw_greedy || (!stochastic && !penalties);
    if (!direct_top1) {
        workspace = speculative_workspace_row(workspace, workspace_row_stride, row);
    }
    const int top1 = direct_top1 ? target_tokens[row * (k + 1) + column]
                                : workspace.dist_idx[sampling_dist_offset(column, 0)];
    const int support = stochastic ? workspace.dist_support[column] : 1;
    const int* p_ids = stochastic ? workspace.dist_idx + sampling_dist_offset(column, 0) : &top1;
    const float* p_prob = stochastic ? workspace.dist_prob + sampling_dist_offset(column, 0) : nullptr;
    const int proposal = position < 0 ? -1 : drafts[row * k + position];
    const int q_at = (row * k + max(0, position)) * kSparseSpeculativeCandidates;
    const int old_length = round_lengths[row] - a - 1;
    const float u = stochastic && position >= 0
                        ? sampling_uniform(cfg.seed, old_length + position + 1,
                                           kSamplePurposeSpeculativeAccept, 0) : 0;
    output = ninfer::make_speculative_proposal_diagnostic(
        stochastic, which == 1 ? 1 : (p == 0 ? 3 : position < 0 ? 2 : 0), position, p, a, proposal, top1,
        p_ids, p_prob, support, candidate_ids + q_at, proposal_q + q_at,
        position < 0 ? 0 : kSparseSpeculativeCandidates, u);
}

} // namespace ninfer::ops
