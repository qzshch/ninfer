#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define NINFER_DIAGNOSTIC_HD __host__ __device__
#else
#define NINFER_DIAGNOSTIC_HD
#endif

namespace ninfer {

// Two optional fixed-size packets per sampled compact row: first rejection/full,
// and one reached accepted proposal. Floats are unavailable in greedy mode except qd.
struct SpeculativeProposalDiagnostic {
    std::int32_t valid = 0;
    std::int32_t stochastic = 0;
    std::int32_t kind = 0; // 0 rejection, 1 reached accepted, 2 full, 3 zero-proposal fallback
    std::int32_t position = -1; // zero based; -1 means no proposal exists at this packet
    std::int32_t extent = 0;
    std::int32_t accepted = 0;
    std::int32_t proposal_id = -1;
    std::int32_t target_top1 = -1;
    std::int32_t target_support_size = 0;
    std::int32_t draft_in_target_support = 0;
    std::int32_t target_top1_in_proposal_support = 0;
    std::int32_t reserved = 0;
    float pd = 0;
    float qd = 0;
    float u = 0;
    float acceptance_probability = 0;
};
static_assert(sizeof(SpeculativeProposalDiagnostic) == 64);
static_assert(std::is_standard_layout_v<SpeculativeProposalDiagnostic>);
static_assert(std::is_trivially_copyable_v<SpeculativeProposalDiagnostic>);

// Consumes the actual normalized bounded supports. Does not filter, sample, mutate
// state, or use a reconstructed dense q. Shared by the optional Device observer and Host oracle.
NINFER_DIAGNOSTIC_HD inline SpeculativeProposalDiagnostic make_speculative_proposal_diagnostic(
    bool stochastic, std::int32_t kind, std::int32_t position, std::int32_t extent,
    std::int32_t accepted, std::int32_t proposal, std::int32_t target_top1,
    const std::int32_t* p_ids, const float* p_prob, std::int32_t p_count,
    const std::int32_t* q_ids, const float* q_prob, std::int32_t q_count, float u) noexcept {
    SpeculativeProposalDiagnostic packet;
    packet.valid = 1;
    packet.stochastic = stochastic;
    packet.kind = kind;
    packet.position = position;
    packet.extent = extent;
    packet.accepted = accepted;
    packet.proposal_id = proposal;
    packet.target_top1 = target_top1;
    packet.target_support_size = p_count;
    packet.u = stochastic && position >= 0 ? u : 0;
    if (position < 0) { return packet; }
    for (std::int32_t i = 0; i < p_count; ++i) {
        if (p_ids[i] == proposal) {
            packet.draft_in_target_support = 1;
            if (stochastic) { packet.pd = p_prob[i]; }
        }
    }
    for (std::int32_t i = 0; i < q_count; ++i) {
        if (q_ids[i] == proposal) { packet.qd = q_prob[i]; }
        if (q_ids[i] == target_top1 && q_prob[i] > 0) {
            packet.target_top1_in_proposal_support = 1;
        }
    }
    if (stochastic && packet.qd > 0) {
        packet.acceptance_probability = packet.pd >= packet.qd ? 1.0f : packet.pd / packet.qd;
    }
    return packet;
}

} // namespace ninfer

#undef NINFER_DIAGNOSTIC_HD
