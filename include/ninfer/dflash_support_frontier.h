#pragma once

#include <cstdint>
#include <cstddef>
#include <span>
#include <type_traits>
#include <vector>

namespace ninfer {

// Observer-only packet. The optional execution path is disabled by default.
enum class DFlashSupportStage : std::int32_t {
    None = 0,
    TopK = 1,
    TopP = 2,
    AfterTopKUnresolved = 3,
    PositiveProbability = 4,
    Greedy = 5,
    Invalid = 6,
    ZeroProbability = 7,
};

struct DFlashSupportFrontier {
    std::int32_t valid = 0;
    std::int32_t raw_rank = -1;
    std::int32_t adjusted_rank = -1;
    std::int32_t effective_top_k = 0;
    std::int32_t committed_count = 0;
    std::int32_t overlay_count = 0;
    std::int32_t proposal_in_raw_candidates = 0;
    std::int32_t target_top1_in_raw_candidates = 0;
    std::int32_t target_top1_candidate_rank = -1;
    std::int32_t penalty_crossed_top_k = 0;
    DFlashSupportStage stage = DFlashSupportStage::None;
    std::int32_t round_anchor_id = -1;
    float raw_logit = 0;
    float adjusted_logit = 0;
    float target_top1_q = 0;
    float proposal_q = 0;
};
static_assert(sizeof(DFlashSupportFrontier) == 64);
static_assert(std::is_standard_layout_v<DFlashSupportFrontier>);
static_assert(offsetof(DFlashSupportFrontier, round_anchor_id) == 44);
static_assert(offsetof(DFlashSupportFrontier, raw_logit) == 48);
inline constexpr std::size_t kDFlashPromptBindingMaximumTokens = 512;
inline std::vector<std::int32_t> bounded_dflash_prompt_identity(
    bool enabled, bool has_media, std::span<const std::int32_t> ids) {
    if (!enabled || has_media || ids.empty() || ids.size() > kDFlashPromptBindingMaximumTokens)
        return {};
    return {ids.begin(), ids.end()};
}

#if defined(__CUDACC__)
#define NINFER_FRONTIER_HD __host__ __device__
#else
#define NINFER_FRONTIER_HD
#endif

// Labels one observed stage; penalty crossing is an auxiliary flag, not a second
// rejection. min-p>0 remains conservative unless full boundary data are supplied.
NINFER_FRONTIER_HD inline DFlashSupportStage classify_dflash_support_frontier(
    bool stochastic, int adjusted_rank, int effective_top_k,
    bool retained_by_target, float min_p, float pd, float qd) noexcept {
    if (!stochastic) return DFlashSupportStage::Greedy;
    if (!(qd > 0 && qd <= 1) || !(pd >= 0 && pd <= 1) ||
        adjusted_rank < 1 || effective_top_k < 1) return DFlashSupportStage::Invalid;
    if (retained_by_target) {
        if (adjusted_rank > effective_top_k) return DFlashSupportStage::Invalid;
        return pd > 0 ? DFlashSupportStage::PositiveProbability : DFlashSupportStage::ZeroProbability;
    }
    if (pd != 0) return DFlashSupportStage::Invalid;
    if (adjusted_rank > effective_top_k) return DFlashSupportStage::TopK;
    return min_p == 0 ? DFlashSupportStage::TopP : DFlashSupportStage::AfterTopKUnresolved;
}

} // namespace ninfer

#undef NINFER_FRONTIER_HD
