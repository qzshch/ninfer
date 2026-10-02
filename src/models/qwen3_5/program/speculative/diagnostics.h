#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

// Pure Host accounting from the acceptance egress already required by execution.
// A position can be proposed but never reached after an earlier rejection.
inline void record_speculative_verification(SpeculativeStats& stats, std::uint32_t extent,
                                             std::uint32_t accepted) {
    const auto width = stats.draft_window;
    if (extent > width || accepted > extent || stats.accepted_per_position.size() != width ||
        stats.attempted_per_position.size() != width || stats.reached_per_position.size() != width ||
        stats.rejected_per_position.size() != width) {
        throw std::logic_error("speculative diagnostic extent is invalid");
    }
    stats.licensed_output_tokens += accepted + 1U;
    if (extent == 0) {
        ++stats.fallback_steps;
        return;
    }
    ++stats.rounds;
    stats.drafted_tokens += extent;
    stats.accepted_tokens += accepted;
    if (accepted == 0) {
        ++stats.zero_accept_rounds;
    } else if (accepted == extent) {
        ++stats.full_accept_rounds;
    } else {
        ++stats.partial_accept_rounds;
    }
    for (std::uint32_t i = 0; i < extent; ++i) {
        ++stats.attempted_per_position[i];
        if (i <= accepted) { ++stats.reached_per_position[i]; }
        if (i < accepted) {
            ++stats.accepted_per_position[i];
        } else if (i == accepted) {
            ++stats.rejected_per_position[i];
        }
    }
}

// Frontend may publish a shorter licensed prefix for stop/EOS/budget or zero on
// cancellation. Only a successful commit calls this for a non-cancelled row.
inline void record_speculative_publication(SpeculativeStats& stats, std::uint32_t licensed,
                                           std::uint32_t published) {
    if (licensed == 0 || licensed > stats.draft_window + 1U || published > licensed) {
        throw std::logic_error("speculative diagnostic publication is invalid");
    }
    stats.published_output_tokens += published;
    stats.published_accepted_tokens += std::min(licensed - 1U, published);
    stats.discarded_licensed_tokens += licensed - published;
    if (!stats.diagnostic_samples.empty()) {
        auto& sample = stats.diagnostic_samples.back();
        if (sample.round_index + 1 == stats.rounds + stats.fallback_steps) {
            if (sample.licensed_tokens != licensed || sample.publication_recorded) {
                throw std::logic_error("sampled speculative publication does not match licensing");
            }
            sample.published_tokens = published;
            sample.publication_recorded = true;
        }
    }
}

} // namespace ninfer::models::qwen3_5::detail
