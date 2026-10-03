#pragma once

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {

// Keep a contiguous prefix whose predicted survival is at least 25%. Always
// verify one draft when the output budget permits. Fixed maximum resources and
// the complete proposal block remain reserved; only target verification is trimmed.
inline std::uint32_t confidence_draft_window(std::span<const float> confidence) {
    if (confidence.empty() || confidence.size() > 7) {
        throw std::invalid_argument("DSpark confidence block must have 1..7 positions");
    }
    double survival      = 1;
    std::uint32_t length = 1;
    for (std::size_t i = 0; i < confidence.size(); ++i) {
        const double p = confidence[i];
        if (!std::isfinite(p) || p < 0 || p > 1) {
            throw std::invalid_argument("DSpark confidence is not a probability");
        }
        survival *= p;
        if (survival >= 0.25) length = static_cast<std::uint32_t>(i + 1);
    }
    return length;
}
} // namespace ninfer::models::qwen3_5
