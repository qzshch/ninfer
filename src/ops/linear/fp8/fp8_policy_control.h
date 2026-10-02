#pragma once

#include "ninfer/ops/linear.h"

#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace ninfer::ops::detail {

// Independent precision-isolation control. Fixed before planning and Graph capture.
inline bool parse_fp8_small_t_a16_control(const char* raw) {
    if (raw == nullptr || std::string_view(raw) == "0") return false;
    if (std::string_view(raw) == "1") return true;
    throw std::invalid_argument("NINFER_FP8_SMALL_T_A16 must be 0 or 1");
}

inline bool fp8_small_t_a16_control_enabled() {
    static const bool enabled =
        parse_fp8_small_t_a16_control(std::getenv("NINFER_FP8_SMALL_T_A16"));
    return enabled;
}

constexpr LinearPolicy fp8_small_t_policy_for_control(LinearPolicy original,
                                                       std::int32_t tokens,
                                                       bool force_a16) noexcept {
    // Invalid policies remain invalid for the caller's ordinary validation.
    return valid_linear_policy(original) && force_a16 && tokens > 0 && tokens <= 32
               ? LinearPolicy::A16Only : original;
}

inline LinearPolicy fp8_small_t_policy(LinearPolicy original, std::int32_t tokens) {
    return fp8_small_t_policy_for_control(original, tokens, fp8_small_t_a16_control_enabled());
}

} // namespace ninfer::ops::detail
