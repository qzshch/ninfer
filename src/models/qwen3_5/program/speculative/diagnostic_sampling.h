#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace ninfer::models::qwen3_5::detail {

inline std::uint32_t diagnostic_environment_value(const char* text, std::uint32_t fallback,
                                                  std::uint32_t maximum) {
    if (text == nullptr) { return fallback; }
    const std::string_view view(text);
    std::uint32_t result = 0;
    const auto parsed = std::from_chars(view.data(), view.data() + view.size(), result);
    if (view.empty() || parsed.ec != std::errc{} || parsed.ptr != view.data() + view.size() ||
        result > maximum) {
        throw std::invalid_argument("invalid bounded DFlash diagnostic environment setting");
    }
    return result;
}

inline bool support_frontier_environment(const char* text) {
    if (text == nullptr) return false;
    const std::string_view value(text);
    if (value == "0") return false;
    if (value == "1") return true;
    throw std::invalid_argument("NINFER_DFLASH_SUPPORT_FRONTIER must be 0 or 1");
}

inline bool sample_dflash_diagnostic_round(std::uint64_t round, std::size_t collected,
                                          std::uint32_t maximum, std::uint32_t every) noexcept {
    return maximum != 0 && every != 0 && collected < maximum && round % every == 0;
}

} // namespace ninfer::models::qwen3_5::detail
