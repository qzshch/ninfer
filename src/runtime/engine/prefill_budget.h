#pragma once
#include <algorithm>
#include <cstdint>

namespace ninfer::runtime {

// One shared budget for all staged owners between decode rounds. Atomic media
// groups may overrun a grant; the debt is carried into subsequent decode rounds.
// Without a decoder, replenish at unit boundaries so cold input cannot deadlock.
class PrefillBudget {
public:
    explicit PrefillBudget(std::uint32_t tokens) : limit_(tokens) {}

    bool enabled() const noexcept { return limit_ != 0; }

    std::uint32_t allowance(bool have_decode, std::uint32_t chunk) noexcept {
        if (!enabled()) return chunk;
        if (!have_decode && credit_ <= 0) credit_ = limit_;
        return credit_ <= 0 ? 0U
                            : static_cast<std::uint32_t>(std::min<std::int64_t>(credit_, chunk));
    }

    void consume(std::uint32_t actual_tokens) noexcept {
        if (enabled()) credit_ -= std::max(1U, actual_tokens);
    }

    void decode_completed() noexcept {
        if (enabled()) credit_ = std::min<std::int64_t>(limit_, credit_ + limit_);
    }
private:
    std::uint32_t limit_;
    std::int64_t credit_ = 0;
};
} // namespace ninfer::runtime
