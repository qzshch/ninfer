#pragma once
#include <algorithm>
#include <cstdint>

namespace ninfer::runtime {

// One shared budget for all staged owners between decode rounds. Atomic media
// groups may overrun a grant; the debt is carried into subsequent decode rounds.
// With no decoder, share the startup workspace chunk without shrinking a unit
// below the configured mixed-work budget (or the workspace chunk if smaller).
// Round-robin service provides fairness; cold work cannot owe a later decoder credit.
class PrefillBudget {
public:
    explicit PrefillBudget(std::uint32_t tokens, std::uint32_t milliseconds = 0,
                            std::uint32_t request_cap = 0)
        : limit_(tokens), time_limit_ns_(std::uint64_t(milliseconds) * 1000000),
          request_cap_(request_cap) {}

    void observe(std::uint32_t tokens, std::uint64_t elapsed_ns) noexcept {
        last_elapsed_ns_ = elapsed_ns;
        // Tiny finalization/capture units are dominated by fixed launch cost. Using
        // their ns/token as a long-chunk rate can collapse the next grant to one token.
        // Their real elapsed time still consumes the shared time credit below.
        if (tokens < 128 || elapsed_ns == 0) return;
        const auto sample = double(elapsed_ns) / tokens;
        // Rise promptly when long-context work becomes expensive; decay slowly.
        ns_per_token_ = ns_per_token_ == 0.0 ? sample
            : std::max(sample, 0.8 * ns_per_token_ + 0.2 * sample);
    }

    bool enabled() const noexcept { return limit_ != 0; }

    static std::uint32_t packed_allowance(bool have_decode, std::uint32_t chunk,
                                          std::uint32_t row_grant,
                                          std::uint32_t owners) noexcept {
        // Submission grouping must not buy capacity by shrinking the scalar
        // row shape or merge multiple grants across a decoder service boundary.
        if (have_decode || owners < 2) return 0;
        const auto total = std::uint64_t(row_grant) * owners;
        return total <= chunk ? static_cast<std::uint32_t>(total) : 0U;
    }

    std::uint32_t allowance(bool have_decode, std::uint32_t chunk,
                            std::uint32_t runnable_owners = 1) noexcept {
        if (request_cap_ != 0) chunk = std::min(chunk, request_cap_);
        if (have_decode && time_limit_ns_ != 0 && ns_per_token_ > 0.0) {
            if (enabled() && time_credit_ns_ <= 0) return 0;
            const auto time_left = enabled() ? double(time_credit_ns_) : double(time_limit_ns_);
            const auto estimate = time_left / (1.25 * ns_per_token_);
            chunk = std::min(chunk, static_cast<std::uint32_t>(
                std::max(1.0, std::min(double(chunk), estimate))));
        }
        if (!enabled()) return chunk;
        mixed_grant_ = have_decode;
        if (!have_decode) {
            credit_ = 0;
            time_credit_ns_ = 0;
            const auto share = std::max(1U, chunk / std::max(1U, runnable_owners));
            return std::min(chunk, std::max(limit_, share));
        }
        return credit_ <= 0 ? 0U
                            : static_cast<std::uint32_t>(std::min<std::int64_t>(credit_, chunk));
    }

    void consume(std::uint32_t actual_tokens) noexcept {
        if (enabled() && mixed_grant_) {
            credit_ -= std::max(1U, actual_tokens);
            if (time_limit_ns_ != 0) time_credit_ns_ -= static_cast<std::int64_t>(last_elapsed_ns_);
        }
    }

    void decode_completed() noexcept {
        if (enabled()) {
            credit_ = std::min<std::int64_t>(limit_, credit_ + limit_);
            if (time_limit_ns_ != 0) time_credit_ns_ = std::min<std::int64_t>(
                time_limit_ns_, time_credit_ns_ + static_cast<std::int64_t>(time_limit_ns_));
        }
    }
private:
    std::uint32_t limit_;
    std::uint64_t time_limit_ns_ = 0;
    std::uint32_t request_cap_ = 0;
    double ns_per_token_ = 0.0;
    std::uint64_t last_elapsed_ns_ = 0;
    std::int64_t credit_ = 0;
    std::int64_t time_credit_ns_ = 0;
    bool mixed_grant_ = true;
};
} // namespace ninfer::runtime
