#pragma once
#include <cstdint>

namespace ninfer::runtime::detail {
struct SparseEpochBinding {
    std::uint64_t request_epoch = 0;
    std::uint64_t engine_request_id = 0;
};

// Engine-owner boundary only. A new materializing Engine id cannot rebind old
// Program counters. An epoch without an owner remains explicitly unbound.
[[nodiscard]] inline bool observe_sparse_epoch(SparseEpochBinding& binding,
                                               std::uint64_t epoch,
                                               std::uint64_t current_engine_request_id) noexcept {
    if (binding.request_epoch == epoch) { return false; }
    binding.request_epoch = epoch;
    binding.engine_request_id = epoch != 0 ? current_engine_request_id : 0;
    return true;
}
// Complete requests can be shorter than the normal sampling interval. A
// terminal boundary must refresh before request storage can be reused; this
// does not increase sampling frequency on ordinary decode boundaries.
[[nodiscard]] inline bool sparse_snapshot_refresh_due(bool force_final_boundary,
                                                      bool epoch_changed,
                                                      bool first_sample,
                                                      std::int64_t elapsed_ns) noexcept {
    return force_final_boundary || epoch_changed || first_sample || elapsed_ns >= 1000000000LL;
}

[[nodiscard]] inline bool sparse_terminal_refresh_due(bool owns_lane,
                                                      std::uint64_t epoch,
                                                      std::uint64_t last_terminal_epoch) noexcept {
    return owns_lane && epoch != 0 && epoch != last_terminal_epoch;
}

} // namespace ninfer::runtime::detail
