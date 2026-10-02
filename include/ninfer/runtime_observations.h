#pragma once

// Host-only observation types. No Ops scheduling, numerical policy or Device API.
// Counters are supplied after their public definition, avoiding a types.h include cycle.
#include <cstdint>
#include <array>
#include <cstddef>

namespace ninfer {

template <class RequestCounters>
struct RuntimeDirectSparseLaneObservation {
    bool available = false;
    bool current_request = false;
    std::uint64_t request_epoch = 0;
    std::uint64_t engine_request_id = 0;
    RequestCounters request_counters;
};

template <class RequestCounters, std::size_t MaximumLanes>
struct RuntimeDirectSparseSamplingObservation {
    bool supported = false;
    std::uint64_t sample_revision = 0;
    std::uint64_t sampled_steady_ns = 0;
    std::uint64_t sample_age_ns = 0;
    std::array<RuntimeDirectSparseLaneObservation<RequestCounters>, MaximumLanes> lanes{};
};

} // namespace ninfer
