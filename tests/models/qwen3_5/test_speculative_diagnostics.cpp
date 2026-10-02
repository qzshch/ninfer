#include "models/qwen3_5/program/speculative/diagnostics.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace {
int failures = 0;
void expect(bool ok, std::string_view label) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
ninfer::SpeculativeStats stats(std::uint32_t width) {
    ninfer::SpeculativeStats result;
    result.backend = ninfer::SpeculativeBackend::DFlash2;
    result.enabled = true;
    result.draft_window = width;
    result.accepted_per_position.assign(width, 0);
    result.attempted_per_position.assign(width, 0);
    result.reached_per_position.assign(width, 0);
    result.rejected_per_position.assign(width, 0);
    return result;
}
std::uint64_t sum(const std::vector<std::uint64_t>& values) {
    return std::accumulate(values.begin(), values.end(), std::uint64_t{0});
}
void conservation(const ninfer::SpeculativeStats& s) {
    expect(s.rounds == s.zero_accept_rounds + s.partial_accept_rounds + s.full_accept_rounds,
           "nonempty round categories conserve");
    expect(s.drafted_tokens == sum(s.attempted_per_position), "attempts conserve live extents");
    expect(s.accepted_tokens == sum(s.accepted_per_position), "accepted positions conserve");
    expect(s.zero_accept_rounds + s.partial_accept_rounds == sum(s.rejected_per_position),
           "exactly one first rejection in each rejecting round");
    expect(s.licensed_output_tokens == s.rounds + s.fallback_steps + s.accepted_tokens,
           "target licenses an accepted prefix and one correction or bonus");
    expect(s.licensed_output_tokens == s.published_output_tokens + s.discarded_licensed_tokens,
           "publication and discarded tail conserve target licensing");
    expect(s.published_accepted_tokens <= s.accepted_tokens &&
               s.published_accepted_tokens <= s.published_output_tokens,
           "published accepted drafts are bounded");
    for (std::size_t i = 0; i < s.draft_window; ++i) {
        expect(s.reached_per_position[i] == s.accepted_per_position[i] + s.rejected_per_position[i],
               "reached position accepts or rejects");
        expect(s.reached_per_position[i] <= s.attempted_per_position[i],
               "unreached suffix is not a rejection");
    }
}
} // namespace

int main() {
    using namespace ninfer::models::qwen3_5::detail;
    auto lane = stats(7);
    // Full width rejects at its third proposal; later proposals are not reached.
    record_speculative_verification(lane, 7, 2);
    record_speculative_publication(lane, 3, 3);
    expect(lane.attempted_per_position == std::vector<std::uint64_t>({1,1,1,1,1,1,1}) &&
               lane.reached_per_position == std::vector<std::uint64_t>({1,1,1,0,0,0,0}) &&
               lane.rejected_per_position == std::vector<std::uint64_t>({0,0,1,0,0,0,0}),
           "first rejection ends reached prefix");
    // Short tail accepts both available proposals, although configured K remains 7.
    record_speculative_verification(lane, 2, 2);
    record_speculative_publication(lane, 3, 1); // Frontend stop keeps one accepted draft.
    record_speculative_verification(lane, 7, 0);
    record_speculative_publication(lane, 1, 1); // Target correction, zero accepted drafts.
    record_speculative_verification(lane, 7, 7);
    record_speculative_publication(lane, 8, 0); // Cancellation discards complete license.
    record_speculative_verification(lane, 0, 0);
    record_speculative_publication(lane, 1, 1); // Budget-limited anchor fallback.
    conservation(lane);
    expect(lane.rounds == 4 && lane.fallback_steps == 1 && lane.full_accept_rounds == 2 &&
               lane.zero_accept_rounds == 1 && lane.partial_accept_rounds == 1,
           "short all-accept and fallback are distinct categories");
    expect(lane.published_accepted_tokens == 3 && lane.discarded_licensed_tokens == 10,
           "licensed acceptance differs from the published draft prefix");
    const auto original = lane;
    auto other_lane = stats(7);
    record_speculative_verification(other_lane, 3, 1);
    record_speculative_publication(other_lane, 2, 2);
    conservation(other_lane);
    expect(lane.accepted_tokens == original.accepted_tokens &&
               lane.attempted_per_position == original.attempted_per_position,
           "concurrent lane accounting has no shared counters");
    // Exhaust all legal widths, rejection boundaries and publication lengths.
    for (std::uint32_t k = 1; k <= 15; ++k) {
        for (std::uint32_t p = 0; p <= k; ++p) {
            for (std::uint32_t a = 0; a <= p; ++a) {
                for (std::uint32_t n = 0; n <= a + 1; ++n) {
                    auto s = stats(k);
                    record_speculative_verification(s, p, a);
                    record_speculative_publication(s, a + 1, n);
                    conservation(s);
                    expect(s.published_accepted_tokens == std::min(a, n),
                           "stop inside accepted prefix retains exactly that prefix");
                }
            }
        }
    }
    auto invalid = stats(7);
    bool threw = false;
    try { record_speculative_verification(invalid, 2, 3); }
    catch (const std::logic_error&) { threw = true; }
    expect(threw && invalid.rounds == 0 && invalid.licensed_output_tokens == 0,
           "invalid egress is rejected before accounting changes");
    std::cout << "speculative diagnostic failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
