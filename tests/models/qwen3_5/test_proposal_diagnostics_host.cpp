#include "ninfer/speculative_diagnostics.h"
#include "models/qwen3_5/program/speculative/diagnostic_sampling.h"
#include "models/qwen3_5/program/speculative/diagnostics.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace ninfer;
    using namespace ninfer::models::qwen3_5::detail;
    int failures = 0;
    const auto expect = [&](bool ok) { if (!ok) { ++failures; } };
    const std::array<std::int32_t, 3> p{248076, 7, 42};
    const std::array<float, 3> pp{0.5f, 0.25f, 0.25f};
    const std::array<std::int32_t, 3> q{7, 42, 1234};
    const std::array<float, 3> qp{0.5f, 0.25f, 0.25f};
    auto packet = make_speculative_proposal_diagnostic(true, 0, 2, 7, 2, 7, p[0],
        p.data(), pp.data(), 3, q.data(), qp.data(), 3, 0.75f);
    expect(packet.pd == 0.25f && packet.qd == 0.5f && packet.acceptance_probability == 0.5f &&
           packet.u == 0.75f && packet.draft_in_target_support == 1 &&
           packet.target_top1_in_proposal_support == 0 && packet.position == packet.accepted);
    packet = make_speculative_proposal_diagnostic(true, 0, 0, 3, 0, 1234, p[0],
        p.data(), pp.data(), 3, q.data(), qp.data(), 3, 0.25f);
    expect(packet.pd == 0 && packet.qd == 0.25f && packet.draft_in_target_support == 0 &&
           packet.acceptance_probability == 0);
    packet = make_speculative_proposal_diagnostic(true, 1, 1, 3, 2, 42, p[0],
        p.data(), pp.data(), 3, q.data(), qp.data(), 3, 0.99f);
    expect(packet.pd == packet.qd && packet.acceptance_probability == 1);
    packet = make_speculative_proposal_diagnostic(false, 0, 0, 7, 0, 7, 248076,
        p.data(), nullptr, 1, q.data(), qp.data(), 3, 0.99f);
    expect(packet.stochastic == 0 && packet.pd == 0 && packet.u == 0 && packet.qd == 0.5f &&
           packet.draft_in_target_support == 0);
    packet = make_speculative_proposal_diagnostic(true, 2, -1, 7, 7, -1, p[0],
        p.data(), pp.data(), 3, nullptr, nullptr, 0, 0);
    expect(packet.valid == 1 && packet.kind == 2 && packet.position == -1 && packet.proposal_id == -1);
    expect(diagnostic_environment_value(nullptr, 0, 256) == 0 &&
           diagnostic_environment_value("256", 0, 256) == 256);
    for (const char* invalid : {"", "257", "-1", "12junk", " 1", "4294967296"}) {
        bool threw = false;
        try { (void)diagnostic_environment_value(invalid, 0, 256); }
        catch (const std::invalid_argument&) { threw = true; }
        expect(threw);
    }
    expect(!sample_dflash_diagnostic_round(0, 0, 0, 1) &&
           sample_dflash_diagnostic_round(0, 0, 4, 16) &&
           !sample_dflash_diagnostic_round(1, 0, 4, 16) &&
           sample_dflash_diagnostic_round(16, 1, 4, 16) &&
           !sample_dflash_diagnostic_round(64, 4, 4, 16));
    for (std::uint32_t k = 1; k <= 15; ++k) {
        for (std::uint32_t a = 0; a <= k; ++a) {
            for (std::uint32_t n = 0; n <= a + 1; ++n) {
                SpeculativeStats stats;
                stats.draft_window = k;
                stats.accepted_per_position.assign(k, 0);
                stats.attempted_per_position.assign(k, 0);
                stats.reached_per_position.assign(k, 0);
                stats.rejected_per_position.assign(k, 0);
                record_speculative_verification(stats, k, a);
                stats.diagnostic_samples.push_back(SpeculativeDiagnosticSample{
                    .round_index = 0, .lane = 2, .frontier = 40000, .licensed_tokens = a + 1});
                record_speculative_publication(stats, a + 1, n);
                const auto& sample = stats.diagnostic_samples.back();
                expect(sample.publication_recorded && sample.published_tokens == n &&
                       sample.licensed_tokens == a + 1 && sample.lane == 2);
            }
        }
    }
    std::cout << "proposal diagnostic Host oracle failures=" << failures << '\n';
    return failures != 0;
}
