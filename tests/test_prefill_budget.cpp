#include "runtime/engine/prefill_budget.h"
#include "models/qwen3_5/program/speculative/adaptive_draft_window.h"
#include <array>
#include <limits>
#include <iostream>
#include <stdexcept>

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    using ninfer::runtime::PrefillBudget;
    using ninfer::models::qwen3_5::confidence_draft_window;
    PrefillBudget budget(1024);
    require(budget.allowance(true, 1024) == 0, "decoder must prime the global budget");
    budget.decode_completed();
    require(budget.allowance(true, 512) == 512, "first owner grant");
    budget.consume(512);
    require(budget.allowance(true, 1024) == 512, "second owner shares the remaining global budget");
    budget.consume(512);
    require(budget.allowance(true, 1024) == 0,
            "three owners must not each spend the full global budget");
    budget.decode_completed();
    budget.consume(1536);
    require(budget.allowance(true, 1024) == 0,
            "atomic media overshoot must block additional prefill");
    budget.decode_completed();
    require(budget.allowance(true, 1024) == 512, "media debt must survive the next decode round");
    budget.consume(512);
    require(budget.allowance(false, 1024) == 1024, "cold-only work cannot deadlock without decode");
    PrefillBudget adaptive(1024);
    require(adaptive.allowance(false, 4096) == 4096,
            "a lone cold request uses the qualified workspace chunk, not the mixed-work cap");
    adaptive.consume(4096);
    require(adaptive.allowance(true, 4096) == 0,
            "the first decoder gets service before another prefill grant");
    adaptive.decode_completed();
    require(adaptive.allowance(true, 4096) == 1024,
            "cold work creates no artificial debt for a newly ready decoder");
    adaptive.consume(1536);
    adaptive.decode_completed();
    require(adaptive.allowance(true, 4096) == 512,
            "mixed atomic media debt still survives a decode round");
    require(adaptive.allowance(false, 4096, 2) == 2048 &&
                adaptive.allowance(false, 4096, 3) == 1365 &&
                adaptive.allowance(false, 4096, 4) == 1024,
            "cold owners fairly share the maximum chunk without allocating larger workspaces");

    for (std::uint32_t owners = 1; owners <= 8; ++owners) {
        require(adaptive.allowance(false, 1024, owners) == 1024,
                "cold lane rotation must not shrink a configured 1K execution unit");
    }
    require(adaptive.allowance(false, 128, 3) == 128,
            "cold minimum cannot exceed the physical workspace chunk");
    require(adaptive.allowance(false, 4096, 8) == 1024,
            "additional cold owners retain a useful unit below the workspace maximum");
    require(adaptive.allowance(false, 1024, 0) == 1024,
            "zero owner snapshot must retain progress for the selected owner");
    adaptive.consume(1365);
    adaptive.decode_completed();
    require(adaptive.allowance(true, 4096, 3) == 1024,
            "fair cold service does not weaken the shared mixed-work cap");
    PrefillBudget legacy(0);
    require(PrefillBudget::packed_allowance(false,1024,1024,3)==0,
            "grouping cannot shrink the qualified cold 1K row to 341 tokens");
    require(PrefillBudget::packed_allowance(false,4096,1365,3)==4095,
            "three unchanged scalar shapes fit one physical workspace");
    require(PrefillBudget::packed_allowance(false,4096,2048,2)==4096 &&
                PrefillBudget::packed_allowance(false,4096,2048,3)==0,
            "grouping preserves grants at the startup capacity boundary");
    require(PrefillBudget::packed_allowance(true,4096,1024,2)==0 &&
                PrefillBudget::packed_allowance(false,4096,1024,1)==0,
            "mixed decode service and one owner keep the scalar transaction");
    require(legacy.allowance(true, 1024) == 1024, "disabled budget preserves legacy unit bound");
    PrefillBudget timed(1024, 20, 512);
    timed.observe(1000, 100000000); // 100 us/token; 20ms with 25% margin permits 160.
    timed.decode_completed();
    require(timed.allowance(true, 4096) == 160, "time target caps mixed service");
    timed.observe(160, 16000000);
    timed.consume(160);
    require(timed.allowance(true, 4096) == 32, "time credit is shared across consecutive units");
    timed.observe(32, 5000000);
    timed.consume(32);
    require(timed.allowance(true, 4096) == 0, "time overshoot gives decode priority before token credit runs out");
    require(timed.allowance(false, 4096) == 512, "cold service uses explicit request cap");
    timed.observe(128, 128000000000ULL);
    timed.decode_completed();
    require(timed.allowance(true, 4096) == 1, "expensive work still advances one token");
    PrefillBudget untimed(1024);
    untimed.observe(1000, 100000000);
    untimed.decode_completed();
    require(untimed.allowance(true, 4096) == 1024, "disabled time target preserves token policy");
    PrefillBudget finalization_safe(1024, 20);
    finalization_safe.observe(1000, 100000000);
    finalization_safe.observe(1, 10000000);
    finalization_safe.decode_completed();
    require(finalization_safe.allowance(true, 4096) == 160,
            "tiny finalization cost cannot collapse the long-prefill grant");
    const std::array<float, 7> all_one{1, 1, 1, 1, 1, 1, 1}, low{.1f, .9f, .9f, .9f, .9f, .9f, .9f};
    const std::array<float, 7> mixed{.95f, .5f, .3f, .9f, .9f, .9f, .9f};
    require(confidence_draft_window(all_one) == 7, "confident block keeps maximum K");
    require(confidence_draft_window(low) == 1, "minimum draft keeps forward progress");
    require(confidence_draft_window(mixed) == 2, "survival cutoff must keep a contiguous prefix");
    auto invalid  = all_one;
    invalid[5]    = std::numeric_limits<float>::quiet_NaN();
    bool rejected = false;
    try {
        (void)confidence_draft_window(invalid);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "missing or invalid confidence must not silently become adaptive K");
    std::cout << "global prefill budget / media debt / adaptive contiguous K passed\n";
}
