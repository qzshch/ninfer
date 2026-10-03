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
    PrefillBudget legacy(0);
    require(legacy.allowance(true, 1024) == 1024, "disabled budget preserves legacy unit bound");
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
