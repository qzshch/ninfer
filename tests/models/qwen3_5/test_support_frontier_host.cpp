#include "ninfer/dflash_support_frontier.h"
#include "models/qwen3_5/program/speculative/diagnostic_sampling.h"
#include <limits>
#include <iostream>
int main() {
    using ninfer::DFlashSupportStage;
    const auto classify = ninfer::classify_dflash_support_frontier;
    int failures = 0;
    failures += classify(true, 21, 20, false, 0, 0, .7f) != DFlashSupportStage::TopK;
    failures += classify(true, 2, 20, false, 0, 0, .7f) != DFlashSupportStage::TopP;
    failures += classify(true, 2, 20, false, .1f, 0, .7f) != DFlashSupportStage::AfterTopKUnresolved;
    failures += classify(true, 2, 20, true, 0, .2f, .7f) != DFlashSupportStage::PositiveProbability;
    failures += classify(false, 2, 1, false, 0, 0, 0) != DFlashSupportStage::Greedy;
    failures += classify(true, 2, 20, false, 0, 0, 0) != DFlashSupportStage::Invalid;
    failures += classify(true, 2, 20, false, 0, 0, std::numeric_limits<float>::quiet_NaN()) != DFlashSupportStage::Invalid;
    failures += classify(true, 0, 20, false, 0, 0, .7f) != DFlashSupportStage::Invalid;
    failures += classify(true, 2, 20, true, 0, 0, .7f) != DFlashSupportStage::ZeroProbability;
    failures += classify(true, 21, 20, true, 0, .1f, .7f) != DFlashSupportStage::Invalid;
    failures += classify(true, 2, 20, false, 0, .1f, .7f) != DFlashSupportStage::Invalid;
    std::vector<std::int32_t> ids(512, 7);
    failures += !ninfer::bounded_dflash_prompt_identity(false, false, ids).empty();
    failures += !ninfer::bounded_dflash_prompt_identity(true, true, ids).empty();
    failures += ninfer::bounded_dflash_prompt_identity(true, false, ids) != ids;
    ids.push_back(8);
    failures += !ninfer::bounded_dflash_prompt_identity(true, false, ids).empty();
    const auto parse = ninfer::models::qwen3_5::detail::support_frontier_environment;
    failures += parse(nullptr) || parse("0") || !parse("1");
    for (const char* invalid : {"", "2", "01", "+1", " 1", "1 "}) {
        try { (void)parse(invalid); ++failures; }
        catch (const std::invalid_argument&) {}
    }
    std::cout << "support frontier Host failures=" << failures << '\n';
    return failures != 0;
}
