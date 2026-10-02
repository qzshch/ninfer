#include "ninfer/engine.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage out;
    out.role = role;
    out.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(text)});
    return out;
}

ninfer::PromptInput prompt(std::string question, bool tool_tail = false) {
    ninfer::PromptInput out;
    out.options.enable_thinking = false;
    std::string history = "Use the user's instruction. Reference notes:\n";
    for (int i = 0; i < 1100; ++i) { history += "alpha beta gamma delta "; }
    out.messages.push_back(message(ninfer::ChatRole::System, std::move(history)));
    out.messages.push_back(message(ninfer::ChatRole::User, std::move(question)));
    // Protocol automatic writes are rounded to a complete pre-query chunk.
    // Storage-level tests cover partial-page/block snapshot ownership separately.
    out.context_cache.markers.push_back({.after_message_count = 2,
        .kind = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::DefaultAutomatic});
    if (tool_tail) {
        auto assistant = message(ninfer::ChatRole::Assistant, "");
        assistant.tool_calls.push_back({.id = "call_reference", .name = "reference",
                                       .arguments_json = "{}"});
        out.messages.push_back(std::move(assistant));
        auto tool = message(ninfer::ChatRole::Tool,
            "Reference result: " + std::string(8000, 'x'));
        tool.tool_call_id = "call_reference";
        out.messages.push_back(std::move(tool));
        out.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"reference","parameters":{"type":"object","properties":{}}}})");
    }
    return out;
}

ninfer::RequestOptions request(bool reuse = true, std::uint32_t output_tokens = 24) {
    ninfer::RequestOptions out;
    out.execution.requested_output_tokens = output_tokens;
    out.execution.sampling.temperature = 0.0F;
    out.execution.sampling.seed = 42;
    out.execution.allow_prefix_reuse = reuse;
    out.stop.include_model_defaults = false;
    return out;
}
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) { std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n"; return 77; }
    try {
        ninfer::EngineOptions options;
        options.artifact_path = artifact;
        options.max_context = 8192;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(8192);
        options.max_concurrency = 2;
        options.prefill_chunk = 256;
        options.kvmem_window_pages = 32;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        if (const char* spec = std::getenv("NINFER_TEST_SPEC")) {
            if (std::string(spec) == "mtp") {
                options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
                options.speculative.draft_tokens = 3;
            } else if (std::string(spec) == "none") {
                options.speculative.backend = ninfer::SpeculativeBackend::None;
                options.speculative.draft_tokens = 0;
                options.speculative.proposal_head = ninfer::ProposalHead::Full;
            } else {
                throw std::invalid_argument("NINFER_TEST_SPEC must be mtp or none");
            }
        }
        options.context_cache.device_state_slots = 2;
        options.context_cache.host_state_slots = 4;
        options.context_cache.host_kv_capacity_bytes = 1ULL << 30;
        ninfer::Engine engine(std::move(options));
        if (std::getenv("NINFER_TEST_COLD_CONTROL") != nullptr) {
            const auto alone = engine.generate(engine.prepare(prompt("Print READY.")),
                                                request(false, 96));
            for (int round = 0; round < 3; ++round) {
                auto a = engine.submit(engine.prepare(prompt("Print READY.")), request(false, 96));
                auto b = engine.submit(engine.prepare(prompt("Print READY.")), request(false, 96));
                const auto ar = a.wait();
                const auto br = b.wait();
                std::cout << "cold pair " << round << " same-as-single="
                          << (ar.generated_token_ids == alone.generated_token_ids) << ','
                          << (br.generated_token_ids == alone.generated_token_ids) << ':';
                for (const auto token : ar.generated_token_ids) { std::cout << ' ' << token; }
                std::cout << " |";
                for (const auto token : br.generated_token_ids) { std::cout << ' ' << token; }
                std::cout << '\n';
            }
            return 0;
        }
        std::cerr << "phase cold long\n";
        const auto cold = engine.generate(engine.prepare(prompt("Print READY.")), request());
        std::cerr << "phase warm long\n";
        const auto warm = engine.generate(engine.prepare(prompt("Print READY.")), request());
        require(cold.prompt.prompt_tokens > 2048, "fixture is not longer than its window");
        require(warm.reused_prompt_tokens > 2048, "Host-backed long prefix was not reused");
        require(warm.generated_token_ids == cold.generated_token_ids,
                "warm checkpoint changed deterministic target output");
        // Compare prefill's target token before batched speculative decoding can
        // introduce the existing batch-shape-dependent floating-point differences.
        // Full single-lane output and tool-tail replay remain exact checks above/below.
        std::cerr << "phase cached parallel\n";
        auto first = engine.submit(engine.prepare(prompt("Print READY.")), request(true, 1));
        auto second = engine.submit(engine.prepare(prompt("Print READY.")), request(true, 1));
        const auto cached_first = first.wait();
        const auto cached_second = second.wait();
        std::cerr << "phase cold parallel\n";
        auto root_first = engine.submit(engine.prepare(prompt("Print READY.")), request(false, 1));
        auto root_second = engine.submit(engine.prepare(prompt("Print READY.")), request(false, 1));
        const auto uncached_first = root_first.wait();
        const auto uncached_second = root_second.wait();
        const auto show = [](const char* label, const auto& result) {
            std::cerr << label << " reused=" << result.reused_prompt_tokens << " tokens:";
            for (const auto token : result.generated_token_ids) { std::cerr << ' ' << token; }
            std::cerr << '\n';
        };
        show("single", cold);
        show("cached0", cached_first);
        show("cached1", cached_second);
        show("root0", uncached_first);
        show("root1", uncached_second);
        require(cached_first.generated_token_ids == uncached_first.generated_token_ids,
                "first cached lane disagrees with two-lane cold control");
        require(cached_second.generated_token_ids == uncached_second.generated_token_ids,
                "second cached lane disagrees with two-lane cold control");
        std::cerr << "phase cold tool replay\n";
        const auto replay_cold = engine.generate(
            engine.prepare(prompt("Print COMPLETE.", true)), request());
        require(replay_cold.reused_prompt_tokens == 0, "new tool schema reused an incompatible prefix");
        std::cerr << "phase warm tool replay\n";
        const auto replay_warm = engine.generate(
            engine.prepare(prompt("Print COMPLETE.", true)), request());
        require(replay_warm.reused_prompt_tokens > 2048,
                "tool-tail replay lost its safe pre-query prefix");
        require(replay_cold.generated_token_ids == replay_warm.generated_token_ids,
                "cached activation changed long tool-tail replay output");
        // Destroy a live cold-prefill owner while another lane remains in flight.
        std::cerr << "phase cancellation\n";
        auto survivor = engine.submit(engine.prepare(prompt("Print SURVIVE.")), request(false));
        {
            auto abandoned = engine.submit(engine.prepare(prompt("Print CANCELLED.")), request(false));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (engine.runtime_stats().prefilling_requests < 2 &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            require(engine.runtime_stats().prefilling_requests == 2,
                    "two admitted requests never owned prefill concurrently");
        }
        require(survivor.wait().generated_token_ids.size() == 24,
                "cancelling one prefill broke the other lane");
        const auto after = engine.generate(engine.prepare(prompt("Print READY.")), request());
        require(after.generated_token_ids == cold.generated_token_ids,
                "cancellation polluted a retained shared checkpoint");
        // Root checkpoints can also leave a deferred GDN fork. Admission must
        // wait for that fork to settle instead of invalidating the next request.
        const auto short_prompt = [](int id) {
            ninfer::PromptInput out;
            out.options.enable_thinking = false;
            out.messages.push_back(message(ninfer::ChatRole::User,
                "Run identifier " + std::to_string(id) +
                "; ignore it. Print consecutive integers starting at 1. Continue to 10000."));
            out.context_cache.markers.push_back({.after_message_count = 1,
                .kind = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence = ninfer::SharedCandidateEvidence::DefaultAutomatic});
            return out;
        };
        std::cerr << "phase short root forks\n";
        require(engine.generate(engine.prepare(short_prompt(0)), request(true, 128))
                    .generated_token_ids.size() == 128, "short retained warmup failed");
        for (int round = 0; round < 2; ++round) {
            auto a = engine.submit(engine.prepare(short_prompt(1 + round * 2)), request(true, 128));
            auto b = engine.submit(engine.prepare(short_prompt(2 + round * 2)), request(true, 128));
            require(a.wait().generated_token_ids.size() == 128,
                    "root checkpoint fork invalidated first concurrent admission");
            require(b.wait().generated_token_ids.size() == 128,
                    "root checkpoint fork invalidated second concurrent admission");
        }
        std::cout << "ok long-prefix=" << warm.reused_prompt_tokens
                  << " prompt=" << warm.prompt.prompt_tokens << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
    return 0;
}
