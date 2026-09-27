#include "runtime/engine/model_instance.h"

#include <iostream>
#include <stdexcept>

int main() {
    ninfer::EngineOptions valid;
    valid.max_context = 16384;
    valid.kvmem_window_pages = 64;
    (void)ninfer::runtime::normalize_engine_options(valid);
    const auto rejects = [](ninfer::EngineOptions options) {
        try {
            (void)ninfer::runtime::normalize_engine_options(options);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    auto bad = valid;
    bad.max_concurrency = 2;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.kvmem_window_pages = 1;
    if (!rejects(bad)) { return 1; }
    bad.kvmem_window_pages = 0xffffffffU;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.speculative.backend = ninfer::SpeculativeBackend::DFlash;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.purpose = ninfer::EnginePurpose::CausalScoring;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.context_cache.enabled = false;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.context_cache.host_kv_capacity_bytes = 0;
    if (!rejects(bad)) { return 1; }
    bad = valid;
    bad.enable_vision = true;
    if (!rejects(bad)) { return 1; }
    // Capture storage follows configured chunk geometry, not an implicit 1024 cap.
    valid.prefill_chunk = 2048;
    (void)ninfer::runtime::normalize_engine_options(valid);
    std::cout << "ok\n";
}
