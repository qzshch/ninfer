// Pure Host diagnostic contract, no CUDA initialization or model weights.
#include "runtime/engine/sparse_epoch_binding.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
void check(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
}
int main() {
    using namespace ninfer::runtime::detail;
    check(!sparse_snapshot_refresh_due(false, false, false, 60000000), "ordinary60ms remains throttled");
    check(sparse_snapshot_refresh_due(true, false, false, 60000000), "terminal60ms forces final counter");
    check(sparse_snapshot_refresh_due(false, true, false, 1), "new epoch forces counter reset snapshot");
    check(sparse_snapshot_refresh_due(false, false, true, 0), "first observation sampled");
    check(!sparse_snapshot_refresh_due(false, false, false, 999999999), "1s lower boundary");
    check(sparse_snapshot_refresh_due(false, false, false, 1000000000), "1s exact boundary");
    check(!sparse_terminal_refresh_due(false, 3, 2), "different Engine owner cannot finalize lane");
    check(!sparse_terminal_refresh_due(true, 0, 0), "zero epoch is not a request");
    check(!sparse_terminal_refresh_due(true, 3, 3), "finish fallback must not double sample");
    check(sparse_terminal_refresh_due(true, 4, 3), "reused Context new request must sample");
    // Actual regression shape: whole-batch copying preserves active other-lane
    // observations, and request2 starts at0 even though Context lifetime bytes grow.
    std::array<std::uint64_t,2> program{0,100},cached{0,80};
    SparseEpochBinding binding{};
    check(observe_sparse_epoch(binding,3,103), "first ownership bind");
    program[0]=589824;
    if (sparse_snapshot_refresh_due(true,false,false,60000000)) cached=program;
    check(cached[0]==589824 && cached[1]==100, "final snapshot copies whole Host batch");
    check(observe_sparse_epoch(binding,4,104), "next request epoch/owner");
    program[0]=0;
    if (sparse_snapshot_refresh_due(false,true,false,1)) cached=program;
    check(cached[0]==0 && binding.engine_request_id==104, "retained Context does not inherit request counter");
    check(!observe_sparse_epoch(binding,4,105) && binding.engine_request_id==104,
          "materializing next Engine ID cannot steal old epoch");
    std::puts("sparse terminal refresh Host contract passed");
    return 0;
}
