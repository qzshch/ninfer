#include "models/qwen3_5/program/retrieval/diagnostics.h"

#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::models::qwen3_5::detail;
    KvmemDiagnostics request_a, request_b;
    const KvmemPlacementStats stale_copy{
        .calls = 1, .demoted_pages = 4, .promoted_pages = 2, .d2h_pages = 3,
        .d2h_bytes = 3072, .h2d_bytes = 2048, .d2h_submit_wait_ns = 40,
        .h2d_submit_wait_ns = 30, .publication_wait_ns = 10, .total_host_wall_ns = 100};
    const KvmemPlacementStats current_replica{
        .calls = 1, .no_copy_calls = 1, .demoted_pages = 1,
        .publication_wait_ns = 5, .total_host_wall_ns = 25};
    auto& decode_main = request_a.placement[static_cast<std::size_t>(KvmemPlacementPhase::Decode)][0];
    add_kvmem_placement(decode_main, stale_copy);
    add_kvmem_placement(decode_main, current_replica);
    add_kvmem_placement(request_a.placement[static_cast<std::size_t>(KvmemPlacementPhase::Replay)][1],
                       current_replica);
    if (decode_main.calls != 2 || decode_main.no_copy_calls != 1 ||
        decode_main.demoted_pages != 5 || decode_main.d2h_pages != 3 ||
        decode_main.promoted_pages != 2 || decode_main.d2h_bytes != 3072 ||
        decode_main.h2d_bytes != 2048 || decode_main.d2h_submit_wait_ns != 40 ||
        decode_main.h2d_submit_wait_ns != 30 || decode_main.publication_wait_ns != 15 ||
        decode_main.total_host_wall_ns != 125 || request_a.placement[3][1].calls != 0 ||
        request_a.placement[2][1].calls != 1 || request_b.placement[3][0].calls != 0) {
        std::cerr << "KVMem aggregation conflated transfers, phases, pools, or requests\n";
        return 1;
    }
    {
        KvmemReplayStepTimer timer(&request_a);
    }
    {
        KvmemReplayStepTimer timer(nullptr);
    }
    if (request_a.replay_units != 1 || request_b.replay_units != 0) {
        std::cerr << "KVMem replay step ownership failed\n";
        return 1;
    }
    std::cout << "KVMem Host diagnostics preserve phase/pool/request ownership\n";
}
