// Reuses the independent full-vocabulary FP64/reference RNG oracle fixtures and
// the original acceptance suite, then compares the optional observer path exactly.
#define main original_speculative_round_test_main
#include "test_speculative_round.cpp"
#undef main

int main(int argc, char** argv) {
    if (cuda_unavailable()) { return 77; }
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--full")) {
        std::cerr << "usage: ninfer_speculative_proposal_diagnostics_test [--full]\n";
        return 2;
    }
    test_proposal_diagnostics = true;
    int failures = 0;
    std::vector<int> counts{1, 3, 5, 7, 15};
    if (argc == 2) {
        counts.clear();
        for (int k = 1; k <= 15; ++k) counts.push_back(k);
    }
    for (int k : counts) {
        for (int batch : {1, 2, 3, 8}) {
            SparseAcceptSuite suite(k, batch);
            failures += suite.sparse_greedy_direct_case();
            failures += suite.sparse_greedy_direct_case(0, true);
            failures += suite.generated_general_case();
        }
        failures += SparseAcceptSuite(k, 8).sparse_general_mixed_case();
        failures += SparseAcceptSuite(k, 1).repeated_history_case(false);
        failures += SparseAcceptSuite(k, 1).repeated_history_case(true);
    }
    std::cout << "optional proposal diagnostics GPU oracle failures=" << failures << '\n';
    return failures != 0;
}
