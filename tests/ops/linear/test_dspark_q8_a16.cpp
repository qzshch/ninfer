#include "ops/linear/linear_test_common.h"
#include <exception>
#include <iostream>
#include <vector>
using namespace ninfer::test::linear;

int main() {
    if (!cuda_available()) {
        std::cerr << "CUDA required\n";
        return 1;
    }
    try {
        int failures = 0;
        for (int k : {5120, 40960}) {
            std::vector<Invocation> calls;
            for (int t : {1,  4,  5,  8,  9,  16, 17, 24,  25,  32,  33,  40,
                          41, 48, 49, 56, 57, 64, 65, 128, 129, 256, 1024})
                calls.push_back({t});
            for (int t : {1, 8, 24, 64, 129})
                calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
            failures +=
                run_shape("DSpark Q8", ActivationCompute::A16, make_q8_g32_fp16_weight,
                          {5120, k, 20261003U + unsigned(k), Comparison::Sampled, true, calls});
        }
        std::cout << "DSpark Q8 failures=" << failures << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
