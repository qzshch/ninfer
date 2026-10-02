#include "ninfer/ops/speculative_round.h"
#include "ops/launcher/dflash_support_frontier.h"
#include <cstdint>
#include <stdexcept>
#include <string>
namespace ninfer::ops {
namespace {
constexpr std::int32_t kSparseMaxDrafts = 15;
constexpr std::int32_t kSparseMaxBatch = 8;
constexpr std::int32_t kSparseCandidates = 16;
void require_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}
void require_dtype(const Tensor& t, DType dtype, const char* op, const char* name) {
    if (t.dtype != dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid dtype for " + name);
    }
    require_contiguous_nonnull(t, op, name);
}
void require_matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                    const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (rows <= 0 || cols <= 0 || t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 ||
        t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid matrix shape for " + name);
    }
}
void require_tensor3(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1,
                     std::int32_t n2, const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (n0 <= 0 || n1 <= 0 || n2 <= 0 || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 ||
        t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}
} // namespace
void speculative_collect_support_frontier(
    const Tensor& logits, const Tensor& verify_ids, const Tensor& drafts, const Tensor& candidate_ids,
    const Tensor& proposal_q, std::int32_t token_domain, const SamplingConfig* configs,
    const Tensor& packets, Tensor& support_frontiers,
    SpeculativeSupportFrontierOptions options, cudaStream_t stream) {
    if (options.mode == SpeculativeSupportFrontierMode::Disabled) return;
    if (options.mode != SpeculativeSupportFrontierMode::ReadRanks)
        throw std::invalid_argument("speculative support frontier: invalid mode");
    constexpr const char* op = "speculative_collect_support_frontier";
    const int k = drafts.ne[0], batch = drafts.ne[1];
    if (k < 1 || k > kSparseMaxDrafts || batch < 1 || batch > kSparseMaxBatch ||
        token_domain <= 0 || token_domain > logits.ne[0] || configs == nullptr)
        throw std::invalid_argument("speculative support frontier: invalid domain/shape/config");
    require_matrix(verify_ids, DType::I32, k + 1, batch, op, "verify_ids");
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_tensor3(logits, DType::BF16, logits.ne[0], k + 1, batch, op, "logits");
    require_tensor3(candidate_ids, DType::I32, kSparseCandidates, k, batch, op, "candidate_ids");
    require_tensor3(proposal_q, DType::FP32, kSparseCandidates, k, batch, op, "proposal_q");
    require_tensor3(packets, DType::I32, 16, 2, batch, op, "packets");
    require_tensor3(support_frontiers, DType::I32, 16, 2, batch, op, "support_frontiers");
    // Read-only inputs must never overlap the output observer buffer.
    for (const Tensor* input : {&logits, &verify_ids, &drafts, &candidate_ids, &proposal_q, &packets}) {
        const auto begin = reinterpret_cast<std::uintptr_t>(input->data);
        const auto end = begin + input->bytes();
        const auto output_begin = reinterpret_cast<std::uintptr_t>(support_frontiers.data);
        const auto output_end = output_begin + support_frontiers.bytes();
        if (begin < output_end && output_begin < end)
            throw std::invalid_argument("speculative support frontier: output aliases input");
    }
    detail::speculative_collect_support_frontier_launch(
        logits, verify_ids, drafts, candidate_ids, proposal_q, token_domain, configs,
        packets, support_frontiers, stream);
}
} // namespace ninfer::ops
