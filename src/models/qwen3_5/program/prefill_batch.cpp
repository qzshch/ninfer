#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/retrieval/block_retrieval.h"
#include "core/device.h"
#include <algorithm>
#include <chrono>
#include <memory>
#include <vector>

namespace ninfer::models::qwen3_5::detail {
using Clock = std::chrono::steady_clock;

std::optional<PrefillBatchProgress>
ProgramImpl::advance_prefill_batch(std::span<const SequenceHandle> sequences,
                                  std::uint32_t token_budget,
                                  runtime::ExecutionTiming* failed_timing) {
    // Finalization, media, replay and capture retain their existing transaction units.
    if (sequences.size() > kMaximumConcurrency)
        throw std::invalid_argument("packed prefill exceeds startup concurrency");
    if (sequences.size() < 2 || token_budget < 128 || pending_transaction_) return std::nullopt;
    const auto aggregate_budget = std::min(token_budget, prefill_chunk);
    const auto per_row = aggregate_budget / static_cast<std::uint32_t>(sequences.size());
    if (per_row < 64) return std::nullopt;
    std::vector<std::uint32_t> lanes, lengths;
    for (const auto& handle : sequences) {
        if (!valid_sequence(handle)) throw std::logic_error("invalid packed prefill capability");
        const auto lane = ContractAccess::lane(handle).value;
        const auto& request = requests[lane];
        if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) return std::nullopt;
        const auto& staged = *request.prefill;
        const auto& sparse = kvmem_lanes_[lane];
        if (staged.pending_capture_offer || staged.query_replay_cursor ||
            staged.prompt.has_media() || staged.mtp_bridge != MtpBridgeMode::None ||
            staged.cursor >= staged.prompt_tokens ||
            staged.prompt_tokens - staged.cursor <= 64) return std::nullopt;
        const auto length = per_row;
        if (staged.prompt_tokens - staged.cursor <= length) return std::nullopt;
        bool preserve_scalar_unit = true;
        auto preserve_boundary = [&](std::uint32_t boundary) {
            if (boundary > staged.cursor && boundary - staged.cursor <= length)
                preserve_scalar_unit = false;
        };
        if (staged.next_capture < staged.capture_groups.size()) {
            if (staged.capture_groups[staged.next_capture].frontier == staged.cursor) return std::nullopt;
            preserve_boundary(staged.capture_groups[staged.next_capture].frontier);
        }
        const auto& frontiers = staged.prompt.identity.rewrite_execution_frontiers;
        const auto rewrite = std::upper_bound(frontiers.begin(), frontiers.end(), staged.cursor);
        if (rewrite != frontiers.end()) preserve_boundary(*rewrite);
        if (kvmem_window_pages && !sparse.query_checkpoint_valid &&
            staged.prompt_tokens > kvmem_window_pages * kPagedKVPageSize) {
            preserve_boundary(sparse.query_begin);
        }
        if (!preserve_scalar_unit || length < 64) return std::nullopt;
        if (std::find(lanes.begin(), lanes.end(), lane) != lanes.end())
            throw std::invalid_argument("duplicate packed prefill lane");
        lanes.push_back(lane); lengths.push_back(length);
    }
    std::vector<execution::PrefillContext> contexts;
    std::vector<std::unique_ptr<execution::TextContext>> cards;
    std::vector<execution::DFlashFeatureSink> sinks;
    std::vector<Tensor> feature_views, position_views;
    std::vector<execution::PackedPrefillSegment> segments;
    std::vector<qwen3_5::DFlashPrefillIngress> draft_ingresses(lanes.size());
    contexts.reserve(lanes.size()); cards.reserve(lanes.size()); sinks.reserve(lanes.size());
    feature_views.reserve(lanes.size()); position_views.reserve(lanes.size()); segments.reserve(lanes.size());
    const auto started = Clock::now();
    try {
        std::uint32_t offset = 0;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            auto& sequence = active_sequence(lanes[row]);
            auto& staged = *requests[lanes[row]].prefill;
            auto& sparse = kvmem_lanes_[lanes[row]];
            const auto selectors = state_selectors(sequence);
            bind_sequence_kv(sequence);
            contexts.push_back(execution::PrefillContext{
                {device, parameters, work, state_images->linear(),
                 replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
                 proposal_head, kvmem_window_pages ? static_cast<float*>(sparse.query_sum.data) : nullptr,
                 kvmem_window_pages ? static_cast<float*>(sparse.key_sums.data) : nullptr,
                 kvmem_capture_slots_, sparse.query_begin, sparse.query_end},
                text_kv_view(sequence), mtp_kv_view(sequence), decoder->text_kv, decoder->mtp_cache(),
                dflash ? &*dflash : nullptr, staged.cursor, nullptr, nullptr, selectors.source,
                selectors.destination, staged.initial_mtp_extent,
                sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0,
                dflash ? &draft_ingresses[row] : nullptr});
            auto& context = contexts.back();
            cards.push_back(std::make_unique<execution::TextContext>(
                device, parameters, work, context.text_kv, state_images->linear(), io, prefill_hidden,
                prefill_chunk, staged.cursor, context.mtp_kv, &decoder->text_kv, decoder->mtp_cache()));
            execution::configure_text_card(*cards.back(), context.execution, nullptr,
                                            selectors.source, selectors.destination, staged.initial_mtp_extent);
            execution::DFlashFeatureSink* sink = nullptr;
            if (dflash) {
                sinks.push_back(execution::dflash_feature_sink(context,
                    [&context](const Tensor& features, const Tensor& positions, bool rewrite) {
                        if (rewrite) throw std::logic_error("packed prefill crossed a rewrite");
                        auto& frame = *context.execution.io.dflash_prefill;
                        *context.dflash_prefill_host_ingress = {
                            .append_count = features.ne[1], .state_destination_slot = context.state_destination_slot,
                            .full_kv_table_row = context.dflash_kv_table_row};
                        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, context.dflash_prefill_host_ingress,
                            sizeof(qwen3_5::DFlashPrefillIngress), cudaMemcpyHostToDevice, context.execution.device.stream));
                        const auto count = static_cast<std::uint32_t>(features.ne[1]);
                        execution::dflash_append_context(context, features, positions, frame.append_count,
                            frame.state_destination_slot, frame.full_kv_table_row, {count, count});
                    }));
                feature_views.push_back(dflash->prefill_features.slice(1, offset, lengths[row]));
                position_views.push_back(dflash->prefill_positions.slice(0, offset, lengths[row]));
                sink = &sinks.back(); sink->features = &feature_views.back(); sink->positions = &position_views.back();
            }
            segments.push_back(execution::PackedPrefillSegment{
                .context = cards.back().get(), .prompt = staged.prompt.token_ids, .tokens = lengths[row],
                .kv_table_row = text_kv_addresses->bound_row(sequence.kv->text), .rope_delta = sequence.rope_delta,
                .backend_kv_table_row = context.dflash_kv_table_row, .sink = sink});
            offset += lengths[row];
        }
        mark_workspace_usage(workspace_plan.text_prefill);
        if (speculative_backend == SpeculativeBackend::Mtp) mark_workspace_usage(workspace_plan.mtp_prefill);
        if (dflash) mark_workspace_usage(workspace_plan.dflash_context);
        auto executed = cards.front()->prefill_packed(segments);
        const auto shared_timing = executed.front().timing;
        executed.front().timing = {};
        if (failed_timing) *failed_timing += shared_timing;
        const auto shared_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        PrefillBatchProgress out; out.timing = shared_timing;
        out.lanes = lanes; out.rows.reserve(lanes.size());
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            auto& request = requests[lanes[row]];
            request.prefill->elapsed_seconds += shared_seconds; // Exposure, not additive GPU work.
            auto step = advance_prefill(active_sequence(lanes[row]), request, failed_timing, lengths[row], &executed[row]);
            if (step.complete || step.processed_prompt_tokens != lengths[row])
                throw std::logic_error("packed prefill unexpectedly finalized or lost a row");
            out.timing += step.timing;
            auto progress = wrap_prefill(lanes[row], std::move(step)); progress.work_tokens = lengths[row];
            out.work_tokens += lengths[row]; out.rows.push_back(std::move(progress));
        }
        return out;
    } catch (...) {
        device.synchronize(); clear_execution_failure_lanes(lanes); throw;
    }
}
} // namespace ninfer::models::qwen3_5::detail
