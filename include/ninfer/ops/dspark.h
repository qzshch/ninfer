#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"

namespace ninfer::ops {

// DSpark anchor-block sliding attention: D=256, Q=20, KV=4, window=2048.
// Committed context is [max(0,anchor-window),anchor), identical for all rows.
// Temporary keys in the same anchor block are causal (key_index<=query_index).
// q/out [256,20,T,B], temporary K/V [256,4,T,B] are BF16.
// Cached K/V are BF16/FP16; positions [T,B], valid [B], lanes [B] are I32.
// T=1..8, B=1..8. Inactive query columns produce exact zero. Context is immutable.
void dspark_sliding_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& lanes,
                              const CyclicKVCacheLayerView& context, Tensor& out,
                              cudaStream_t stream);

// Append the represented BF16 K and BF16->FP16 V for counts[b] sequential
// positions into lane[b]'s 2048-slot ring. Counts are in [0,T], T<=2048;
// uncommitted columns and every other cache slot remain unchanged.
void dspark_context_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                           const Tensor& counts, const Tensor& lanes,
                           CyclicKVCacheLayerView context, cudaStream_t stream);

// Greedy vanilla Markov proposals, with sequential predecessor conditioning.
// logits are BF16 [V,K,B]; W1/W2 are BF16 [256,V] codebooks. Each Markov
// dot product is accumulated in FP32, rounded to BF16, added to the BF16
// backbone logit, rounded to BF16, then argmaxed over [0,public_tokens).
// Position zero conditions on anchors[b], subsequent positions on drafts[k-1,b].
// Ties choose the lowest vocabulary ID. anchors [B], drafts [K,B] are I32;
// K=1..7, B=1..8. Inputs are read-only and mutually nonoverlapping with output.
void dspark_markov_greedy(const Tensor& logits, const Tensor& w1, const Tensor& w2,
                          const Tensor& anchors, std::int32_t public_tokens,
                          WorkspaceArena& workspace, Tensor& drafts, cudaStream_t stream);

[[nodiscard]] std::size_t dspark_markov_workspace_capacity_bytes(std::int32_t vocabulary,
                                                                 std::int32_t batch);

} // namespace ninfer::ops
