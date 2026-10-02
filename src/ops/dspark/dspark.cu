#include "ninfer/ops/dspark.h"
#include "core/device.h"
#include "core/layout.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
using Bf16 = __nv_bfloat16;

void require(const Tensor& t, DType dtype, int a, int b, int c, int d, const char* label) {
    if (t.dtype != dtype || !t.data || !t.is_contiguous() || t.ne[0] != a || t.ne[1] != b ||
        t.ne[2] != c || t.ne[3] != d)
        throw std::invalid_argument(std::string("dspark: invalid ") + label);
}

__device__ float warp_sum(float x) {
    for (int step = 16; step; step >>= 1) x += __shfl_down_sync(0xffffffff, x, step);
    return __shfl_sync(0xffffffff, x, 0);
}

__global__ void attention(const Bf16* q, const Bf16* k, const Bf16* v, const int* positions,
                          const int* valid, const int* lanes, const Bf16* ck, const half* cv,
                          int pitch, int width, Bf16* out) {
    const int h = blockIdx.x, t = blockIdx.y, b = blockIdx.z;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qbase = 256 * (h + 20 * (t + width * b));
    if (t >= valid[b]) {
        for (int d = threadIdx.x; d < 256; d += blockDim.x) out[qbase + d] = __float2bfloat16(0.f);
        return;
    }
    const int start = positions[width * b];
    const int begin = max(0, start - 2048), count = start - begin;
    float query[8], accum[8]                      = {};
#pragma unroll
    for (int j = 0; j < 8; ++j) query[j] = __bfloat162float(q[qbase + lane + j * 32]);
    float m = -INFINITY, l = 0;
    for (int key = warp; key < count + t + 1; key += 8) {
        const bool cached = key < count;
        const int p       = cached ? begin + key : key - count;
        const int kb      = cached ? 256 * ((p % 2048) + pitch * (h / 5 + 4 * lanes[b]))
                                   : 256 * (h / 5 + 4 * (p + width * b));
        float dot         = 0, value[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int idx = kb + lane + 32 * j;
            dot += query[j] * __bfloat162float(cached ? ck[idx] : k[idx]);
            value[j] = cached ? __half2float(cv[idx]) : __bfloat162float(v[idx]);
        }
        const float score = warp_sum(dot) * 0.0625f;
        const float nm = fmaxf(m, score), a = expf(m - nm), z = expf(score - nm);
#pragma unroll
        for (int j = 0; j < 8; ++j) accum[j] = accum[j] * a + value[j] * z;
        l = l * a + z;
        m = nm;
    }
    __shared__ float acc[8][256], ms[8], ls[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[warp][lane + j * 32] = accum[j];
    if (lane == 0) {
        ms[warp] = m;
        ls[warp] = l;
    }
    __syncthreads();
    const int d   = threadIdx.x;
    float maximum = -INFINITY;
    for (int w = 0; w < 8; ++w) maximum = fmaxf(maximum, ms[w]);
    float numerator = 0, denominator = 0;
    for (int w = 0; w < 8; ++w) {
        const float z = ls[w] == 0 ? 0 : expf(ms[w] - maximum);
        numerator += z * acc[w][d];
        denominator += z * ls[w];
    }
    out[qbase + d] = __float2bfloat16(numerator / denominator);
}

__global__ void append(const Bf16* k, const Bf16* v, const int* positions, const int* counts,
                       const int* lanes, int width, int pitch, Bf16* ck, half* cv) {
    const int b = blockIdx.z, t = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
    if (t >= counts[b]) return;
    const int src = d + 256 * (h + 4 * (t + width * b));
    const int dst = d + 256 * ((positions[t + width * b] % 2048) + pitch * (h + 4 * lanes[b]));
    ck[dst]       = k[src];
    cv[dst]       = __float2half_rn(__bfloat162float(v[src]));
}

__device__ bool better(float x, int id, float y, int other) {
    return x > y || (x == y && id < other);
}

__global__ void markov_scores(const Bf16* logits, const Bf16* w1, const Bf16* w2,
                              const int* anchors, const int* drafts, int vocabulary,
                              int public_tokens, int steps, int step, int tiles, float* scores,
                              int* ids) {
    const int b = blockIdx.y, thread = threadIdx.x;
    const int prev = step == 0 ? anchors[b] : drafts[step - 1 + steps * b];
    __shared__ float predecessor[256], best[128];
    __shared__ int token[128];
    for (int j = thread; j < 256; j += 128) predecessor[j] = __bfloat162float(w1[j + 256 * prev]);
    __syncthreads();
    float maximum = -INFINITY;
    int chosen    = public_tokens;
    for (int id = blockIdx.x * 128 + thread; id < public_tokens; id += tiles * 128) {
        float bias = 0;
        for (int j = 0; j < 256; ++j) bias += predecessor[j] * __bfloat162float(w2[j + id * 256]);
        const float rounded = __bfloat162float(__float2bfloat16(bias));
        const float value   = __bfloat162float(__float2bfloat16(
            rounded + __bfloat162float(logits[id + vocabulary * (step + steps * b)])));
        if (better(value, id, maximum, chosen)) {
            maximum = value;
            chosen  = id;
        }
    }
    best[thread]  = maximum;
    token[thread] = chosen;
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
        if (thread < stride &&
            better(best[thread + stride], token[thread + stride], best[thread], token[thread])) {
            best[thread]  = best[thread + stride];
            token[thread] = token[thread + stride];
        }
        __syncthreads();
    }
    if (thread == 0) {
        scores[blockIdx.x + b * tiles] = best[0];
        ids[blockIdx.x + b * tiles]    = token[0];
    }
}

__global__ void choose(const float* scores, const int* ids, int tiles, int steps, int step,
                       int* drafts) {
    const int b = blockIdx.x;
    float best  = -INFINITY;
    int token   = INT_MAX;
    for (int i = 0; i < tiles; ++i)
        if (better(scores[i + b * tiles], ids[i + b * tiles], best, token)) {
            best  = scores[i + b * tiles];
            token = ids[i + b * tiles];
        }
    drafts[step + b * steps] = token;
}
} // namespace

void dspark_sliding_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid, const Tensor& lanes,
                              const CyclicKVCacheLayerView& context, Tensor& out,
                              cudaStream_t stream) {
    const int width = q.ne[2], batch = q.ne[3];
    if (width < 1 || width > 8 || batch < 1 || batch > 8 || context.capacity != 2048 ||
        context.padded_capacity < 2048 || context.head_dim != 256 || context.num_kv_heads != 4 ||
        context.lane_capacity < batch)
        throw std::invalid_argument("dspark: unsupported attention profile");
    require(q, DType::BF16, 256, 20, width, batch, "query");
    require(out, DType::BF16, 256, 20, width, batch, "output");
    require(k, DType::BF16, 256, 4, width, batch, "temporary key");
    require(v, DType::BF16, 256, 4, width, batch, "temporary value");
    require(positions, DType::I32, width, batch, 1, 1, "positions");
    require(valid, DType::I32, batch, 1, 1, 1, "valid columns");
    require(lanes, DType::I32, batch, 1, 1, 1, "lanes");
    require(context.k, DType::BF16, 256, context.padded_capacity, 4, context.lane_capacity,
            "cached key");
    require(context.v, DType::FP16, 256, context.padded_capacity, 4, context.lane_capacity,
            "cached value");
    attention<<<dim3(20, width, batch), 256, 0, stream>>>(
        static_cast<const Bf16*>(q.data), static_cast<const Bf16*>(k.data),
        static_cast<const Bf16*>(v.data), static_cast<const int*>(positions.data),
        static_cast<const int*>(valid.data), static_cast<const int*>(lanes.data),
        static_cast<const Bf16*>(context.k.data), static_cast<const half*>(context.v.data),
        context.padded_capacity, width, static_cast<Bf16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void dspark_context_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                           const Tensor& counts, const Tensor& lanes,
                           CyclicKVCacheLayerView context, cudaStream_t stream) {
    const int width = k.ne[2], batch = k.ne[3];
    if (width < 1 || width > 2048 || batch < 1 || batch > 8 || context.capacity != 2048 ||
        context.padded_capacity < 2048 || context.head_dim != 256 || context.num_kv_heads != 4 ||
        context.lane_capacity < batch)
        throw std::invalid_argument("dspark: unsupported append profile");
    require(k, DType::BF16, 256, 4, width, batch, "append key");
    require(v, DType::BF16, 256, 4, width, batch, "append value");
    require(positions, DType::I32, width, batch, 1, 1, "append positions");
    require(counts, DType::I32, batch, 1, 1, 1, "append counts");
    require(lanes, DType::I32, batch, 1, 1, 1, "append lanes");
    require(context.k, DType::BF16, 256, context.padded_capacity, 4, context.lane_capacity,
            "append cache key");
    require(context.v, DType::FP16, 256, context.padded_capacity, 4, context.lane_capacity,
            "append cache value");
    append<<<dim3(4, width, batch), 256, 0, stream>>>(
        static_cast<const Bf16*>(k.data), static_cast<const Bf16*>(v.data),
        static_cast<const int*>(positions.data), static_cast<const int*>(counts.data),
        static_cast<const int*>(lanes.data), width, context.padded_capacity,
        static_cast<Bf16*>(context.k.data), static_cast<half*>(context.v.data));
    CUDA_CHECK(cudaGetLastError());
}

std::size_t dspark_markov_workspace_capacity_bytes(int vocabulary, int batch) {
    if (vocabulary <= 0 || batch < 1 || batch > 8)
        throw std::invalid_argument("dspark: invalid Markov size");
    WorkspaceLayoutBuilder layout;
    const int tiles = 128;
    (void)layout.alloc(DType::FP32, {tiles, batch});
    (void)layout.alloc(DType::I32, {tiles, batch});
    return layout.peak_bytes();
}

void dspark_markov_greedy(const Tensor& logits, const Tensor& w1, const Tensor& w2,
                          const Tensor& anchors, int public_tokens, WorkspaceArena& workspace,
                          Tensor& drafts, cudaStream_t stream) {
    const int vocabulary = logits.ne[0], steps = logits.ne[1], batch = logits.ne[2];
    if (steps < 1 || steps > 7 || batch < 1 || batch > 8 || public_tokens < 1 ||
        public_tokens > vocabulary)
        throw std::invalid_argument("dspark: invalid Markov profile");
    require(logits, DType::BF16, vocabulary, steps, batch, 1, "logits");
    require(w1, DType::BF16, 256, vocabulary, 1, 1, "predecessor codebook");
    require(w2, DType::BF16, 256, vocabulary, 1, 1, "successor codebook");
    require(anchors, DType::I32, batch, 1, 1, 1, "anchors");
    require(drafts, DType::I32, steps, batch, 1, 1, "drafts");
    const int tiles = 128;
    auto scope      = workspace.scope();
    auto scores     = workspace.alloc(DType::FP32, {tiles, batch});
    auto ids        = workspace.alloc(DType::I32, {tiles, batch});
    for (int step = 0; step < steps; ++step) {
        markov_scores<<<dim3(tiles, batch), 128, 0, stream>>>(
            static_cast<const Bf16*>(logits.data), static_cast<const Bf16*>(w1.data),
            static_cast<const Bf16*>(w2.data), static_cast<const int*>(anchors.data),
            static_cast<const int*>(drafts.data), vocabulary, public_tokens, steps, step, tiles,
            static_cast<float*>(scores.data), static_cast<int*>(ids.data));
        choose<<<batch, 1, 0, stream>>>(static_cast<const float*>(scores.data),
                                        static_cast<const int*>(ids.data), tiles, steps, step,
                                        static_cast<int*>(drafts.data));
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ninfer::ops
