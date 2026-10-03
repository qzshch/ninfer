#include "ninfer/ops/dspark.h"
#include "core/device.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace ninfer;

namespace {
void check(bool ok, const char* label) {
    if (!ok) throw std::runtime_error(label);
}

std::uint16_t bf(float x) {
    auto u = std::bit_cast<std::uint32_t>(x);
    u += 0x7fff + ((u >> 16) & 1);
    return u >> 16;
}

float unbf(std::uint16_t x) { return std::bit_cast<float>(std::uint32_t(x) << 16); }

std::uint16_t hf(float x) { return std::bit_cast<std::uint16_t>(__float2half_rn(x)); }

float unhf(std::uint16_t x) { return __half2float(std::bit_cast<__half>(x)); }

std::uint32_t rng = 20261003;

float random_value() {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (int(rng % 65) - 32) / 64.f;
}

template <class T>
struct Buffer {
    void* p = nullptr;
    std::size_t n;

    explicit Buffer(const std::vector<T>& x) : n(x.size()) {
        CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(p, x.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }

    ~Buffer() { cudaFree(p); }

    std::vector<T> read() const {
        std::vector<T> x(n);
        CUDA_CHECK(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return x;
    }

    Tensor tensor(DType type, std::initializer_list<int> shape) { return Tensor(p, type, shape); }
};

std::vector<std::uint16_t> data(std::size_t n) {
    std::vector<std::uint16_t> x(n);
    for (auto& v : x) v = bf(random_value());
    return x;
}

void attention_case(int width, int batch, int frontier, bool graph) {
    const int slots = 8, pitch = 2048;
    auto q = data(256 * 20 * width * batch), k = data(256 * 4 * width * batch), v = data(k.size());
    auto ck = data(256 * 4 * pitch * slots), cv = ck;
    for (auto& x : cv) x = hf(unbf(x));
    auto initial = std::vector<std::uint16_t>(q.size(), 0x7fc1);
    std::vector<int> pos(width * batch), valid(batch), lanes(batch);
    for (int b = 0; b < batch; ++b) {
        valid[b] = std::max(1, width - b);
        lanes[b] = (b * 3 + 2) % slots;
        for (int t = 0; t < width; ++t) pos[t + width * b] = frontier + b * 3 + t;
    }
    Buffer dq(q), dk(k), dv(v), dck(ck), dcv(cv), dout(initial);
    Buffer dp(pos), dn(valid), dl(lanes);
    auto tq = dq.tensor(DType::BF16, {256, 20, width, batch});
    auto tk = dk.tensor(DType::BF16, {256, 4, width, batch});
    auto tv = dv.tensor(DType::BF16, {256, 4, width, batch});
    auto to = dout.tensor(DType::BF16, {256, 20, width, batch});
    auto tp = dp.tensor(DType::I32, {width, batch});
    auto tn = dn.tensor(DType::I32, {batch});
    auto tl = dl.tensor(DType::I32, {batch});
    CyclicKVCacheLayerView cache{dck.tensor(DType::BF16, {256, pitch, 4, slots}),
                                 dcv.tensor(DType::FP16, {256, pitch, 4, slots}),
                                 2048,
                                 2048,
                                 4,
                                 256,
                                 slots};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (graph) {
        cudaGraph_t g;
        cudaGraphExec_t e;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::dspark_sliding_attention(tq, tk, tv, tp, tn, tl, cache, to, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &g));
        CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
        for (int i = 0; i < 2; ++i) CUDA_CHECK(cudaGraphLaunch(e, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaGraphExecDestroy(e);
        cudaGraphDestroy(g);
    } else {
        ops::dspark_sliding_attention(tq, tk, tv, tp, tn, tl, cache, to, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto result = dout.read();
    double sq = 0, reference_sq = 0, max_error = 0;
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < width; ++t)
            for (int h = 0; h < 20; ++h) {
                const int qb = 256 * (h + 20 * (t + width * b));
                if (t >= valid[b]) {
                    for (int d = 0; d < 256; ++d)
                        check(result[qb + d] == 0, "inactive attention column changed");
                    continue;
                }
                int start = pos[b * width], begin = std::max(0, start - 2048),
                    count = start - begin;
                std::vector<double> scores(count + t + 1);
                double maximum = -INFINITY;
                for (int j = 0; j < int(scores.size()); ++j) {
                    int kb = j < count ? 256 * ((begin + j) % 2048 + pitch * (h / 5 + 4 * lanes[b]))
                                       : 256 * (h / 5 + 4 * (j - count + width * b));
                    double dot = 0;
                    for (int d = 0; d < 256; ++d)
                        dot += double(unbf(q[qb + d])) * unbf((j < count ? ck : k)[kb + d]);
                    maximum = std::max(maximum, scores[j] = dot / 16.);
                }
                double denominator = 0;
                for (auto& x : scores) {
                    x = std::exp(x - maximum);
                    denominator += x;
                }
                for (int d = 0; d < 256; ++d) {
                    double reference = 0;
                    for (int j = 0; j < int(scores.size()); ++j) {
                        int vb = j < count
                                     ? 256 * ((begin + j) % 2048 + pitch * (h / 5 + 4 * lanes[b]))
                                     : 256 * (h / 5 + 4 * (j - count + width * b));
                        reference += scores[j] * (j < count ? unhf(cv[vb + d]) : unbf(v[vb + d]));
                    }
                    reference /= denominator;
                    double error = unbf(result[qb + d]) - reference;
                    check(std::abs(error) <= 3e-4 + 4.5e-3 * std::abs(reference),
                          "attention differs from FP64 oracle");
                    sq += error * error;
                    reference_sq += reference * reference;
                    max_error = std::max(max_error, std::abs(error));
                }
            }
    check(std::sqrt(sq / std::max(reference_sq, 1e-20)) <= 2.3e-3,
          "attention aggregate error too large");
    check(dq.read() == q && dk.read() == k && dv.read() == v && dck.read() == ck &&
              dcv.read() == cv,
          "attention mutated input/cache");
    check(dp.read() == pos && dn.read() == valid && dl.read() == lanes,
          "attention mutated selectors");
    cudaStreamDestroy(stream);
    std::cout << "attention T=" << width << " B=" << batch << " L=" << frontier
              << " graph=" << graph << " max_error=" << max_error << '\n';
}

void append_case() {
    const int width = 8, batch = 3, slots = 8, pitch = 2048;
    auto k = data(256 * 4 * width * batch), v = data(k.size()), ck = data(256 * pitch * 4 * slots),
         cv = ck;
    for (auto& x : cv) x = hf(unbf(x));
    auto expected_k = ck, expected_v = cv;
    std::vector<int> pos(width * batch), counts{0, 5, 8}, lanes{7, 2, 0};
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < width; ++t) {
            pos[t + width * b] = 2045 + b + t;
            if (t >= counts[b]) continue;
            for (int h = 0; h < 4; ++h)
                for (int d = 0; d < 256; ++d) {
                    int src = d + 256 * (h + 4 * (t + width * b)),
                        dst = d + 256 * ((pos[t + width * b] % 2048) + pitch * (h + 4 * lanes[b]));
                    expected_k[dst] = k[src];
                    expected_v[dst] = hf(unbf(v[src]));
                }
        }
    Buffer dk(k), dv(v), dck(ck), dcv(cv);
    Buffer dp(pos), dc(counts), dl(lanes);
    auto tk = dk.tensor(DType::BF16, {256, 4, width, batch}),
         tv = dv.tensor(DType::BF16, {256, 4, width, batch});
    auto tp = dp.tensor(DType::I32, {width, batch}), tc = dc.tensor(DType::I32, {batch}),
         tl = dl.tensor(DType::I32, {batch});
    CyclicKVCacheLayerView cache{dck.tensor(DType::BF16, {256, pitch, 4, slots}),
                                 dcv.tensor(DType::FP16, {256, pitch, 4, slots}),
                                 2048,
                                 2048,
                                 4,
                                 256,
                                 slots};
    ops::dspark_context_append(tk, tv, tp, tc, tl, cache, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    check(dck.read() == expected_k && dcv.read() == expected_v, "append changed wrong ring bytes");
    check(dk.read() == k && dv.read() == v && dp.read() == pos && dc.read() == counts &&
              dl.read() == lanes,
          "append mutated input");
    std::cout << "append B3 ragged/zero/wrap/permuted lanes passed\n";
}

void markov_case(int vocabulary, int steps, int batch, bool ties, bool graph) {
    auto logits = data(std::size_t(vocabulary) * steps * batch),
         w1 = data(std::size_t(vocabulary) * 256), w2 = data(w1.size());
    if (ties) {
        std::fill(logits.begin(), logits.end(), 0);
        std::fill(w1.begin(), w1.end(), 0);
        std::fill(w2.begin(), w2.end(), 0);
    }
    int public_tokens = vocabulary - 3;
    std::vector<int> anchors(batch), out(steps * batch, -123), expected(out.size());
    for (int b = 0; b < batch; ++b) anchors[b] = (b * 31 + 7) % public_tokens;
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < steps; ++t) {
            int prev       = t == 0 ? anchors[b] : expected[t - 1 + steps * b];
            double maximum = -INFINITY;
            int chosen     = 0;
            for (int id = 0; id < public_tokens; ++id) {
                double bias = 0;
                for (int j = 0; j < 256; ++j)
                    bias += double(unbf(w1[j + 256 * prev])) * unbf(w2[j + 256 * id]);
                double score = unbf(
                    bf(unbf(bf(float(bias))) + unbf(logits[id + vocabulary * (t + steps * b)])));
                if (score > maximum) {
                    maximum = score;
                    chosen  = id;
                }
            }
            expected[t + steps * b] = chosen;
        }
    Buffer dl(logits), d1(w1), d2(w2);
    Buffer da(anchors), dd(out);
    auto tl = dl.tensor(DType::BF16, {vocabulary, steps, batch}),
         t1 = d1.tensor(DType::BF16, {256, vocabulary}),
         t2 = d2.tensor(DType::BF16, {256, vocabulary});
    auto ta = da.tensor(DType::I32, {batch}), td = dd.tensor(DType::I32, {steps, batch});
    WorkspaceArena workspace(ops::dspark_markov_workspace_capacity_bytes(vocabulary, batch));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (graph) {
        cudaGraph_t g;
        cudaGraphExec_t e;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::dspark_markov_greedy(tl, t1, t2, ta, public_tokens, workspace, td, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &g));
        CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(e, stream));
        CUDA_CHECK(cudaGraphLaunch(e, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaGraphExecDestroy(e);
        cudaGraphDestroy(g);
    } else {
        ops::dspark_markov_greedy(tl, t1, t2, ta, public_tokens, workspace, td, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    check(dd.read() == expected, "Markov sequential token IDs differ from FP64/BF16 oracle");
    check(dl.read() == logits && d1.read() == w1 && d2.read() == w2 && da.read() == anchors,
          "Markov mutated inputs");
    cudaStreamDestroy(stream);
    std::cout << "Markov V=" << vocabulary << " K=" << steps << " B=" << batch << " ties=" << ties
              << " graph=" << graph << " passed\n";
}

unsigned long long mix(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

float draw(unsigned long long seed, int pos) {
    auto key = mix(seed ^ (static_cast<unsigned long long>(pos) * 0xD1B54A32D192ED03ull));
    key = mix(key ^ (static_cast<unsigned long long>(ops::kSamplePurposeDSparkProposal) << 21));
    return float(key >> 40) / 16777216.f;
}

void sampled_markov_case(int steps, int batch, bool graph, bool head) {
    const int vocabulary = 257, h = 5120;
    auto w1 = data(256 * vocabulary), w2 = data(256 * vocabulary);
    auto hidden = data(h * steps * batch), cw = data(h + 256);
    std::vector<std::uint16_t> cb{bf(0.125f)};
    std::vector<float> unary(16 * steps * batch);
    std::vector<int> ids(unary.size()), anchors(batch), positions(batch), counts(vocabulary, 19);
    std::vector<ops::SamplingConfig> configs(batch);
    for (int b = 0; b < batch; ++b) {
        anchors[b]                  = 3 + b;
        positions[b]                = 7913 + 5 * b;
        configs[b].seed             = 937 + b;
        configs[b].temperature      = b == 0 ? 0 : (b == 1 ? 1.f : 0.3f);
        configs[b].presence_penalty = 9.f;
        for (int k = 0; k < steps; ++k)
            for (int c = 0; c < 16; ++c) {
                const int at = c + 16 * (k + steps * b);
                ids[at]      = (c * 13 + k * 3 + b) % vocabulary;
                unary[at]    = (16 - c) * 0.13f + random_value();
            }
    }
    Buffer dcounts(counts);
    for (auto& cfg : configs) cfg.token_counts = static_cast<int*>(dcounts.p);
    Buffer di(ids);
    Buffer du(unary);
    Buffer dh(hidden), d1(w1), d2(w2);
    Buffer da(anchors), dp(positions);
    Buffer dc(configs);
    Buffer dw(cw), db(cb);
    Buffer dd(std::vector<int>(steps * batch, -9));
    Buffer dq(std::vector<float>(16 * steps * batch, -9)),
        df(std::vector<float>(steps * batch, -9));
    auto ti = di.tensor(DType::I32, {16, steps, batch}),
         tu = du.tensor(DType::FP32, {16, steps, batch});
    auto th = dh.tensor(DType::BF16, {h, steps, batch}),
         t1 = d1.tensor(DType::BF16, {256, vocabulary}),
         t2 = d2.tensor(DType::BF16, {256, vocabulary});
    auto ta = da.tensor(DType::I32, {batch}), tp = dp.tensor(DType::I32, {batch}),
         td = dd.tensor(DType::I32, {steps, batch});
    auto tq = dq.tensor(DType::FP32, {16, steps, batch}),
         tf = df.tensor(DType::FP32, {steps, batch});
    auto tw = dw.tensor(DType::BF16, {h + 256}), tb = db.tensor(DType::BF16, {1});
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto run = [&] {
        ops::dspark_markov_sample(ti, tu, th, t1, t2, ta, tp,
                                  static_cast<const ops::SamplingConfig*>(dc.p),
                                  head ? tw : Tensor{}, head ? tb : Tensor{}, td, tq, tf, stream);
    };
    if (graph) {
        cudaGraph_t g;
        cudaGraphExec_t e;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        run();
        CUDA_CHECK(cudaStreamEndCapture(stream, &g));
        CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(e, stream));
        CUDA_CHECK(cudaGraphLaunch(e, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaGraphExecDestroy(e);
        cudaGraphDestroy(g);
    } else {
        run();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const auto result = dd.read();
    const auto q = dq.read(), conf = df.read();
    for (int b = 0; b < batch; ++b) {
        int prev = anchors[b];
        for (int k = 0; k < steps; ++k) {
            const int at = k + steps * b, row = 16 * at;
            double edge[16], prob[16], sum = 0;
            int best = 0;
            for (int c = 0; c < 16; ++c) {
                double bias = 0;
                for (int r = 0; r < 256; ++r)
                    bias += double(unbf(w1[r + 256 * prev])) * unbf(w2[r + 256 * ids[row + c]]);
                edge[c] = unbf(bf(unbf(bf(unary[row + c])) + unbf(bf(float(bias)))));
                if (edge[c] > edge[best] ||
                    (edge[c] == edge[best] && ids[row + c] < ids[row + best]))
                    best = c;
            }
            const double temp = configs[b].temperature;
            for (int c = 0; c < 16; ++c) {
                prob[c] = temp > 0 ? std::exp((edge[c] - edge[best]) / temp) : (c == best ? 1 : 0);
                sum += prob[c];
            }
            double cumulative = 0, actual_sum = 0;
            int expected = ids[row + best];
            bool picked  = false;
            for (int c = 0; c < 16; ++c) {
                prob[c] /= sum;
                actual_sum += q[row + c];
                check(std::abs(q[row + c] - prob[c]) < 2e-5,
                      "sampled q differs from FP64 represented oracle");
                cumulative += prob[c];
                if (!picked && draw(configs[b].seed, positions[b] + k) < cumulative) {
                    expected = ids[row + c];
                    picked   = true;
                }
            }
            check(std::abs(actual_sum - 1) < 2e-6, "proposal q is not normalized");
            check(result[at] == expected, "sequential sampled token / RNG mismatch");
            if (head) {
                double linear = unbf(cb[0]);
                for (int i = 0; i < h; ++i)
                    linear += double(unbf(hidden[i + h * at])) * unbf(cw[i]);
                for (int r = 0; r < 256; ++r)
                    linear += double(unbf(w1[r + 256 * prev])) * unbf(cw[h + r]);
                check(std::abs(conf[at] - 1 / (1 + std::exp(-linear))) < 2e-6,
                      "confidence differs from FP64 oracle");
            } else
                check(std::isnan(conf[at]), "absent confidence must be unavailable");
            prev = result[at];
        }
    }
    check(di.read() == ids && du.read() == unary && dh.read() == hidden && d1.read() == w1 &&
              d2.read() == w2 && dcounts.read() == counts,
          "sampled Markov mutated input or committed counts");
    cudaStreamDestroy(stream);
    std::cout << "sampled Markov K=" << steps << " B=" << batch << " graph=" << graph
              << " head=" << head << " FP64 passed\n";
}

} // namespace

int main() {
    try {
        attention_case(1, 1, 0, false);
        attention_case(8, 3, 63, false);
        attention_case(8, 3, 2048, true);
        attention_case(8, 3, 2051, false);
        append_case();
        sampled_markov_case(1, 1, false, false);
        sampled_markov_case(7, 3, true, true);
        sampled_markov_case(7, 8, false, true);
        markov_case(257, 7, 3, false, true);
        markov_case(257, 1, 1, true, false);
        markov_case(248320, 7, 3, false, false);
        std::cout << "DSpark independent oracles passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
