# Experimental DSpark backend

This initial DSpark route uses deterministic greedy draft tokens (point-mass proposals),
including when the target sampler has nonzero temperature. Target verification still
uses that requested sampler. Stochastic Markov proposal distributions are a separate
optimization and are not implemented here.

DSpark uses the existing public Engine, Text/Vision preparation, sparse KVMem,
prefix state, batched verification and result publication. Select it explicitly:

```bash
ninfer-serve model-with-dspark.ninfer --spec dspark --draft-tokens 7
```

K is 1..7. Do not add `--lm-head-draft`: this implementation evaluates the full
public vocabulary before applying the Markov head. The supported geometry is
H5120, five local layers, D256/Q20/KV4, a 2048-token draft context and rank256
vanilla Markov weights, as in
[the RedHat Qwen3.8-27B drafter](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark).
Its source block size is eight; fixed K7 is the initially tested route. Adaptive
confidence-based verification is not implemented and confidence weights are not
loaded. Other drafter geometries, vocabulary remapping, non-causal sliding
windows and indexed proposal heads fail explicitly.

The eight source auxiliary IDs include the embedding state at index zero. They
map to zero-based decoder outputs 3/11/19/27/35/43/51/59. The shared context is
[anchor-2048, anchor), clipped at zero; query positions see their own and earlier
query slots. The existing local draft cache stores BF16 keys and FP16 values.
Anchor output row zero predicts the first new draft token. Each
Markov step conditions on the anchor or the previously selected token, with the
checkpoint's BF16 dot-result and logit-addition boundaries. CUDA tests compare
attention against an independent FP64 softmax and Markov IDs against FP64 dots
with explicit BF16 casts, including ring wrap, ragged/permuted lanes, tie breaks,
full vocabulary and Graph replay. The added Q8 projection shapes have separate
FP64 reference and launch-boundary tests.

To attach the official checkpoint to an existing encoded target without
requantizing the target:

```bash
python -m tools.convert.attach_dspark --base target.ninfer --draft checkpoint/ \
  --draft-out scratch/dspark-q8.ninfer --out target-with-dspark.ninfer
```

The tool quantizes draft projections to Q8, retains direct BF16 norms/Markov
weights, checks target config and frontend resources, copies encoded target
objects unchanged, and verifies every output object by SHA256. The output and
draft paths must not already exist. WSL mapped SMB drives that lack hardlinks
can additionally pass `--windows-python /mnt/c/Python311/python.exe`, using
Windows no-replace rename of the completed file. This affects offline publication,
not inference. Ordinary raw-checkpoint conversion also accepts `dspark` in
`--components` and a `--source dspark=...` companion.

A Huihui target can share its own embedding/head with an original-model-trained
drafter. This changes draft compatibility and possible acceptance, not the target
weight bytes. Q8 draft conversion can also affect proposals. Existing target
verification remains responsible for accepted output; backend/batch floating-point
paths can differ, so do not infer bitwise output identity or general quality
acceptance from an artifact join or a few successful prompts. Measure net committed
output per wall second, per-position acceptance, proposal cost, zero/full-accept
rounds, context length, KV transfer and memory headroom for the actual workload.
