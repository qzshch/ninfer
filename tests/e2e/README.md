# KVMem inference regression pipeline

Run from the NInfer checkout on the CUDA 13.1 / sm_120a machine. Python 3.11,
CMake, Ninja, idle test ports (8095 through 8100), and an explicit v3 model are required.
The runner owns only its child process; it fails if another server owns the port.
It does not stop services, retry failed requests, or treat HTTP 200 as successful
generation. An engine error, missing SSE terminator, missing usage, incorrect
recall, or silent retrieval capture fails the run.

```bash
export PYTHON=/home/druid/.local/bin/python3.11
export NINFER_MODEL='/mnt/d/LLM Model/qwen3_8_27b_nvfp4.ninfer'
bash tests/e2e/run_kvmem_pipeline.sh smoke
bash tests/e2e/run_kvmem_pipeline.sh regression
bash tests/e2e/run_kvmem_pipeline.sh long
```

`smoke` builds affected targets, runs numerical/storage/options tests and real
JSON/SSE generation, tool histories with large output budgets, cancellation, and
subsequent inference. `regression` additionally crosses a small KV window twice,
recalls a historical needle, checks that nonzero Q features selected and restored
Host pages, generates at least 512 tokens across page-growth boundaries, and
exercises 2048-token chunks, ordinary decoding, and dense mode.
It also runs a multi-chunk tool-tail replay with `max_tokens=1`: no spare output
budget may hide omitted replay service quanta. The focused `replay-budget` profile
runs that request and verifies subsequent inference; use the default 16K context,
64-page window and a chunk no larger than 2048 for this fixture.
It also constrains Host KV to 32 MiB, requires an over-budget request to fail
with HTTP 400 before execution, and verifies subsequent inference stays healthy.
`long` adds the production 96K-device-window / 256K-logical-context ladder through
250K nominal tokens, twice on the same engine after cache-producing requests.
Actual token counts and inference latency are recorded, not inferred from bytes.

Reports are saved under `out/kvmem-tests/<UTC timestamp>/`: JSON, JUnit XML, engine
logs, GPU memory/utilization samples, hardware, and source metadata. `NINFER_TEST_OUTPUT` and `NINFER_BUILD_DIR`
override output/build directories. A missing GPU is a failure, not a successful
skip. Existing scripts outside this repository are not needed.
`NINFER_TEST_PORT` overrides the first port; each matrix configuration uses the
next port to avoid TIME_WAIT conflicts between successive server processes.

For a focused run against a freshly built binary:

```bash
"$PYTHON" tests/e2e/kvmem_suite.py --model "$NINFER_MODEL" \
  --output /tmp/kvmem-check-unique --profile regression --window 64 --context 16384
```

Numerical oracle tests qualify attention mathematics. The model tests establish
the listed inference behaviors, not LongMemEval quality parity or universal
long-context accuracy. A report is only evidence for its recorded configuration.
Recall requires the correct final answer line, not a substring appearing anywhere.
`exact_answer_format` separately records whether the model omitted all extra prose;
format compliance is reported rather than used as a storage-stability gate.

The manual GitHub Actions workflow requires a self-hosted Linux x64 runner with
the `ninfer-sm120a` label and a repository variable `NINFER_E2E_MODEL` pointing to
the model on that runner. Optionally set `NINFER_E2E_PYTHON`. It uploads reports
even after failure. Workflow installation and remote execution are separate from
a passing local run.

## Diagnostic quality comparisons

`kvmem_quality.py` uses frozen, seeded fixtures for needles at 10/50/90% history
depth, long tool tails, two-hop questions, updated facts, and unanswerable questions.
It grades the entire answer as a single exact JSON object; prose containing the
correct substring and duplicate JSON keys fail. Each report retains complete
requests, raw responses, gold answers, token usage, engine logs, and GPU samples.

```bash
"$PYTHON" tests/e2e/kvmem_quality.py fixtures --output out/quality/fixtures.json
"$PYTHON" tests/e2e/kvmem_quality.py run --fixtures out/quality/fixtures.json \
  --model "$NINFER_MODEL" --window 0 --port 8105 --output out/quality/dense
"$PYTHON" tests/e2e/kvmem_quality.py run --fixtures out/quality/fixtures.json \
  --model "$NINFER_MODEL" --window 64 --port 8106 --output out/quality/sparse
"$PYTHON" tests/e2e/kvmem_quality.py compare \
  out/quality/dense/quality.json out/quality/sparse/quality.json
```

The reference server can run the same fixtures with `--engine kvmem --binary
/path/llama-kvmem-server --model /path/model.gguf`. Sparse reference runs require
explicit `--reference-budget` and `--reference-reserve` token counts. `--window 0`
selects its dense route. NInfer's current rolling window and the reference's separate
selection/generation budgets differ; record them explicitly when interpreting results.
NInfer INT8 and reference q8_0 KV, and their model weight formats, are not identical.

The comparison rejects incomplete runs or different fixture hashes. It reports
paired regressions/improvements and always labels this small synthetic suite
`diagnostic-only`; it cannot certify KVMem quality equivalence. Representative
long-context datasets and matched reference controls are still required.

## LongMemEval-S full-history predictions

`longmemeval-source.json` pins the official cleaned 500-question source by revision,
size and SHA256, plus the official evaluator revision. Download the `url` in that
lock file to a local data path. Preparation refuses different bytes. The conversational
prompt adaptation preserves every dated session and both user/assistant turns,
removes annotation fields, and places the actual question in the final user message.
It does not truncate or select evidence sessions. This differs from the paper's
generation prompt and must be reported as such.

```bash
"$PYTHON" tests/e2e/kvmem_longmemeval.py prepare --data /path/longmemeval_s_cleaned.json \
  --output out/lme/full-fixtures.json
# Optional pipeline diagnostic: add --per-stratum 1 to freeze ten questions,
# selected by ID hash across task type and abstention, without looking at answers.
"$PYTHON" tests/e2e/kvmem_longmemeval.py run --fixtures out/lme/full-fixtures.json \
  --model "$NINFER_MODEL" --context 262144 --window 512 --host-mib 12288 \
  --port 8120 --output out/lme/ninfer-sparse-32k
```

Reference invocation uses the same `--engine kvmem`, `--binary`, `--reference-budget`
and `--reference-reserve` arguments as the diagnostic runner. Run dense and sparse
configurations serially on one GPU. Context overflow, a length-truncated answer, or
a request error stays in the report and fails the run; there is no silent shortening
or automatic retry. `fixtures.json` preserves the complete prompts;
`predictions.json` preserves responses, usage, timing, gold and request hashes.
`hypotheses.jsonl` is compatible with the official evaluator's input format.

Prediction success does not assign correctness: reports remain `scored: false`.
The official evaluator uses task-specific semantic rubrics. Preserve those scores
separately from stricter review of contradictions and unsupported assertions in the
whole answer; do not replace semantic scoring with substring matching. An incomplete
prediction set or a ten-question pilot cannot establish benchmark equivalence.

`kvmem_quality_stats.py` reports conservative paired accuracy-difference intervals
from exact binomial bounds on discordant pairs and a Bonferroni correction.
The question-only interval assumes independent questions. LongMemEval comparisons
require a separate, source-bound cluster map because original/abstention variants
and reused evidence are dependent candidates:

```bash
"$PYTHON" tests/e2e/kvmem_longmemeval.py clusters --data /path/longmemeval_s_cleaned.json \
  --output out/lme/clusters.json
```

This grouping links question-ID stems and shared answer-session IDs transitively,
without looking at model answers. The frozen source has 466 groups: 432 singletons
and 34 pairs. Group membership does not modify inference fixtures. The interval
allows arbitrary dependence within each group and assumes independent groups with
a common distribution within each group-size stratum. It retains the original
question-weighted accuracy: for a size-s group's net correct-answer change D,
E[D] = sum(P(D >= j)) - sum(P(D <= -j)), j=1..s. Exact binomial bounds on those
tail probabilities, with Bonferroni across strata and tails, yield the reported
conservative interval. Unidentified dependence remains a limitation; the grouping
is a safeguard, not proof of independence. Small strata can make the interval wide.
Even zero disagreements has nonzero uncertainty, and 500 questions do not guarantee
enough power to certify a 1pp margin. Statistics do not establish
comparability of model weights, budgets or grading, and never auto-certify parity.

For this integration the agreed overall noninferiority margin is **1 percentage
point**, supported by a paired **95%** interval; task families are reported separately.
The comparison emits an interval-only indicator against that fixed margin, while
`equivalence_established` stays false until the dataset, execution and comparability
requirements have also been qualified. Small pilot intervals are insufficient.

Whole-answer reviews can be attached without altering the raw predictions:

```bash
"$PYTHON" tests/e2e/kvmem_semantic_review.py --predictions out/lme/run/predictions.json \
  --output out/lme/run/reviews.json
# A reviewer now fills boolean rubric/strict labels and rationales, and identifies
# the actual reviewer and rubric. Null/unknown labels stay unresolved.
"$PYTHON" tests/e2e/kvmem_semantic_review.py --predictions out/lme/run/predictions.json \
  --reviews out/lme/run/reviews.json --output out/lme/run/reviewed.json
```

The importer rejects stale hashes, edited answers, missing IDs, nonboolean or
unresolved labels, and missing rationales. Manual pilot review is labeled as such,
not as the official GPT-4o score. Compare complete `reviewed.json` files with
`kvmem_quality.py compare --clusters out/lme/clusters.json`; all failures remain
in the denominator. Missing/stale clustering is rejected for LongMemEval.
Source adjudication retains the original `judge_report_sha256`, `judge_settings`,
literal evidence and any citation-source corrections, and adds identified
`adjudication` metadata describing the actual reviewer and protocol. Comparison
requires the same adjudication protocol on both sides. The imported report binds
the complete review using SHA256 of its canonical `encode()` JSON; raw predictions
and API reports stay unchanged. An unresolved source/gold conflict remains null,
so it cannot silently enter a completed accuracy report.

## Qwen semantic judge

The user's selected provider is the existing Qwen API. Run the judge where its
credential file already lives; the key is read locally and is never written into
reports. Use the same explicit API/model, rubric and settings for every comparison:

```bash
"$PYTHON" tests/e2e/kvmem_judge_evidence.py --data /path/longmemeval_s_cleaned.json \
  --output out/lme/grading-evidence.json
"$PYTHON" tests/e2e/kvmem_qwen_judge.py --predictions out/lme/run/predictions.json \
  --evidence out/lme/grading-evidence.json \
  --api-url "$QWEN_API_URL" --protocol anthropic \
  --api-key-file "$QWEN_KEY_FILE" --model "$QWEN_JUDGE_MODEL" \
  --output out/lme/run/qwen-review
```

Before scoring a benchmark, run the fixed calibration with the same API arguments,
replacing `--predictions ...` with
`--calibration tests/e2e/qwen-judge-calibration.json` and using a fresh output folder.
Its nine predefined base/strict label pairs are never sent to the API. A mismatch
returns nonzero and is retained in `calibration.json`; passing this small check
does not prove general grading accuracy.

This task-specific rubric adaptation returns separate base/strict labels with
reasons. The judge sees the whole answer and dated source evidence, but no engine
identity. Evidence contains all complete sessions identified by the frozen source's
answer-session annotations, including repeated IDs in history order. It is selected
independently of model answers and used only for grading, never inference. The file
hash is bound to both engines' judge settings. These sessions are not exhaustive
history: omission alone does not prove a historical claim false; unresolved material
claims require review against the full source. `question_date` anchors relative
intervals but does not discard provided records, consistent with the full-history
generation task. Record dates and event dates must be distinguished.
Raw requests, responses, timestamps, returned model and hashes are preserved.
The protocol is explicit and bound to resume/comparison settings. Anthropic bases
append `/v1/messages` and use top-level system/thinking fields; OpenAI bases append
`/chat/completions`. Negative verdicts must include literal candidate and comparison
quotes, checked against the actual input. This detects fabricated quotations, but
does not prove the judge's semantic interpretation; retain calibration and review
evidence before using scores for acceptance.
Malformed JSON, duplicate keys, unresolved labels, invalid quotes, truncated responses
and HTTP errors remain failed attempts; later cases still run. `reviews.json` retains
null labels for unresolved cases, and no completed `reviewed.json` is produced until
every case is valid. `--resume` refuses changed predictions/settings/rubric and retains
semantic/schema-invalid responses without calling the API again. Transport failures
can be retried explicitly; do not resample judgments until a desired label appears.
The audited citation-source policy may correct only a wrong source-field name when
the unchanged quote occurs verbatim in exactly one other supplied field. It never
changes quotes or labels. `--revalidate-from /path/old-report` permits that parser-only
revalidation of identical raw requests/responses; it refuses altered inputs/rubrics
and is mutually exclusive with `--resume`. Original reports remain unchanged.
Model aliases do not guarantee immutable server weights: record this limitation
when no dated snapshot is available. These are Qwen scores, not official GPT-4o scores.
Paired semantic comparisons reject different judges/rubrics/settings, incomplete
execution, missing labels and inconsistent totals.

## SWE mini agentic supplement

`kvmem_swe.py` uses the existing NAS EvalScope 1.12.0 / swebench 4.1.0 installation.
Its source lock binds every byte and ordered instance ID of a local JSONL snapshot.
The current NAS mini set contains 50 instances: 25 Django and 25 Sphinx. The frozen
cache has no verified immutable upstream revision; its content hash is the authority.
This is an agentic executable-test supplement, not a replacement for the 500-question
long-memory acceptance set. Fifty questions cannot by themselves establish a 1pp margin.

```bash
"$PYTHON" tests/e2e/kvmem_swe.py --data /path/swe-mini-frozen.jsonl \
  --source-lock /path/swe-mini-source.json --api-url http://dev-host:8127/v1 \
  --output /path/new-run --prepare-only
# Add --pilot-per-repo 1 for a deterministic two-instance preparation.
# To execute, use another fresh output directory and omit --prepare-only.
```

Preparation validates the actual EvalScope local loader, source hashes, IDs, package
versions and Docker access. A manifest records package-source hashes, image identities,
generation configuration and selected instances. Execution uses one request at a time,
greedy/no-thinking generation, no cached answers, no automatic HTTP retries, and at
most 250 tool-loop steps per instance. Model input follows the agentic adapter's issue
description and tool protocol; oracle source text is not supplied. Keep both engines'
complete traces, patches and executable test results. `evalscope_finished` means only
that EvalScope returned: verify all instance outcomes and environment failures before
computing paired resolved rates. Neither preparation nor process completion certifies parity.

The development endpoint must use an independent port outside 8080/8081. For a
WSL server bound to localhost, run a process-scoped bridge on its Windows host:

```powershell
node tests/e2e/kvmem_nas_bridge.cjs WINDOWS_LAN_IP 8127 WSL_SERVER_PORT NAS_IP
```

Only the selected NAS IP can connect; the bridge does not alter firewall rules
or system forwarding, and rejects production ports 8080/8081. Stop it after the
benchmark. NAS-to-Windows-to-WSL `/health` returned 200 with the reference server
on 8128. Use NO_PROXY for the development host when the NAS has an HTTP proxy.
This is a connectivity check, not an agentic SWE result.

For network-dependent SWE tests, `--container-proxy http://NAS_LAN_IP:7897`
passes an existing reachable proxy to agent sandboxes and the separate grading
containers. It records the setting, preserves local HTTP/HTTPS test servers through
NO_PROXY, and scopes the grading SDK override to this process and `swebench/sweb.eval.*`
images. It changes no host proxy, image, test or scoring rule. Proxy URLs with
credentials are rejected because manifests are retained. Validate gold patches
before attributing a network-dependent failure to either model. On the NAS pilot,
Sphinx's gold patch initially failed three PASS_TO_PASS external-link checks;
with the existing NAS proxy, both fixed pilot instances passed their full gold checks.

`kvmem_suite.py --profile replay-cancel --context 196608 --window 64` tests cancellation
after retrieval with a long tool suffix. It requires the next request to finish within
five seconds. It waits until replay actually begins, then disconnects; this check
is also part of `long`. A 135258-token tool-tail replay originally blocked the
next request for 37.27 seconds. Bounded replay steps restored cancellation; retain
the run reports for backend-specific timing and the final binary's qualification.

Each server run writes `host-memory.csv` with process RSS, anonymous/file memory,
swap, and Linux VM headroom every two seconds. To avoid repeating a WSL-wide OOM,
the runner terminates only its owned server if MemAvailable + SwapFree falls below
3 GiB. It preserves `memory-abort.json` and fails the run; this is not a passing
capacity check. Windows system Commit is a separate host limit and still needs
host-side monitoring. Recorded execution overrides are restricted to the CUDA
Graph and allocator diagnostic flags; credentials are never recorded.
