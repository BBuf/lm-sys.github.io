# NVIDIA low-latency validation: reproduction protocol

This protocol compares SGLang with the NVIDIA plugin disabled and enabled on
the same four-GPU GB300 node. It separates model correctness, controlled static
timing, diagnostic profiles, and one recovered AgentX request.

## Frozen environment

- Hardware: four GB300 GPUs, SM103, 152 SMs/GPU, TP4, EP1.
- Container: `lmsysorg/sglang:nightly-dev-cu13-20261019-3c560089`, ARM64.
- Driver: 580.178.04; CUDA 13.0; PyTorch 2.14.1+cu130.
- FlashInfer 0.7.2; CUTLASS DSL 4.8.0; Transformers 5.19.0; fastokens 0.3.4.
- SGLang: `YAMY1234/sglang` at
  `7b7015e72e4ffd6641159de0d8f3f466313f4e8b` (integration PR #44945).
- Plugin: `sgl-project/sgl-nvidia-mega` at
  `d47deb3390e14741c9420ac702954e7611a14da1`.
- Model: `deepseek-ai/DeepSeek-V4.1-Flash` at
  `314113f8a7dadb388215396ba5fbee2c610a04d6`.

Install the two pinned Python packages with `pip install --no-deps -e` to retain
the container's binary dependencies. Record the complete package inventory.
Keep the plugin's shipped default slots. Both arms require the same per-rank
Engram host-table placement; record huge-page coverage for both Engram layers
on every rank. Startup compilation and graph capture are outside the timed region.

```bash
export SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_PROFILE_V2=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code \
  --tp 4 --ep-size 1 --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla --mem-fraction-static 0.8 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --cuda-graph-max-bs-decode 64 --max-running-requests 256 \
  --chunked-prefill-size 8192 --enable-decoder-swa-bounded-replay \
  --context-length 1048576 --watchdog-timeout 600 --enable-metrics \
  --random-seed 496233168 \
  --host 127.0.0.1 --port 31080
```

For the plugin arm, append only `--enable-nvidia-megakernel`. Confirm active
target and draft plugins on all four TP ranks. Confirm captured decode batches
include 1, 2, 3, 4, 5, 6, 7, 8 and the configured ceiling of 64. The pinned
plugin requires these intermediate graph sizes for correct decode.

## Correctness before timing

Unset all `SGLANG_SIMULATE_ACC_*` variables. Run the pinned plugin unit suite,
then the same `TestMegakernelVerify` serving suite against each arm. The native
activation test assumes a graph ceiling of 16; the runner replaces that check
with explicit all-rank activation and coverage assertions for the actual ceiling.

The serving checks are GSM8K-200 at parallelism 12 and 16 (8 shots, 512 new-token
limit), decode-versus-prefill mean KL on eight short prompts, mean KL on four
18K–27K prompts, and passkey retrieval. Thresholds are respectively >85%, <0.03,
<0.06, and an exact passkey substring. These checks can detect gross failures;
passing does not establish numerical equivalence or improved model accuracy.

From the pinned plugin checkout, the executed commands were:

```bash
python -m pytest -q test/unit
SGLANG_NVIDIA_MEGAKERNEL_E2E_URL=http://127.0.0.1:31080 \
SGLANG_NVIDIA_MEGAKERNEL_E2E_MODEL_PATH="$MODEL_PATH" \
python -m pytest -q -s test/e2e/test_deepseek_v41_server.py::TestMegakernelVerify \
  -k 'not the_tested_batches_run_the_plugin'
```

The excluded method only checks its assumed graph ceiling; the runner verifies
all four ranks' target/draft activation and actual graph batches separately.
The 249 unit tests and four serving tests per arm passed.

The pinned legacy GSM8K helper uses the first eight test examples as few-shot
demonstrations and includes those examples among the first 200 questions. It is
therefore described as a regression gate, not a clean held-out benchmark. A
follow-up audit repeats parallelism 12/16 at 512 tokens with per-question output
and completion metadata, then runs parallelism 8 with a 1,024-token cap. This
checks truncation and the low-concurrency path after the initial scores differed.
With common seed `496233168`, the off/on repeats scored 179/181, 180/185 and 181/181
out of 200 for the three settings, respectively. No response reached its cap.
Reported real acceptance lengths weighted by request verification calls ranged
from 3.39 to 3.45. Initial gates used different automatic seeds and scored 183/176
and 184/179 off/on; all observations remain in the evidence, and neither pair
establishes numerical equivalence. The paired audit includes prompt/answer
hashes, per-question outcomes, completion lengths and speculative counters.

## Controlled static timing

Use fresh servers in **off / on / on / off** order. For these timing runs only:

```bash
export SGLANG_SIMULATE_ACC_LEN=3.51
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
```

Each launch runs concurrency 1, 2, 4, 8 in forward and reverse order. Requests
contain exactly 8,192 deterministic random input IDs and produce 1,024 tokens;
sampling is greedy with `ignore_eos`. Prompt seeds match across arms and launches.
The cache is flushed before each concurrency point. At C1, warm up for at least
60 seconds and time 30 requests. At C2/C4/C8, warm up 2C requests and time 5C.
The client is a closed loop with at most C outstanding requests.

For each fresh server, run these two passes in sequence (set `ARM_RUN` to a
new directory for each of off-r1, on-r1, on-r2 and off-r2):

```bash
python static_raw.py --base-url http://127.0.0.1:31080 \
  --conc 1,2,4,8 --isl 8192 --osl 1024 --c1-requests 30 \
  --c1-warmup-seconds 60 --salt r1 --out "$ARM_RUN/pass1"
python static_raw.py --base-url http://127.0.0.1:31080 \
  --conc 1,2,4,8 --isl 8192 --osl 1024 --c1-requests 30 \
  --c1-warmup-seconds 60 --salt r2 --reverse --out "$ARM_RUN/pass2"
```

The first static baseline resolved seed `496233168` automatically; subsequent
launches explicitly pin that value. The saved server information verifies
the same effective seed in every performance arm. Initial correctness gates
used different automatic seeds; the follow-up audit pins the same seed in both arms.

- Per-user decode speed follows the supplied static client:
  `(completion_tokens - 1) / (HTTP completion time - first text chunk time)`.
- Aggregate throughput is total completion tokens divided by the timed point's
  wall time, including prefill. Divide by four for tokens/s/GPU.
- Plot coordinates average the two pass summaries within each launch, then
  average the two launch summaries. Error bars span the two launch values;
  they are not confidence intervals. Small dots retain individual pass summaries.
  The y values aggregate pass medians.
  The supplied client's percentile rule sorts n values and uses index
  `int(q * (n - 1) + 0.5)`; for even n, its p50 is the upper middle value.
  The request-level CSV retains unrounded timings so this convention is explicit.
- Chunk intervals are engine-step observations under speculative decoding,
  not individual-token latency samples.
- C1/C2/C4/C8 produce 6/12/24/48 logical verify rows with DSpark block 5.
  Record executed graph dimensions separately instead of assuming no padding.

Fixed acceptance controls execution work. These numbers do not measure a real
workload's acceptance rate and do not form a complete serving Pareto frontier.
The match-expected simulator targets mean acceptance 3.51; individual steps and
short logging windows vary. The benchmark includes normal serving effects such
as prefill/decode scheduling within the closed loop, rather than isolating a
single CUDA kernel.
Before/after metrics also verify every timed point's completed token count and
request-level speculative verify-call count. Their ratio ranges 3.483–3.524
across the 32 measured pass/concurrency points. This completed-output ratio
includes end-of-request trimming and is reported separately from the simulator
setting. Per-point counts accompany `static-results.json`.

## Diagnostic profiles

After unprofiled timing on the first launch of each arm, replay the same C1
8,192-token prompt once without profiling for warmup. Flush the cache, then
capture a full 1,024-token request with five requested steps per stage, CPU and
GPU activities, shapes, detailed annotations, and unmerged TP rank traces.
Preserve all prefill and decode traces; use the four decode traces for analysis.

```bash
python capture_static_profile.py --base-url http://127.0.0.1:31080 \
  --out "$ARM_RUN/profiles" --name "$ARM_RUN_NAME"
```

The extraction joins each CPU `step[DRAFT...]` or `step[VERIFY...]` annotation
to its enclosed `cudaGraphLaunch`, then selects GPU kernels by that launch's
correlation ID. There are five graph replays for each phase on every rank.
The target has 40 ordered MoE instances per replay, matching the model's 40
layers and the plugin's sequential layer loop. The C1 trace executes six verify
rows. Decode graph display lanes are not assumed to be independent physical
streams.

The MoE figure shows routing through routed-expert finalization/reduction/mHC.
It omits concurrent shared-expert work and mHC preparation. Its representative
pair has the same rank, replay ordinal and layer ordinal in both arms and
minimizes the sum of relative distances from each arm's median span over all
800 matched pairs. Full per-rank graph summaries and the complete set of path
spans accompany the figure.

Check actual graph replay, target/draft boundaries, logical and padded rows,
rank skew and every GPU stream. Report kernel-duration sums separately from
the union of busy intervals and elapsed step time. PDL consumers can include
dependency waits in their duration. Use unprofiled repetitions for speed claims.

## One recovered AgentX request

The user supplied a single recovered long-context chat, not a full current
AgentX benchmark. Its formatted input is 337,833 tokens, including an explicitly
primed 336,128-token cached prefix; output is fixed to 1,362 tokens. The original
request had 11 additional runtime nonce tokens that were not recovered.

The matched replay uses acceptance 5.4, ragged verification in static mode,
BF16 Markov W2, thinking enabled, high reasoning effort, fastokens, a 16,384
prefill chunk, maximum 256 running requests, and prefill/decode interval zero.
Both plugin arms retain the same required host Engram and FP8 KV setup.
Actual client concurrency remains one. The historical server cap of two clamps
captured graph batches to [1,2] in this runtime, omitting plugin-required batches
5–8. We stopped that initial baseline before measurement and restarted all four
arms with the same cap 256 and full graph ceiling 64; the incomplete attempt is
excluded from every comparison.
This differs from the historical GPU-Engram recipe, so historical latency is
not used as a baseline.

Use fresh servers in off/on/on/off order, with one warmup and ten measured
requests each. Before every request, flush and prime the exact prefix, then
verify the reported cached-token count and cache-counter delta. Verify the
formatted token SHA-256 before comparing any timing:

`f9833d8b4040e71770333ff363cb979929b363c14660fa5cfa4f1a26eae653e8`.

The unchanged supplied chat client measures TPOT as
`(last nonempty text/reasoning chunk - first nonempty chunk) / (1362 - 1)`;
TTFT is request start to first nonempty text/reasoning chunk. This differs from
the static client's HTTP-completion convention. Do not combine their numbers.
Only aggregate timings, token counts and hashes are published; the private chat
payload is excluded from the blog and reproduction bundle.
