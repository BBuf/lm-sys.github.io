---
title: "SGLang Low-Latency Mode: Kernels and Runtime"
author: "Authors to be confirmed"
date: "October 18, 2026"
previewImg: /images/blog/sglang-vendor-kernel-runtime/runtime.svg
type: blog
---

Low-latency inference offers many opportunities for optimizations tailored to specific models, hardware and workloads. These optimizations often require extensive kernel fusion and duplicate code across implementations, making them difficult to maintain. To support this specialization, we introduce **SGLang Low-Latency Mode**. It brings specialized kernels into SGLang through a plugin framework and delivers substantial latency reductions on both AMD and NVIDIA GPUs.

*[Figure 1: Low-latency inference results]*

## Kernels for Low-Latency Inference

### AMD Low-Latency Kernels

### NVIDIA Low-Latency Kernels

[`sgl-nvidia-mega`](https://github.com/sgl-project/sgl-nvidia-mega) is the NVIDIA plugin, enabled with `--enable-nvidia-megakernel`. Its first target is **DeepSeek-V4.1-Flash with DSpark speculative decoding on four GB300 GPUs (TP4)**. The plugin binds to the weights, KV cache and scheduler that SGLang already manages, and replaces the execution of covered target-verify and draft steps; prefill and uncovered shapes stay on SGLang's own forward. In our measurements it raises per-user decode speed from 745.6 to 1,030.2 tokens/s at one request (+38%), and by 12% at eight concurrent requests.

**What batch one looks like.** DSpark drafts five tokens, and the target model verifies them together with the last accepted token, so each request contributes six rows to a verify step. One, two, four and eight concurrent requests produce 6, 12, 24 and 48 rows. At this width a decoder layer is a chain of small, dependent kernels. The attention's `wo_a` projection alone streams 16 MiB of weights per GPU to produce six output rows, and every tensor-parallel boundary waits for the other three GPUs. In SGLang's stock path, a single verify step at one request executes 1,205 GPU kernels.

**Where the time goes.** We profiled warmed CUDA-graph replays at one request: five target-verify and five draft replays on each of the four GPUs, with CPU stage annotations and launch correlation IDs identifying the GPU work of each replay. The plugin barely changes the kernel count of the verify graph (1,189 instead of 1,205), yet the median decoder layer takes 68.4 µs instead of 96.8 µs. Across a replay, the summed duration of all kernels falls by only 2–5%, while the replay's span falls by about a fifth. Most of the gain therefore comes from the critical path: work starts earlier, waits less, or leaves the critical path entirely.

| Region of one decoder layer, one request (median) | SGLang | With plugin |
| --- | ---: | ---: |
| Attention, from the previous MoE output to the output projection | 33.4 µs | 18.3 µs |
| Attention output projection and tensor-parallel reduction | 10.9 µs | 9.4 µs |
| MoE, from routing to the reduced output | 44.4 µs | 36.5 µs |
| Whole layer, from one routing step to the next | 96.8 µs | 68.4 µs |

*Medians over 780–800 instances per arm (40 layers, five replays, four GPUs). Region medians do not sum to the layer median, and layers with Engram or an indexer take longer than the rest. Profiler instrumentation lengthens kernels, so these traces explain mechanisms; the serving results at the end come from separate runs without a profiler.*

The four changes below follow the order of a layer's execution, starting with the attention region, where the largest share of time moved.

**1. Launch the next kernel before its input is ready.** [Programmatic dependent launch (PDL)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) lets a kernel start while its predecessor is still running and wait (`griddepcontrol.wait`) only before it reads the predecessor's output. Everything before that wait overlaps the predecessor: barrier and tensor-memory setup, weight loads, and reads of KV entries the graph has already written. The predecessor decides how early this can begin, because its dependents launch only once it executes `griddepcontrol.launch_dependents` or finishes.

SGLang's kernels already use PDL, and some are built for it: the fused `wo_a` kernel [streams its weights before its wait](https://github.com/sgl-project/sglang/blob/7b7015e72e4ffd6641159de0d8f3f466313f4e8b/python/sglang/kernels/jit/csrc/deepseek_v4/wo_a_fused.cuh#L444-L461). In the stock chain, however, most predecessors release their dependents late. The plugin moves those release points forward: the MoE reduction releases the next layer's first projection [when it starts polling](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/runtime/__init__.py#L40-L43), the [query RMSNorm](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/attention/q_norm_quant.py#L88-L92) and the [attention combine](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/attention/swapab.py#L42-L48) release on entry, and [`wo_a`](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/attention/src/wo_a_fused.cu#L637-L640) releases `wo_b` right after its setup.

The kernels around those release points change as well. The query and KV projections move from FlashInfer's persistent dense block-scaled GEMM to its low-latency variant, which places the weight on the 128-wide MMA dimension and the few token rows on the other, so the first projection spreads over 56 CTAs instead of 14. The sparse attention [applies RoPE as it loads the query](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/attention/chain.py#L1-L10), removing a separate query RoPE kernel. In layers whose KV and index streams are joined before the query, it also [gathers KV before its wait](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/attention/step.py#L258-L264).

Across the 600 layer instances without Engram or an indexer, the effect is visible directly in the traces. The next layer's first projection launches 3.4 µs before the MoE reduction finishes, compared with 0.1 µs in the stock path, and the attention kernel starts 6.1 µs after the reduction instead of 17.0 µs.

The two Engram layers change the most, by 42–49 µs each. An Engram table lookup, its quantization and the WKV projection depend only on token hashes, yet the stock path runs them on the layer's critical path. The plugin [computes them on a side stream](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/engram/prefetch.py#L1-L22) at the start of the step, with each GPU handling a quarter of the WKV output columns. The all-reduces stay on the main stream, in the same order on every GPU, and the result is bitwise identical to SGLang's inline path. The [dispatch plan](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/plan.py#L26-L28) enables this prefetch only at six rows, because its authors measured it slower at twelve.

**2. Publish the projection output from the GEMM epilogue.** At the end of attention, each GPU holds a partial sum of the `wo_b` projection. The four partials must be summed before the mHC residual update and the next RMSNorm. SGLang already fuses this boundary into two kernels. The GEMM writes its partial to memory, and one fused kernel [reads it back and pushes it](https://github.com/sgl-project/sglang/blob/7b7015e72e4ffd6641159de0d8f3f466313f4e8b/python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh#L356-L365) to all four GPUs with one store each, waits for the other three, reduces, applies mHC and normalizes.

The plugin also uses two kernels, but moves the push into the GEMM. While each output value is still in registers, the [publishing epilogue](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/wo_b_publish.py#L42-L80) packs eight BF16 columns with warp shuffles and writes them with a single 16-byte NVLink multicast store (`multimem.st`) into its slot in every GPU's mailbox. The host then [launches only the consumer](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/all_reduce_mhc.py#L1251-L1265), as the GEMM's PDL dependent. The consumer polls the mailbox, then reduces, mixes and normalizes; its code is unchanged, so moving the publish does not change its results. Each CTA's stores leave as soon as its tile is finished rather than after the whole GEMM grid completes, and nothing rereads the partial to communicate it. The GEMM still writes its ordinary output, and the boundary is not one kernel.

![Attention output handoff in SGLang and the NVIDIA plugin for DeepSeek-V4.1-Flash on GB300](/images/blog/sglang-vendor-kernel-runtime/nvidia-output-handoff.svg)

*Kernel boundaries at the attention output, one request (six verify rows). Both paths run two kernels per GPU; the plugin moves the push from the start of the second kernel into the epilogue of the first. Box sizes are not durations. Compared with FlashInfer's Lamport all-reduce that the consumer comes from, which publishes in a kernel of its own, the fusion also removes that launch and its grid-wide wait.*

The median handoff takes 9.4 µs instead of 10.9 µs. The plugin's GEMM also differs from the stock one, so the difference cannot be credited to publication alone. The same mailbox serves the MoE output, where a finalize kernel publishes the routed and shared-expert sums before the reduction. Both paths cover 1–16 rows, which in our tests means one and two requests.

**3. Start the expert GEMMs before routing finishes.** At one request, six rows each select six of 384 experts, giving 36 token-expert pairs. The stock path runs a Triton top-k, a TensorRT-LLM routing kernel (through FlashInfer) that builds the expert-major permutation, the two expert GEMMs and a fused finalize/all-reduce/mHC kernel. The plugin changes the front of this chain:

- A [top-k kernel](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/moe/routed_expert_gemm.py#L143-L145) with one thread per expert reproduces SGLang's router math bit for bit.
- With at most eight rows, any expert is chosen at most eight times and fits in one eight-row tile. The [route kernel](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/moe/src/route.cuh#L29-L34) uses this to place each expert's tile with popcounts over a presence bitmap, without scans or sorting. Its [first instruction](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/moe/src/route.cu#L9-L13) releases GEMM1, whose barrier and tensor-memory setup then overlaps routing.
- Each GPU's slice of an expert has 576 intermediate columns, padded to 640 in the shared weight layout. GEMM1 [never loads the all-zero tile](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/moe/routed_expert_gemm.py#L259-L263), so it runs 9 of the stock kernel's 10 weight tiles per expert.

Tile height is not one of the differences: the stock TensorRT-LLM kernels also process eight rows per expert tile, and an expert chosen by more than eight rows gets additional tiles. Across all 800 MoE windows per arm, routing completes 5.2 µs after it starts instead of 9.6 µs, GEMM1 starts after 2.6 µs instead of 9.3 µs, and the median window shrinks from 44.4 to 36.5 µs. GEMM2 and the reduction take about as long as before.

![Measured MoE timeline at one request with and without the NVIDIA plugin](/images/blog/sglang-vendor-kernel-runtime/nvidia-moe-timeline.svg)

*One matched pair from the same GPU, layer and replay, chosen because both windows lie closest to their arm's median. Bars show measured start times and durations on a shared scale, including overlap; a PDL kernel's bar includes the time it spends waiting for its producer. Shared-expert and mHC-coefficient kernels run on side streams.*

In the plugin trace, the GEMM1 bar is longer than the stock one because it launches earlier and its duration includes the wait for routing. The path is shorter without every kernel being faster.

**4. Keep sequential draft sampling inside one kernel.** DSpark's Markov head proposes draft tokens one at a time, and each proposal depends on the previous token. In the stock draft graph, each of the five positions runs six dependent kernels: embedding lookup, head GEMM, partial argmax, NVLink exchange, a second partial pass and a finish kernel. The plugin's [Markov kernel](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/markov.cu#L9-L44) runs all five positions in one cooperative launch per GPU. Each warp keeps its 16-row slice of the vocabulary-sharded head in registers for every step. GPUs exchange 64-bit candidate keys through a symmetric-memory mailbox, which serves as both the grid barrier and the cross-GPU argmax, and every wait has a deadline, so a stalled peer causes a fault rather than a hang. The proposal span falls from 85.1 µs to 34.2 µs (median of 20 replays per arm), and 30 kernels become one. The draft graph goes from 167 to 97 kernels; the remaining reduction comes from the draft model's attention, MoE and projection kernels. SGLang's speculative worker and acceptance logic are unchanged.

**Coverage.** Each path covers a range of verify rows, and dispatch uses the row count of the CUDA graph being replayed. The table marks which of the paths discussed above run at each measured concurrency; column headers give the number of requests and, in parentheses, verify rows.

| Plugin path | Rows | 1 (6) | 2 (12) | 4 (24) | 8 (48) |
| --- | :---: | :---: | :---: | :---: | :---: |
| Attention chain: early-released norm, `wq_b` and `wo_a` | 1–48 | ✓ | ✓ | ✓ | ✓ |
| Small-batch sparse attention with RoPE on load | 1–24 | ✓ | ✓ | ✓ | |
| Publishing GEMM and mailbox reduction (attention and MoE outputs) | 1–16 | ✓ | ✓ | | |
| MoE top-k, route kernel and expert GEMM pair | 1–48 | ✓ | ✓ | ✓ | ✓ |
| Low-latency `wqkv_a` GEMM and Engram prefetch | 1–8 | ✓ | | | |

As concurrency grows, fewer of these paths apply and each kernel has more rows to process, which is consistent with the per-user gain falling from 38% to 12%. A shape outside a path's range uses SGLang's kernel for that operation, while the rest of the step stays on the plugin. The supported configuration requires full decode CUDA graphs: kernels and buffers are prepared at startup, and each captured graph records the selected path. Prefill and eager fallback steps use SGLang's forward. The plugin declares a [numerical tolerance for each replaced operation](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/spec.py#L131-L136). Several are exactly zero, while some replacements are not bitwise, such as the TF32 GEMM that computes the mHC coefficients.

**Serving results.** Both arms ran on the same four-GPU GB300 node with the same software, effective server arguments and environment; only `--enable-nvidia-megakernel` differed. Four fresh servers ran in off/on/on/off order, and each measured concurrency 1, 2, 4 and 8 in forward and then reverse order. Inputs were 8,192 tokens and outputs 1,024 tokens, with identical prompts in both arms. At concurrency one, each pass had at least 60 seconds of warmup followed by 30 timed requests; higher concurrency used 2C warmup and 5C timed requests. All 800 timed requests are retained.

![Per-user decode speed versus aggregate output throughput per GPU for DeepSeek-V4.1-Flash on four GB300 GPUs, with and without the NVIDIA plugin](/images/blog/sglang-vendor-kernel-runtime/nvidia-low-concurrency.svg)

*DeepSeek-V4.1-Flash · 4 × GB300 · TP4 · DSpark block size 5 · FP8 KV cache · simulated mean acceptance length 3.51 in both arms. Large markers average the four pass summaries per arm; small dots are individual passes; bars span the two fresh-launch means and are not confidence intervals. Lines connect measured points only.*

| Requests | Per user, off | Per user, on | Change | Per GPU, off | Per GPU, on |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 745.6 | 1,030.2 | +38.2% | 165.8 | 219.3 |
| 2 | 647.6 | 837.5 | +29.3% | 266.3 | 327.1 |
| 4 | 541.0 | 658.1 | +21.6% | 408.5 | 468.6 |
| 8 | 373.0 | 416.3 | +11.6% | 552.8 | 599.2 |

Per-user decode speed, in tokens/s per user, is `(output tokens − 1) / (HTTP completion time − first text arrival)`, using the client's p50 summary. At one request, the client receives a new chunk every 3.40 ms instead of 4.70 ms (median gap between streamed chunks, one per decoding step). Aggregate throughput, in output tokens/s per GPU, is measured separately: total output tokens over the timed interval, including prefill, divided by four GPUs. Time to first token is essentially unchanged (173.9 ms off and 175.5 ms on at one request), because prefill runs SGLang's path; this unchanged prefill time is why throughput gains are smaller than per-user decode gains.

Both arms simulate the same mean acceptance length, 3.51, which fixes the amount of speculative work. It is a control, not a measurement of real-workload acceptance; completed tokens per verify call ranged from 3.483 to 3.524 across measured points. Four low-concurrency points do not form a full Pareto frontier. Both arms use per-rank Engram host tables, with 100% huge-page coverage reported for both Engram layers on every rank. The environment pins SGLang at `7b7015e72e4f` (the integration in [PR #44945](https://github.com/sgl-project/sglang/pull/44945)), the plugin at `d47deb3390e1`, model revision `314113f8a7da`, and PyTorch 2.14.1+cu130 on CUDA 13.

## Plugin System: Design and Integration

Specialized kernels can require substantial amounts of code. Each combination of hardware, model architecture and parallel configuration may need its own implementation. Different workloads add further specialization. More combinations mean more code to review, test and maintain. Bringing every implementation into SGLang would make the core codebase increasingly difficult to maintain and place a growing workload on its reviewers and maintainers.

![External kernel libraries register model specifications with SGLang, which validates configurations, builds the plugin and dispatches supported steps.](/images/blog/sglang-vendor-kernel-runtime/plugin-architecture.svg)

*External kernel libraries connect to SGLang through registration, validation, build and runtime dispatch. The diagram shows the shared design; the AMD and NVIDIA interfaces are still evolving.*

To manage these low-latency improvements, we introduce a plugin system for external kernel libraries. [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) and [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega) keep specialized kernels, weight preparation and model-specific execution code together so developers can optimize them as a unit. Each library declares its requirements and supported configurations through model specifications. SGLang uses these specifications to integrate the implementation with its serving runtime. Kernel teams can iterate in their own repositories while reusing SGLang's scheduling and runtime resources.

The integration separates startup preparation from execution:

- **Register:** A plugin registers a specification for each model architecture it supports. The specification describes the implementation's requirements and provides the entry point for building it. This gives SGLang a way to discover external implementations while keeping the details of model support in the library that owns the kernels.
- **Guard:** Specialized kernels make assumptions about hardware, precision, parallelism and serving features. SGLang checks the declared requirements before loading weights so an incompatible configuration fails early. Additional checks on the loaded model run before the plugin is built. These checks distinguish an unsupported server configuration from a supported server encountering a step outside the plugin's coverage.
- **Build:** Weight preparation and execution resources are set up during initialization so requests can reuse them. The integration accommodates the needs of each implementation: AMD can retain checkpoint tensors before weight conversion, while NVIDIA receives a binding through which it can access runtime state as that state becomes available. SGLang builds the plugin before sizing the KV cache so its allocations are accounted for, then installs dispatch in the model's forward path.
- **Runtime:** Specialized and general execution coexist because a plugin may cover only certain step types and token counts. The installed dispatcher uses the plugin's support conditions to select its forward or SGLang's original forward. Tensor-parallel ranks must agree on coverage to avoid entering incompatible collectives. This selection fits into ModelRunner's existing execution flow. During graph capture, the chosen path is recorded for that shape; serving can then replay the graph without repeating Python dispatch.

This design lets kernel teams ship focused optimizations without adding every specialized implementation to SGLang's core codebase. It concentrates integration work at explicit interfaces and allows each library to expand its coverage incrementally. Specialized kernels can serve the cases they optimize while SGLang's general forward handles the remaining steps within a supported configuration.

## Future Plans

<p style="font-size: 1.375rem; font-weight: 700; color: #111; margin-top: 1.5rem;">Broader Model Support</p>

Megakernel development is underway for GLM-5.3, K3, MiniMax M3 and other models on both AMD and NVIDIA GPUs. We plan to integrate these implementations into the plugin framework as they become ready. These low-latency kernels show promising potential for further performance improvements.

<p style="font-size: 1.375rem; font-weight: 700; color: #111; margin-top: 1.5rem;">CI and Performance Validation</p>

External kernel implementations sit outside SGLang's regular code review process. Strong CI coverage at the plugin interface is therefore essential to verify that these implementations meet SGLang's correctness and runtime requirements. We plan to test compatibility checks, weight binding, forward outputs and state updates, including KV-cache writes. End-to-end tests will also cover graph execution, fallback behavior and performance regressions as SGLang and the external libraries evolve.

<p style="font-size: 1.375rem; font-weight: 700; color: #111; margin-top: 1.5rem;">Evolving Shared Plugin Interfaces</p>

Although the plugin framework is now in place, it is still at an early stage. We expect to refine its interfaces and design through long-term iteration as we integrate more kernels and learn from their use in production.
