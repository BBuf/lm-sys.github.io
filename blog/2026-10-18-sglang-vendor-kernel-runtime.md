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

[`sgl-nvidia-mega`](https://github.com/sgl-project/sgl-nvidia-mega) provides model-specific kernels for NVIDIA GPUs through `--enable-nvidia-megakernel`. Its first implementation supports **DeepSeek-V4.1-Flash with DSpark speculative decoding on four GB300 GPUs (TP4)**. It reuses the weights, KV cache and serving scheduler managed by SGLang, and specializes the execution of supported target verify steps and draft operations.

At low concurrency, the useful work in each matrix multiplication is small, while the layer still has to perform routing, attention, residual mixing and tensor-parallel communication. DSpark also changes what "batch 1" means: with a draft block size of five, one request produces **six rows in a target verify step**. Concurrency 1, 2, 4 and 8 therefore corresponds to 6, 12, 24 and 48 logical verify rows; CUDA graph padding can increase the executed width. These are the shapes the low-latency kernels target.

**Fusing output publication into the GEMM.** The NVIDIA attention-output handoff uses two kernels per TP rank. **The first fuses the `wo_b` GEMM and mailbox publication into one kernel:** its epilogue writes TP-partial rows directly to the peer mailboxes. The second is a separate consumer that reduces those rows, updates the mHC residual streams, combines the next input and applies RMSNorm. Relative to the unfused Lamport path, the [publishing GEMM](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/wo_b_publish.py#L42-L80) removes a standalone publish launch, its grid wait and the extra read of the projection output needed for publication. The [consumer dispatch](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/all_reduce_mhc.py#L1248-L1264) skips that publish launch when the GEMM has already published. Our stock SGLang baseline already uses a GEMM followed by a fused all-reduce/mHC/normalization kernel, so the measured comparison is two kernels versus two at this boundary. The benefit is bringing publication into the producer and avoiding the publication reread; the entire handoff does not become one kernel. The MoE post uses a related mailbox path that can also combine routed-expert finalization and the shared-expert contribution.

![Attention output handoff in SGLang and the NVIDIA plugin for DeepSeek-V4.1-Flash on GB300](/images/blog/sglang-vendor-kernel-runtime/nvidia-output-handoff.svg)

*Attention-output boundary at concurrency one (six verify rows). Each box is one kernel, and both measured paths have two kernels per TP rank. Inside the plugin's first kernel, GEMM and mailbox publication are fused; reduction, mHC and RMSNorm execute in the second. The fusion removes a separate publication stage from the unfused Lamport implementation, not a kernel from the already-fused SGLang baseline. Box widths and arrow lengths do not represent durations; mHC coefficient preparation is outside this view.*

**Kernels for small token batches.** With few concurrent requests, each expert receives only a small number of token rows. The plugin's MoE kernels group tokens by expert and process each expert's two matrix multiplications in tiles of eight rows. Here, eight is the kernel's internal tile size: an expert with more than eight rows uses additional tiles. Attention projections also use kernels designed for small row counts. Each optimization has its own supported range, so increasing concurrency changes which ones run. For the target verify step, DSpark block size five produces six token rows per request: concurrency 1, 2, 4 and 8 gives 6, 12, 24 and 48 rows before graph padding. The table shows three optimizations relevant to these test points.

| Optimization | Supported verify token rows, after graph padding | Covered concurrency points in this test |
| --- | --- | --- |
| Attention output: fuse `wo_b` GEMM with mailbox publication, then run the reduction/mHC/RMSNorm consumer | 1–16 | 1 and 2 requests |
| MoE: expert GEMMs with eight-row tiles and fast router top-k | 1–48 | 1, 2, 4 and 8 requests |
| Engram: prefetch lookup results and WKV computation | 1–8 | 1 request |

For example, **four concurrent requests produce 24 verify rows**. At that width, the specialized MoE kernels still run, while the attention-output publication path and Engram prefetch listed above are outside their supported ranges. Those regions use their other execution paths; the plugin remains active for the rest of the step. The ranges follow the [dispatch plan](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/plan.py) with the [default options](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/runtime/__init__.py#L14-L32) used in our measurements. Dispatch uses the row count of the executed CUDA graph, including any padding, so request count alone does not determine which kernels run.

**Overlapping work across kernel boundaries.** The implementation keeps separate kernel launches and connects eligible stages with [programmatic dependent launch (PDL)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html). A consumer can begin independent work early, then wait before reading data from its producer. For example, the specialized sparse-attention path can gather KV data before its query dependency resolves when the graph has already ordered the relevant cache writes. The [attention dispatcher](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/attention/step.py) checks this condition. PDL offers an opportunity for overlap; actual overlap and its benefit depend on the workload and available GPU resources.

At one request, Engram lookup and WKV preparation can also run ahead on a side stream. Its collectives remain on the main stream in a consistent order across ranks. Keeping this prefetch limited to its measured small-width range avoids assuming that overlap remains beneficial as the MoE workload grows.

**Keeping sequential draft sampling on the GPU.** DSpark's Markov proposal depends on the previously drafted token. The plugin runs this proposal in one kernel per rank, retains reusable head data across draft positions and exchanges token candidates through GPU mailboxes. The draft model also has its own attention, MoE and projection kernels. This reduces boundaries within the draft proposal while retaining SGLang's speculative worker and acceptance flow. The [Markov kernel](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/markov.cu) implements the sequential proposal and cross-rank agreement.

The supported serving configuration uses full decode CUDA graphs. Kernels and buffers are prepared at model initialization, and the graph records the selected path for each covered shape. Prefill, unsupported widths and ordinary eager fallback steps use SGLang's original forward. Cross-GPU waits have deadlines, and the plugin declares per-operation numerical tolerances as part of its [model specification](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/spec.py).

**What changes in a warmed CUDA graph.** We captured five target-verify and five draft graph replays on each of the four TP ranks at concurrency one. CPU stage annotations and CUDA launch correlation IDs identify the GPU work belonging to each replay. The target graph executes 1,189 kernels with the plugin, versus 1,205 without it; the draft graph executes 97 versus 167. In particular, ten partial and five finish kernels in the stock Markov proposal become one plugin proposal kernel. These counts are consistent across every captured rank and replay. Most of the target benefit therefore needs an explanation beyond launch-count reduction.

![Measured routed MoE timeline with and without the NVIDIA plugin, showing overlapping operations for six verify rows](/images/blog/sglang-vendor-kernel-runtime/nvidia-moe-timeline.svg)

*One matched target-layer example, selected by proximity to both arms' median routing-to-post span. Across 40 layers, five replays and four ranks (800 windows per arm), that span is 44.42 µs with the plugin off and 36.54 µs with it on. Rows identify operations and preserve their measured overlap; concurrent shared-expert work and mHC preparation are omitted.*

The timeline shows how routing, expert GEMMs and the output handoff compose. A shorter path does not require every individual kernel to be faster: GEMM 1 in the illustrated plugin path has a longer instrumented duration, while later operations overlap. PDL consumers can spend part of their measured duration waiting for dependencies, so adding kernel durations would overstate elapsed work. These CUPTI traces are diagnostic evidence; the serving results below come from separate runs without a profiler.

**Controlled decode results.** We independently compared the shipped default plugin slots on one four-GPU GB300 node, using TP4, FP8 KV cache, FlashMLA and the same software revisions and effective server configuration. Four fresh servers ran in off/on/on/off order. Each launch measured concurrency 1, 2, 4 and 8 in forward and reverse order, with identical 8,192-token inputs and 1,024-token outputs across arms. At concurrency one, each pass included at least 60 seconds of warmup followed by 30 timed requests. Higher-concurrency points used 2C warmup requests and 5C timed requests.

![Independently measured low-concurrency throughput and per-user decode speed for DeepSeek-V4.1-Flash on four GB300 GPUs](/images/blog/sglang-vendor-kernel-runtime/nvidia-low-concurrency.svg)

*DeepSeek-V4.1-Flash · 4 × GB300 · TP4 · DSpark block size 5 · FP8 KV cache. “Plugin on” enables `--enable-nvidia-megakernel`; “off” retains SGLang's stock execution. Both arms simulate a mean acceptance length of 3.51. Large points average the four pass summaries per arm; small dots show individual passes, and bars span the two fresh-launch means. The bars are not confidence intervals. Connecting lines only guide the eye across these low-concurrency measurements.*

| Concurrent requests | Plugin off, tokens/s/user | Plugin on, tokens/s/user | Decode speed increase |
| --- | ---: | ---: | ---: |
| 1 | 745.6 | 1,030.2 | 38.2% |
| 2 | 647.6 | 837.5 | 29.3% |
| 4 | 541.0 | 658.1 | 21.6% |
| 8 | 373.0 | 416.3 | 11.6% |

Per-user decode speed (x-axis) is `(output tokens − 1) / (HTTP completion time − first text arrival)`, summarized with the supplied client's p50 convention. Aggregate output throughput (y-axis) is measured separately over the complete timed request interval, including prefill, then divided by four GPUs. All 800 timed requests were retained for verification.

Matching the simulated mean acceptance controls the amount of speculative work; it does not measure a real workload's acceptance rate. Individual steps and short windows still vary; completed output tokens per request-level verify call ranged from 3.483 to 3.524 over the measured points. These points cover low-concurrency serving, not a complete Pareto frontier. Both arms use the same per-rank Engram host tables and TP synchronization settings, with 100% huge-page coverage reported for both Engram layers on every rank. The environment pins SGLang at `7b7015e72e4f`, the plugin at `d47deb3390e14`, model revision `314113f8a7da`, and PyTorch 2.14.1+cu130 on CUDA 13. Reproduction requires that pinned SGLang integration build from [PR #44945](https://github.com/sgl-project/sglang/pull/44945).

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
