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

**Extending fusion across the producer and its consumer.** SGLang already has fused all-reduce and mHC post-processing paths. The NVIDIA plugin extends the attention-output handoff: the output projection (`wo_b`) uses a low-latency GEMM whose epilogue publishes partial rows directly into a mailbox on every TP rank. A consumer then waits for the required rows, reduces them, updates the mHC residual streams and prepares the normalized input for the next sublayer. The MoE post uses a related mailbox path that can also combine routed-expert finalization and the shared-expert contribution. This makes the communication handoff part of the surrounding computation. The [boundary implementation](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/boundary/boundaries.py) selects these paths by their supported widths.

![Attention output handoff in SGLang and the NVIDIA plugin for DeepSeek-V4.1-Flash on GB300](/images/blog/sglang-vendor-kernel-runtime/nvidia-output-handoff.svg)

*Attention output handoff at small verify widths. The plugin couples the GEMM's output publication to the reduction consumer. The boxes describe operations and dataflow; they do not represent measured durations or a count of CUDA launches.*

**Kernels sized for a few requests.** The routed MoE uses a routing kernel and two block-scaled `tcgen05` GEMMs with eight-row tiles. Tokens are grouped by expert, with additional tiles when an expert receives more than eight rows. The router's top-k kernel preserves the reference selection behavior. The attention chain also uses small-M projection kernels and specialized normalization, quantization and output-projection stages. These paths have different coverage, so the implementation makes a per-width choice:

| Optimization in the target step | Supported logical rows in the current slot plan | Low-concurrency cases with DSpark block size 5 |
| --- | --- | --- |
| Publishing `wo_b` and its attention-post consumer | 1–16 | 1–2 requests |
| Routed MoE GEMM pair and fast router top-k | 1–48 | 1–8 requests |
| Engram lookup/WKV prefetch | 1–8 | 1 request |
| Two-shot reduction and mHC post-processing | 49–384 | Wider steps; outside the 1–8 request comparison |

*Coverage follows the [slot plan](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/plan.py). The selected kernel also depends on the built configuration and executed graph width. A plugin-supported step can still use SGLang's existing operations for individual regions.*

**Overlapping work across kernel boundaries.** The implementation keeps separate kernel launches and connects eligible stages with [programmatic dependent launch (PDL)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html). A consumer can begin independent work early, then wait before reading data from its producer. For example, the specialized sparse-attention path can gather KV data before its query dependency resolves when the graph has already ordered the relevant cache writes. The [attention dispatcher](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/attention/step.py) checks this condition. PDL offers an opportunity for overlap; actual overlap and its benefit depend on the workload and available GPU resources.

At one request, Engram lookup and WKV preparation can also run ahead on a side stream. Its collectives remain on the main stream in a consistent order across ranks. Keeping this prefetch limited to its measured small-width range avoids assuming that overlap remains beneficial as the MoE workload grows.

**Keeping sequential draft sampling on the GPU.** DSpark's Markov proposal depends on the previously drafted token. The plugin runs this proposal in one kernel per rank, retains reusable head data across draft positions and exchanges token candidates through GPU mailboxes. The draft model also has its own attention, MoE and projection kernels. This reduces boundaries within the draft proposal while retaining SGLang's speculative worker and acceptance flow. The [Markov kernel](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/sm103/markov.cu) implements the sequential proposal and cross-rank agreement.

The supported serving configuration uses full decode CUDA graphs. Kernels and buffers are prepared at model initialization, and the graph records the selected path for each covered shape. Prefill, unsupported widths and ordinary eager fallback steps use SGLang's original forward. Cross-GPU waits have deadlines, and the plugin declares per-operation numerical tolerances as part of its [model specification](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/sgl_nvidia_mega/deepseek_v41/spec.py).

**Controlled decode results.** The [reported comparison](https://github.com/Oasis-Git/lm-sys.github.io/pull/1) uses DeepSeek-V4.1-Flash on one four-GPU GB300 node, TP4, FP8 KV cache and the FlashMLA attention backend. Both arms use the same SGLang build and launch settings; the plugin flag is the only difference. On static 8K-input / 1K-output requests, reported per-user decode speed improves by 38%, 30%, 23% and 11% at concurrency 1, 2, 4 and 8.

![Reported per-user decode speed for DeepSeek-V4.1-Flash with DSpark on four GB300 GPUs, comparing SGLang with and without the NVIDIA megakernel flag](/images/blog/sglang-vendor-kernel-runtime/nvidia-static-decode.svg)

*DeepSeek-V4.1-Flash · 4 × GB300 · TP4 · DSpark block size 5 · FP8 KV cache · static 8K input / 1K output. "Plugin off" is SGLang without `--enable-nvidia-megakernel`; "plugin on" enables the shipped NVIDIA kernel slots. Speculative acceptance is fixed at 3.51 tokens per verify step in both arms. This is a controlled low-concurrency decode comparison, separate from an AgentX workload or a full serving Pareto frontier.*

Fixing acceptance makes the two arms execute a comparable number of verify steps, which helps isolate execution cost. It does not measure the acceptance rate of a real workload. The [reproduction protocol](https://github.com/sgl-project/sgl-nvidia-mega/blob/d47deb3390e14741c9420ac702954e7611a14da1/benchmark/README.md) also specifies the per-rank Engram host table, TP synchronization settings and prepared huge-page conditions. Those conditions need to accompany reproduced results because host-table placement can affect this workload.

Accuracy is evaluated separately with real DSpark acceptance. The existing report gives GSM8K-200 scores of 196/200 without the plugin and 197/200 with it at the reported concurrency settings, and full passes on its 16K and 24–32K long-context sets (96/96 and 32/32). These are the observed test results; the one-question difference does not establish an accuracy improvement or numerical equivalence.

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
