---
title: "SGLang Low-Latency Mode: Kernels and Runtime"
author: "Authors to be confirmed"
date: "October 18, 2026"
previewImg: /images/blog/sglang-vendor-kernel-runtime/runtime.svg
type: blog
---

Today we introduce low-latency inference mode in SGLang. It combines specialized kernels with runtime support to reduce inference latency. The plugin system lets us iterate aggressively on these optimizations as hardware and models evolve. Our initial implementations support AMD and NVIDIA GPUs.

*[Figure 1: Low-latency inference results]*

## Kernels for Low-Latency Inference

### AMD Low-Latency Kernels

### NVIDIA Low-Latency Kernels

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

<p style="font-size: 1.375rem; font-weight: 700; color: #111; margin-top: 1.5rem;">Broader Model, Hardware, and Workload Coverage</p>

We plan to extend support to more models, GPU architectures and parallel configurations. We will also expand the batch sizes and context lengths where specialized execution provides a measurable benefit. Each extension will be evaluated against SGLang's general forward to establish where it improves latency and where the general path remains preferable.

<p style="font-size: 1.375rem; font-weight: 700; color: #111; margin-top: 1.5rem;">CI and Performance Validation</p>

As SGLang and the external libraries evolve together, CI needs to catch both interface breakages and execution regressions. We plan to strengthen coverage from plugin contract tests to end-to-end serving tests on supported hardware. These tests should check output correctness, graph execution and fallback behavior, alongside latency and throughput under representative workloads.

<p style="font-size: 1.375rem; font-weight: 700; color: #111; margin-top: 1.5rem;">Evolving Shared Plugin Interfaces</p>

The AMD and NVIDIA integrations address different implementation needs today. As we gain experience with both, we plan to identify common requirements in registration, compatibility checks and resource binding. We will use that experience to improve shared interfaces and reduce duplicated integration work while preserving the flexibility each kernel library needs.
