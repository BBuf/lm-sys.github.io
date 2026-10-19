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

To manage these low-latency improvements, we introduce a plugin system that connects external kernel libraries to SGLang. Kernel developers maintain and optimize their implementations in separate packages while SGLang provides the integration with its serving runtime. The initial implementations live in [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) and [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega). Plugins take over the model's forward for supported steps and use the weights and runtime state made available through the integration. SGLang continues to manage request scheduling and runtime resources. Each plugin must preserve the forward's output and state-update semantics, including writes to the KV cache. The interfaces continue to evolve with the needs of both implementations.

The integration starts with registration. A plugin entry point registers a specification for each supported model architecture. Guards check its hardware and configuration requirements before weight loading. SGLang then loads the weights and builds the plugin before sizing the KV cache. This stage also prepares the resources needed by the specialized implementation. AMD provides a weight observer that can retain checkpoint tensors before conversion. NVIDIA creates a binding that gives the plugin access to runtime state such as the KV pool and attention metadata when that state becomes available.

At runtime, a dispatcher selects the plugin forward for covered steps and keeps the original forward for other shapes. This fallback applies within a compatible configuration; unsupported startup configurations are rejected. Tensor-parallel ranks check that their execution coverage agrees so they do not enter incompatible collectives. Graph capture records the selected path for each shape. Replay then runs the recorded GPU work without repeating Python dispatch.

## Future Plans

### Broader Model, Hardware, and Workload Coverage

### Validation and Performance

### Evolving Shared Plugin Interfaces
