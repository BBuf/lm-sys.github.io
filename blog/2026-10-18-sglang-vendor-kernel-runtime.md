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

Specialized kernels can require substantial amounts of code. Each combination of hardware, model architecture and parallel configuration may need its own implementation. Different workloads add further specialization. As these combinations grow, so does the work required to review the code and validate its correctness and performance. Bringing every implementation into SGLang would make the core codebase increasingly difficult to maintain and place a growing workload on its reviewers and maintainers.

To manage these low-latency improvements, we introduce a plugin system that connects external kernel libraries to SGLang. Kernel developers maintain and optimize their implementations in separate packages while SGLang provides the integration with its serving runtime. The initial implementations live in [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) and [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega). Plugins take over the model's forward for supported steps and use the weights and runtime state made available through the integration. SGLang continues to manage request scheduling and runtime resources. Each plugin must preserve the forward's output and state-update semantics, including writes to the KV cache. The interfaces continue to evolve with the needs of both implementations.

Each plugin registers a model-specific specification that declares its requirements and supported execution scope. When the user enables the backend, SGLang validates the configuration and builds the plugin after weight loading but before sizing the KV cache. Unsupported startup configurations are rejected. Within a compatible configuration, covered steps use the plugin and other shapes retain the original forward. Tensor-parallel ranks check that their execution coverage agrees so they do not enter incompatible collectives. This dispatch also fits the existing graph execution path: capture records the selected implementation for a shape, and replay runs the recorded GPU work without repeating Python dispatch.

## Future Plans

### Broader Model, Hardware, and Workload Coverage

### Validation and Performance

### Evolving Shared Plugin Interfaces
