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

To manage these low-latency improvements, we introduce a plugin system that connects external kernel libraries to SGLang. The initial libraries are [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) and [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega). Each provides the three components shown on the left: model specifications that declare support and requirements, weight preparation for its execution path, and specialized kernels. Kernel developers maintain these implementations in separate packages. SGLang provides the integration with its serving runtime.

The right side shows the integration flow. **Register** brings model-specific kernels from the external libraries into SGLang through plugin specifications. **Guard** validates hardware and configuration compatibility before weight loading and rejects unsupported setups. **Build** prepares the plugin after model loading, binds its weights and installs dispatch in the model's forward path. Plugin resources are prepared before SGLang sizes the KV cache. **Runtime** coordinates execution in SGLang's ModelRunner: supported steps use the plugin's specialized forward, while other steps use SGLang's general forward. Tensor-parallel ranks must agree on this coverage. With graph execution enabled, capture records the selected path and serving replays it without repeating Python dispatch.

## Future Plans

### Broader Model, Hardware, and Workload Coverage

### Validation and Performance

### Evolving Shared Plugin Interfaces
