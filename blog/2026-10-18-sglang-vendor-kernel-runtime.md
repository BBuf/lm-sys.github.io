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

To manage these low-latency improvements, we introduce a plugin system that connects external kernel libraries to SGLang. The left side of the figure shows [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) and [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega). Each library provides model specifications, weight preparation and specialized kernels. The right side shows how SGLang integrates these components:

- **Register:** Register model-specific kernels from external libraries. Each plugin supplies a specification that identifies the model and declares its requirements and supported execution scope.
- **Guard:** Validate hardware/configuration compatibility. SGLang checks the selected plugin against the GPU, model and serving configuration before loading weights. An incompatible setup raises an error.
- **Build:** Install forward dispatch and bind weights. After loading the model, SGLang builds the plugin and installs a dispatcher in the model's forward path. The plugin prepares the resources its kernels need before SGLang sizes the KV cache.
- **Runtime:** Coordinate plugin and native execution in SGLang's ModelRunner. The plugin's support conditions determine which path runs. Supported steps use the **specialized forward**; other steps use **SGLang's general forward**. Tensor-parallel ranks must agree on the supported steps. With graph execution enabled, capture records the selected path and serving replays it without repeating Python dispatch.

## Future Plans

### Broader Model, Hardware, and Workload Coverage

### Validation and Performance

### Evolving Shared Plugin Interfaces
