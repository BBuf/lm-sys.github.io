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

To manage these low-latency improvements, we introduce a plugin system for external kernel libraries. The first two implementations are [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) and [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega). Each library owns its specialized kernels and model-specific execution code. SGLang manages request scheduling and runtime resources. This separation lets kernel developers iterate independently while keeping their implementations connected to SGLang's serving runtime.

The integration follows the stages in the figure. Plugins register their model-specific implementations and declare the configurations they support. SGLang checks compatibility before loading weights, then builds the plugin and installs dispatch in the model's forward path. The integration also provides the weights and runtime state needed by the kernels. AMD can retain checkpoint tensors before weight conversion. NVIDIA uses a binding to access state such as the KV cache and attention metadata. Both implementations must preserve the model forward's outputs and required state updates.

SGLang's ModelRunner coordinates specialized and general execution. Steps supported by the plugin use its forward implementation; other steps use SGLang's original forward. This choice follows the plugin's declared coverage, with agreement across tensor-parallel ranks. An incompatible startup configuration raises an error rather than silently falling back. With graph execution enabled, capture records the selected path for each shape. Serving then replays that graph without repeating the Python dispatch.

## Future Plans

### Broader Model, Hardware, and Workload Coverage

### Validation and Performance

### Evolving Shared Plugin Interfaces
