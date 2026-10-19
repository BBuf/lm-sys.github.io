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

Specialized kernels can require substantial amounts of code. Different hardware, models and parallel configurations can each call for a different implementation. Workload differences add further specialization. Keeping all of these implementations in SGLang would place a growing burden on maintainers who must review, test and maintain them. We therefore keep the kernels in external libraries and connect them to SGLang through plugin interfaces. This lets kernel developers iterate aggressively within their own packages while SGLang focuses on their integration with the serving runtime. The initial implementations live in [sgl-amd-mono](https://github.com/sgl-project/sgl-amd-mono) and [sgl-nvidia-mega](https://github.com/sgl-project/sgl-nvidia-mega). Their interfaces are evolving alongside the kernels.

The integration operates at the model's forward entry point. A plugin can organize the execution of a supported step and fuse operations across it. SGLang continues to manage request scheduling and runtime resources. The plugin uses the model weights and serving state provided through the integration and must preserve the forward's output and state-update semantics, including writes to the KV cache. The two implementations expose the resources their kernels need: AMD can retain checkpoint tensors before model-specific weight conversion, while NVIDIA provides a binding for runtime state such as the KV pool and attention metadata. Keeping these integration points explicit makes dependencies easier to review as both the kernels and SGLang evolve.

Each plugin registers a model-specific specification that declares its requirements and supported execution scope. When the user enables the backend, SGLang validates the configuration and builds the plugin after weight loading but before sizing the KV cache. Unsupported startup configurations are rejected. Within a compatible configuration, covered steps use the plugin and other shapes retain the original forward. Tensor-parallel ranks check that their execution coverage agrees so they do not enter incompatible collectives. This dispatch also fits the existing graph execution path: capture records the selected implementation for a shape, and replay runs the recorded GPU work without repeating Python dispatch.

## Future Plans

### Broader Model, Hardware, and Workload Coverage

### Validation and Performance

### Evolving Shared Plugin Interfaces
