# Documentation

This directory holds the detailed operating and implementation contract for
`olmo-sglang`. The repository README intentionally stays short.

| Document | Use it when |
|---|---|
| [Getting started](getting-started.md) | Installing the extension or launching standalone OLMo inference |
| [Compatibility](compatibility.md) | Selecting SGLang/FLA versions or checking whether a checkpoint is supported |
| [Design](design.md) | Modifying model execution, KDA state, caching, or speculative verification |
| [Validation](validation.md) | Reproducing a correctness gate or using a synthetic checkpoint |
| [Status](status.md) | Deciding whether a feature is ready for an experiment or production |
| [Development](development.md) | Navigating the source tree and running repository checks |

Operational instructions describe the current code. Measurements and completed
probes in the status and validation documents are evidence, not performance
guarantees for a different checkpoint or runtime.
