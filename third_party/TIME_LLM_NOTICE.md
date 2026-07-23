# Time-LLM attribution

`mqids/projectors.py::ReprogrammingAttention` is adapted from the
`ReprogrammingLayer` in Time-LLM:

- Repository: https://github.com/KimMeen/Time-LLM
- Paper: *Time-LLM: Time Series Forecasting by Reprogramming Large Language Models*
- Upstream license: Apache License 2.0
- Upstream copyright notice: Copyright 2024 Time-LLM contributors

The adapted implementation replaces Time-LLM's dense full-vocabulary prototype
mapping with a small fixed bank of selected Qwen token embeddings and adds an
explicit residual path and gate. The Apache-2.0 license is available at
https://www.apache.org/licenses/LICENSE-2.0.
