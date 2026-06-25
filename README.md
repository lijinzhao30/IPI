# IPIBench: Evaluating Interactive Proactive Intelligence of MLLMs under Continuous Streams

https://lijinzhao30.github.io/IPIBench/IPI_Demo.mp4

[Project Page](https://lijinzhao30.github.io/IPIBench/) · [Paper](https://lijinzhao30.github.io/IPIBench/paper.pdf)

## Overview

Recent multimodal large language models (MLLMs) have made strong progress on reactive visual question answering. Real streaming assistants, however, need to move beyond passively answering questions: they must continuously observe visual streams, proactively respond at the right moment, support multi-turn interaction, and adapt when users add, modify, or cancel requests.

**IPIBench** evaluates this capability as **Interactive Proactive Intelligence** under continuous video streams. It covers three major interaction settings:

- **Proactive Monitoring**: when to trigger, what to understand, and how to maintain sustained and stable responses.
- **Proactive Task Management**: how to manage user instructions across cancellation, modification, and multiple concurrent tasks.
- **Interleaved Reactive–Proactive Requests**: how to coordinate reactive questions with ongoing proactive goals in multi-turn streaming scenarios.

![IPIBench teaser](assets/IPIBench.png)

## Contributions

- We introduce **IPIBench**, the first benchmark for evaluating interactive proactive intelligence of MLLMs under streaming video settings.
- We conduct systematic evaluations and failure analyses on proprietary, open-source, and online streaming models, revealing unstable proactive triggering and weak multi-turn interaction coordination.
- We propose **IPI-Agent**, a training-free agentic framework with an interaction-control policy and temporal-gating mechanism, improving proactive triggering stability and multi-turn coordination for existing offline MLLMs.

## IPI-Agent

IPI-Agent turns existing offline MLLMs into more stable, stateful streaming assistants without additional training. It separates interaction control, temporal gating, memory, and response generation so that the agent can decide when to respond, when to stay silent, and how to preserve context across evolving user requests.

![IPI-Agent architecture](assets/IPI-Agent.png)

## Repository status

This repository is currently being prepared for release.

- ✅ Project assets and paper overview are available.
- ✅ Demo video is available under `assets/`.
- 🚧 Code is coming soon.

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026ipibench,
  title={IPIBench: Evaluating Interactive Proactive Intelligence of MLLMs under Continuous Streams},
  author={Li, Jinzhao and Chen, Yinuo and Song, Wenxuan and Lei, Yijia and Zhang, Yichi and Yan, Honglei and Pan, Panwang and Liu, Miao},
  journal={arXiv preprint arXiv:2605.27074},
  year={2026}
}
```
