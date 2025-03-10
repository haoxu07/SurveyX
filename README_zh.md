<h2 align="center">SurveyX: Academic Survey Automation via Large Language Models</h2>

<p align="center">
  <i>
✨ 欢迎来到SurveyX！在这个Github中您可以通过提交 issue 的方式定制生成特定领域的学术综述论文。📚
  </i>
  <br>
  <a href="https://arxiv.org/abs/2502.14776">
      <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B" alt="arxiv paper">
  </a>
  <a href="http://www.surveyx.cn">
    <img src="https://img.shields.io/badge/SurveyX-Web-blue" alt="surveyx.cn">
  </a>
  <a href="https://huggingface.co/papers/2502.14776">
    <img src="https://img.shields.io/badge/Huggingface-🤗-yellow" alt="huggingface paper">
  </a>
  <a href="https://discord.gg/stWuP7SY">
    <img src="https://img.shields.io/badge/channel-Discord-5865f1" alt="discord channel">
  </a>
</p>

经过一段时间的紧张开发，SurveyX 的第一个用户界面版本现已上线！🎉
我们热情邀请您试用。不要错过！🔥

<div align="center">
  👉 <strong><a href="https://surveyx.cn/">Visit SurveyX</a></strong> 👈
</div>

## 📋 Features

- 用户友好的界面
- 高效的调查管理
- 实时数据分析

## 💡 Getting Started

1. 访问 [SurveyX](https://surveyx.cn/).
2. 注册一个账户.
3. 开始轻松创建和管理您的综述论文！


## 🛠️ Feedback

我们重视您的反馈！如有任何问题或建议，请随时联系我们。感谢您的支持！❤️ 立即加入我们的微信天使用户群！🚀 扫描下方二维码，与我们一起塑造未来！💡

<div align="center">
  <img src="assets/user_groups_123.jpg" alt="Wechat Group" width="300" />
</div>

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=IAAR-Shanghai/SurveyX&type=Timeline)](https://star-history.com/#IAAR-Shanghai/SurveyX&Timeline)

## 🤔 什么是SurveyX？

![surveyx_frame](assets/SurveyX.png)

**SurveyX** 是一个先进的学术综述论文自动生成系统，利用大型语言模型（LLMs）生成高质量、领域特定的学术论文综述。🚀

只需提供 **论文标题** 和 **关键词** 进行文献检索，用户就可以请求针对特定主题的综合学术综述论文。

如果您想了解SurveyX的工作原理或想了解更多关于底层技术和方法，请访问我们的 📑[网站](http://www.surveyx.cn) 以了解更多。

## 🤔 这个Git仓库是干什么的？

这个GitHub仓库旨在为用户提供一个平台，通过提交问题来请求生成高质量、领域特定的学术综述。该仓库的主要目的是让用户轻松创建和接收定制的学术综述论文，这些内容由 SurveyX 自动生成。📄

通过提交包含论文标题和关键词的问题，用户可以快速生成关于特定主题的综合综述论文。这个过程通过自动化论文创作简化了学术研究，节省了用户在编译研究内容上的时间和精力。📚💡
**请注意，我们当前仅支持英文的输入与英文的文献生成，更多语言的支持会在未来更新**

## 🖋️ 如何通过提交问题请求定制论文

要请求一篇论文，请创建一个新的issue并提供以下详细信息：

- **Paper Title**: Provide the title of the paper you need.
- **Keywords for Literature Search**: Provide keywords separated by commas that will help retrieve relevant literature and guide the content generation (e.g. "AI in healthcare, climate change impact on agriculture, ethical implications of AI"). We recommend that the number of keywords be limited to **4 ~ 5** for optimal results.
- **Your email**(optional): Please provide your email address so that we can notify you promptly once the paper is ready.

### 💬 issue 的示例提交

> **Title**: Controllable text generation of LLM: A Survey
>
> **Keywords**: AI, healthcare, ethical implications, technology adoption, AI-driven diagnostics
>
> **Email**: <xxxxxxxx@SurveyX.cn>

一旦提交请求，生成的论文将被放置在 **user requests** 文件夹中。请允许1-2个工作日进行处理和生成。⏳

## 📝 已生成的主题

![many_papers](assets/many_papers.png)

### 示例论文

| Title                                                        | Keywords                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [From BERT to GPT-4: A Survey of Architectural Innovations in Pre-trained Language Models](./examples/Computation_and_Language/Transformer.pdf) | Transformer, BERT, GPT-3, self-attention, masked language modeling, cross-lingual transfer, model scaling |
| [Unsupervised Cross-Lingual Word Embedding Alignment: Techniques and Applications](./examples/Computation_and_Language/low.pdf) | low-resource NLP, few-shot learning, data augmentation, unsupervised alignment, synthetic corpora, NLLB, zero-shot transfer |
| [Vision-Language Pre-training: Architectures, Benchmarks, and Emerging Trends](./examples/Computation_and_Language/multimodal.pdf) | multimodal learning, CLIP, Whisper, cross-modal retrieval, modality fusion, video-language models, contrastive learning |
| [Efficient NLP at Scale: A Review of Model Compression Techniques](./examples/Computation_and_Language/model.pdf) | model compression, knowledge distillation, pruning, quantization, TinyBERT, edge computing, latency-accuracy tradeoff |
| [Domain-Specific NLP: Adapting Models for Healthcare, Law, and Finance](./examples/Computation_and_Language/domain.pdf) | domain adaptation, BioBERT, legal NLP, clinical text analysis, privacy-preserving NLP, terminology extraction, few-shot domain transfer |
| [Attention Heads of Large Language Models: A Survey](./examples/Computation_and_Language/attn.pdf) | attention head, attention mechanism, large language model, LLM,transformer architecture, neural networks, natural language processing |
| [Controllable Text Generation for Large Language Models: A Survey](./examples/Computation_and_Language/ctg.pdf) | controlled text generation, text generation, large language model, LLM,natural language processing |
| [A survey on evaluation of large language models](./examples/Computation_and_Language/eval.pdf) | evaluation of large language models,large language models assessment, natural language processing, AI model evaluation |
| [Large language models for generative information extraction: a survey](./examples/Computation_and_Language/infor.pdf) | information extraction, large language models, LLM,natural language processing, generative AI, text mining |
| [Internal consistency and self feedback of LLM](./examples/Computation_and_Language/inter.pdf) | Internal consistency, self feedback, large language model, LLM,natural language processing, model evaluation, AI reliability |
| [Review of Multi Agent Offline Reinforcement Learning](./examples/Computation_and_Language/multi-agent.pdf) | multi agent, offline policy, reinforcement learning,decentralized learning, cooperative agents, policy optimization |
| [Reasoning of large language model: A survey](./examples/Computation_and_Language/reason.pdf) | reasoning of large language models, large language models, LLM,natural language processing, AI reasoning, transformer models |
| [Hierarchy Theorems in Computational Complexity: From Time-Space Tradeoffs to Oracle Separations](examples/Computational_Complexity/P_vs_.pdf) | P vs NP, NP-completeness, polynomial hierarchy, space complexity, oracle separation, Cook-Levin theorem |
| [Classical Simulation of Quantum Circuits: Complexity Barriers and Implications](examples/Computational_Complexity/BQP.pdf) | BQP, quantum supremacy, Shor's algorithm, post-quantum cryptography, QMA, hidden subgroup problem |
| [Kernelization: Theory, Techniques, and Limits](examples/Computational_Complexity/fixed.pdf) | fixed-parameter tractable (FPT), kernelization, treewidth, W-hierarchy, ETH (Exponential Time Hypothesis), parameterized reduction |
| [Optimal Inapproximability Thresholds for Combinatorial Optimization Problems](examples/Computational_Complexity/PCP.pdf) | PCP theorem, approximation ratio, Unique Games Conjecture, APX-hardness, gap-preserving reduction, LP relaxation |
| [Hardness in P: When Polynomial Time is Not Enough](examples/Computational_Complexity/SETH.pdf) | SETH (Strong Exponential Time Hypothesis), 3SUM conjecture, all-pairs shortest paths (APSP), orthogonal vectors problem, fine-grained reduction, dynamic lower bounds |
| [Consistency Models in Distributed Databases: From ACID to NewSQL](examples/Database/CAP.pdf) | CAP theorem, ACID vs BASE, Paxos/Raft, Spanner, NewSQL, sharding, linearizability |
| [Cloud-Native Databases: Architectures, Challenges, and Future Directions](examples/Database/CAP.pdf) | cloud databases, AWS Aurora, Snowflake, storage-compute separation, auto-scaling, pay-per-query, multi-tenancy |
| [Graph Database Systems: Storage Engines and Query Optimization Techniques](examples/Database/graph.pdf) | graph traversal, Neo4j, SPARQL, property graph, subgraph matching, RDF triplestore, Gremlin |
| [Real-Time Aggregation in TSDBs: Techniques for High-Cardinality Data](examples/Database/time.pdf) | time-series data, InfluxDB, Prometheus, downsampling, time windowing, high-cardinality indexing, stream processing |
| [Self-Driving Databases: A Survey of AI-Powered Autonomous Management](examples/Database/auto.pdf) | autonomous databases, learned indexes, query optimization, Oracle AutoML, workload forecasting, anomaly detection |
| [Multi-Model Databases: Integrating Relational, Document, and Graph Paradigms](examples/Database/mmd.pdf) | multi-model database, MongoDB, ArangoDB, JSONB, unified query language, schema flexibility, polystore |
| [Vector Databases for AI: Efficient Similarity Search and Retrieval-Augmented Generation](examples/Networking_and_Internet_Architecture/vector.pdf) | vector database, FAISS, Milvus, ANN search, embedding indexing, RAG (Retrieval-Augmented Generation), HNSW |
| [Software-Defined Networking: Evolution, Challenges, and Future Scalability](examples/Networking_and_Internet_Architecture/open.pdf) | OpenFlow, control plane/data plane separation, NFV orchestration, network slicing, P4 language, OpenDaylight, scalability bottlenecks |
| [Beyond 5G: Architectural Innovations for Terahertz Communication and Network Slicing](examples/Networking_and_Internet_Architecture/network.pdf) | network slicing, MEC (Multi-access Edge Computing), beamforming, mmWave, URLLC (Ultra-Reliable Low-Latency Communication), O-RAN, energy efficiency |
| [IoT Network Protocols: A Comparative Study of LoRaWAN, NB-IoT, and Thread](examples/Networking_and_Internet_Architecture/LPWAN.pdf) | LPWAN, LoRa, ZigBee 3.0, 6LoWPAN, TDMA scheduling, RPL routing, device density management |
| [Edge Caching in Content Delivery Networks: Algorithms and Economic Incentives](examples/Networking_and_Internet_Architecture/CDN.pdf) | CDN, Akamai, cache replacement policies, DASH (Dynamic Adaptive Streaming), QoE optimization, edge server placement, bandwidth cost reduction |
| [A survey on  flow batteries](examples/Other/battery.pdf)    | battery electrolyte formulation                              |
| [Research on battery electrolyte formulation](examples/Other/flow_battery.pdf) | flow batteries  |

## 📃 引用SurveyX

如果这个项目对您的项目/论文有帮助，请引用我们：

```plain text
@misc{liang2025surveyxacademicsurveyautomation,
      title={SurveyX: Academic Survey Automation via Large Language Models}, 
      author={Xun Liang and Jiawei Yang and Yezhaohui Wang and Chen Tang and Zifan Zheng and Shichao Song and Zehao Lin and Yebin Yang and Simin Niu and Hanyu Wang and Bo Tang and Feiyu Xiong and Keming Mao and Zhiyu li},
      year={2025},
      eprint={2502.14776},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14776}, 
}
```

<hr style="border: 1px solid #ecf0f1;">

## ⚠️ 注意

- 我们的检索引擎可能无法访问许多需要商业许可的论文。如果您的研究主题需要来自非arXiv来源的论文，由于检索范围的限制，生成论文的质量和全面性可能会受到影响。
- 目前我们仅支持生成英语学术综述论文。其他语言的支持仍在开发中，会在不久的将来上线！
- 为了确保所有用户的公平访问，每个用户每天重复（或多次）提交的issue将会被忽略，优先满足多样化用户需求。

## ⚠️ 免责声明

SurveyX使用先进的语言模型协助生成学术论文。然而，请注意生成的内容仅作为研究辅助工具。用户应验证生成论文的准确性，因为SurveyX无法保证完全符合学术标准。
