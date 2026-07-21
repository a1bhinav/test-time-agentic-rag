# Test-Time Strategies for More Efficient and Accurate Agentic RAG

Code for the ACL SRW 2026 paper *Test-Time Strategies for More Efficient and Accurate Agentic RAG*.

Paper: [[[link](https://aclanthology.org/2026.acl-srw.41.pdf)]].

## Abstract

Agentic Retrieval-Augmented Generation (RAG) systems iteratively interleave reasoning, search, and generation, but this comes at the cost of long contexts crowded with redundant retrieved passages and rising token budgets per question. We investigate inference-time strategies that can be added to an already-trained agentic RAG model — without further fine-tuning — to make it both more efficient and more accurate. Building on Search-R1, we evaluate two complementary strategies on HotpotQA: (1) **deduplicated retrieval**, which suppresses repeat documents across successive search calls within a single trajectory, and (2) **cache-extracted retrieval**, in which a small auxiliary extractor model condenses each retrieved batch into a focused evidence cache before it is fed back to the agent. We show that these test-time strategies reduce token usage and improve answer accuracy relative to the Search-R1 baseline, suggesting that careful management of the agent's context — not just better policies — is a meaningful axis for improving agentic RAG.

## Architecture overview

The pipeline consists of three loosely coupled components:

1. **Reasoning agent** — a Search-R1 PPO-trained Qwen2.5 (3B/7B) model (`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-{3b,7b}-em-ppo`) that interleaves `<think>`, `<search>`, and `<answer>` turns and stops on `</search>` to delegate retrieval.
2. **Retriever** — a local FastAPI server exposing `POST /retrieve` over a Wikipedia/HotpotQA index (Search-R1's `retrieval_server`). The agent reaches it at `http://127.0.0.1:8000/retrieve`.
3. **Evidence shaping (test-time strategies)** — applied between retrieval and the agent's next turn:
   - *Baseline*: passages are concatenated and returned as-is.
   - *No-duplicate-docs*: a per-trajectory set of seen `doc_id`s filters out repeats; `topk` is auto-expanded until 3 fresh docs are found.
   - *Cache-extracted*: an auxiliary Qwen2.5-7B-Instruct extractor compresses retrieved passages into a `<cache>...</cache>` evidence block, which replaces the raw passages.

```
   +-------------+        <search>q</search>        +-----------+
   |  Search-R1  |  ------------------------------> |  FastAPI  |
   |   agent     |  <-- shaped <information> ------ |  retriever|
   +-------------+                                  +-----------+
          ^                                              ^
          |                                              |
          +---- evidence-shaping strategy ---------------+
               (none / dedup / cache-extracted)
```

## Repository structure

```
search_r1_code/
  Search-R1/                          # vendored fork of upstream Search-R1
    LICENSE, Notice.txt               # original Apache-2.0 license + notice (preserved)
    infer.py                          # baseline single-question demo
    infer_hotpot_500_no_dup_docs.py   # OUR test-time strategy: de-duplicated retrieval
    infer_hotpot_500_caching.py       # OUR test-time strategy: cache-extracted (contextualization)
    inference_script.sh               # SLURM template wiring retriever + inference
    retrieval_launch.sh               # launches the FAISS retriever HTTP server
    results/                          # OUR inference logs and retrieved-docs cache
    ...                               # all other files unchanged from upstream Search-R1
  retrieve.py                         # OUR standalone retriever harness over inference20.md
  retrieve_output.json                # cached retrieval outputs for inference20.md
  infer_hotpot_log_retrieval.py       # OUR variant of infer.py that logs every retrieval call
  inference20.md                      # 20-question reasoning-trace dump
README.md, LICENSE, .gitignore        # repo-level files
```

### What is vendored vs. what is ours

`search_r1_code/Search-R1/` is a **vendored fork** of the upstream
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) repository (Bytedance Ltd. and
affiliates), redistributed under Apache-2.0 with the original `LICENSE` and
`Notice.txt` preserved in-tree. Most files inside this directory (everything under
`verl/`, `search_r1/`, `scripts/`, `docs/`, `example/`, `setup.py`,
`requirements.txt`, etc.) are unmodified upstream code.

Our test-time modifications for the paper live in:

- `search_r1_code/Search-R1/infer_hotpot_500_no_dup_docs.py` — **de-duplication** pipeline: per-trajectory tracking of seen `doc_id`s; `topk` is auto-expanded until 3 fresh docs are returned.
- `search_r1_code/Search-R1/infer_hotpot_500_caching.py` — **contextualization** pipeline: an auxiliary Qwen2.5-7B-Instruct extractor compresses retrieved passages into a `<cache>...</cache>` evidence block before the agent sees them.
- *Hybrid pipeline* — combines the de-duplication and contextualization strategies. **TODO:** the script is not yet checked in; it will land at `search_r1_code/Search-R1/infer_hotpot_500_hybrid.py`.
- `search_r1_code/Search-R1/inference_script.sh` and `retrieval_launch.sh` — orchestration scripts (SLURM template + retriever launcher).
- `search_r1_code/retrieve.py`, `retrieve_output.json`, `inference20.md`, `infer_hotpot_log_retrieval.py` — supporting harnesses for retrieval inspection.

For the latest upstream Search-R1 (training, evaluation, additional retrievers,
ongoing fixes), clone the original repository directly:
<https://github.com/PeterGriffinJin/Search-R1>.

## Installation

Tested with Python 3.10, CUDA 12.x, and 1× A100 (80 GB) for the 7B model.

```bash
git clone <this-repo-url> agentic-rag
cd agentic-rag/search_r1_code/Search-R1
pip install -r requirements.txt
# Search-R1 adds verl/vLLM extras; follow Search-R1/README.md if you need training,
# only inference dependencies are required to reproduce our results.
```

### Environment variables

Inference uses local Hugging Face models and a local retriever, so no LLM-provider API keys are required for the main results.

**Paths** — the inference and retriever scripts read these via `os.environ.get(...)` and fall back to `./data` / `./models` if unset:

| Variable | Default | Purpose |
|---|---|---|
| `DATASET_DIR` | `./data` | Parent directory containing the `nq_hotpotqa/val_split_500.parquet` evaluation split. |
| `MODEL_HUB_DIR` | `Qwen/Qwen2.5-7B-Instruct` | Local snapshot path for the auxiliary cache-extractor model. Leave unset to download from the HuggingFace Hub. |
| `RETRIEVER_INDEX_DIR` | `./data/save_path` | Directory containing `e5_Flat.index` and `wiki-18.jsonl` for the FAISS retriever (used by `retrieval_launch.sh`). |

**Optional LLM-provider keys** — only needed if you wire in an OpenAI-backed extractor as a drop-in replacement for the local cache model:

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
# Azure OpenAI alternative:
export AZURE_OPENAI_API_KEY="YOUR_KEY_HERE"
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export OPENAI_API_VERSION="2024-08-01-preview"
```

Never hardcode keys; the scripts read everything through `os.environ.get(...)`.

## Reproducing the main results

1. **Build the retrieval index** (once). Download the HotpotQA + NQ Wikipedia index following [Search-R1/docs/retriever.md](search_r1_code/Search-R1/docs/retriever.md). Then export `DATASET_DIR` to the parent of `nq_hotpotqa/val_split_500.parquet` and `RETRIEVER_INDEX_DIR` to the directory holding `e5_Flat.index` and `wiki-18.jsonl`.

2. **Start the retriever** in a separate shell, listening on port 8000:
   ```bash
   cd search_r1_code/Search-R1
   python search_r1/search/retrieval_server.py --port 8000  # see retrieval_server.py for index args
   ```

3. **Run each test-time strategy**:
   ```bash
   cd search_r1_code/Search-R1
   # Baseline (single-question smoke test)
   python infer.py
   # Strategy 1: deduplicated retrieval over HotpotQA val-500
   python infer_hotpot_500_no_dup_docs.py
   # Strategy 2: cache-extracted retrieval over HotpotQA val-500
   python infer_hotpot_500_caching.py
   ```
   Outputs are written to `results/inference500_no_repeat_docs.txt`, `results/inference_w_caching.txt`, and `results/inference500.txt`.

4. **Score**: exact-match against `golden_answers` from the same val-500 split. The reference logs in `results/` correspond to the numbers reported in the paper.

## Citation

```bibtex
@inproceedings{sharma2026agenticrag,
  title     = {Test-Time Strategies for More Efficient and Accurate Agentic RAG},
  author    = {Sharma, Abhinav and Zhang, Brian and Guntur, Deepti and Zuo, Zhiyang and Chaudhari, Shreyas and Zhao, Wenlong and Dernoncourt, Franck and Mathur, Puneet and Rossi, Ryan A. and Lipka, Nedim},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop (ACL SRW)},
  year      = {2026},
  note      = {BibTeX placeholder; replace with the ACL Anthology entry once available.}
}
```

## License & acknowledgements

This repository is released under the MIT License (see [LICENSE](LICENSE)) for the code authored for this paper.

The `search_r1_code/Search-R1/` subdirectory is vendored from [Search-R1](https://github.com/PeterGriffinJin/Search-R1) (Bytedance Ltd. and affiliates) and remains under its original Apache-2.0 license; see `search_r1_code/Search-R1/LICENSE` and `Notice.txt`. We thank the Search-R1 authors for releasing their training code and checkpoints.
