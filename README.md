# Faithfulness and correctness of RAG attributions for automated fact-checking

Code for my Master's thesis <br>
*"Explaining Retrieval-Augmented Generation: Faithful Answer Attribution for Automated Fact-Checking"* <br>
at Technische Universität Berlin, Quality & Usability Lab, 2026 <br>
First Supervisor: Prof. Sebastian Möller, Second Supervisor: Prof. Axel Küpper <br>
Additional Supervisors: Charlott Jakob, Nils Feldhus

### Summary
Retrieval-augmented LLMs are prominently used for automated fact-checking to generate textual verdicts grounded in external evidence. Automatically generated citations can be used to attribute LLM-generated statements to external sources, thereby enhancing the verifiability and trustworthiness of automated fact-checking systems for human users. This thesis compares attribution methods from different paradigms (post-hoc, direct, and contributive attribution), evaluating both their correctness (i.e., whether cited sources *factually support* a statement) and their faithfulness (i.e., whether the cited sources actually *cause* the LLM to generate a statement) on the AVeriTeC fact-checking dataset.


### Interactive results exploration

[![Open in Streamlit](https://img.shields.io/badge/Open_in-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=262730)](https://fact-checking-rag-attributions.streamlit.app/)

Explore the attributions and their correctness and faithfulness interactively!

<!-- 
- 🤗 Experiment results: [jonaspeschel/fact-checking-attribution-results](https://huggingface.co/datasets/jonaspeschel/fact-checking-attribution-results)
- 🤗 Datasets: [AVeriTeC with scraped gold evidence](https://huggingface.co/datasets/jonaspeschel/AVeriTeC-with-scraped-gold-evidence), [AVeriTeC with custom web evidence](https://huggingface.co/datasets/jonaspeschel/AVeriTeC-with-scraped-web-evidence), [MultiFieldQA-en](https://huggingface.co/datasets/jonaspeschel/MultiFieldQA-en) -->

## Repository structure

```
context_cite_metrics/         # Attribution methods + evaluation (Ch. 3 & 4.2)
verdict_verification/         # Verdict verification with pruned context (Ch. 4.3)
user_study/                   # Streamlit app used for the user study (Ch. 4.4)
custom_averitec_dataset/      # custom web retrieval pipeline (App. F)
eval_metrics_visualization/   # Streamlit app for interactive results exploration
```


### Attribution methods

The following methods are implemented in [context_cite_metrics/answer_attribution_methods.py](context_cite_metrics/answer_attribution_methods.py):

| Method | Identifier |
|---|---|
| Contributive attribution (**[ContextCite](https://arxiv.org/abs/2409.00729)**) | `context_cite` |
| Direct attribution | `longcite_llm_direct` |
| NLI post-hoc: naive / sliding window (3, 5) / greedy sampling | `nli_post_hoc_*` |
| LLM post-hoc | `llm_post_hoc` |
| Semantic similarity (`all-mpnet-base-v2`) | `semantic_similarity` |
| Leave-one-out | `leave_one_out` |

Answers are generated with [LongCite-llama3.1-8b](https://huggingface.co/THUDM/LongCite-llama3.1-8b) or [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

## Setup

```bash
git clone git@github.com:jonas-peschel/faithful-fact-check.git
cd faithful-fact-check
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt
```

Create a `.env` file in the repository root with the API keys needed for the steps you want to run:

```bash
HF_TOKEN=...          # gated models (Llama 3.1-8B-Instruct) and datasets
OPENAI_API_KEY=...    # GPT-4o citation correctness evaluation
DEEPSEEK_API_KEY=...  # statement decomposition, citation correctness evaluation, verdict verification
SERPER_API_KEY=...    # Google search for web evidence retrieval
```

## Pipeline overview

The experiments run as a sequence of scripts that pass JSON results files from one step to the next. Each script documents its arguments in `--help`. 

**1. Custom evidence retrieval** (optional: the resulting dataset is available on **[🤗 Hugging Face](https://huggingface.co/datasets/jonaspeschel/AVeriTeC-with-custom-web-evidence)**). Run from `custom_averitec_dataset/pipeline_scripts/`:

| Script | Purpose |
|---|---|
| `search_query_generation.py` | Generate search queries per claim (Ministral-3-14B) |
| `evidence_retrieval.py` | Google search via Serper API + scraping with trafilatura/pdfplumber |
| `evidence_ranking.py` | Hybrid BM25 + dense ranking (mxbai-embed), then re-ranking (mxbai-rerank) |

**2. Answer attribution.** Run from `context_cite_metrics/`:

| Script | Purpose |
|---|---|
| `answer_attribution_methods.py` | Generate answers and compute attribution scores per method |
| `statement_decomposition.py` | Decompose answers into atomic statements (using DeepSeek-V3) for NLI-based post-hoc attribution |
| `get_citations.py` | Turn continuous attribution scores into discrete citations |
| `format_results.py` | Convert results into the evaluation format |

**3. Faithfulness and correctness evaluation.** Run from `context_cite_metrics/`:

| Script | Purpose |
|---|---|
| `llm_judge_eval_cite.py` | Citation correctness (precision & recall) evaluation via an LLM judge (adapted from LongCite/SelfCite) |
| `llm_pred_veracity.py`, `veracity_classification_eval_correct.py` | Veracity label prediction from the justification and its evaluation |
| `calc_cc_metrics.py` | Citation faithfulness evaluation (top-k log-prob drop and linear datamodeling score) |
| `plot_log_prob_drop.py`, `plot_linear_datamodeling_score.py` | Plots for the faithfulness metrics |
| `merge_eval_results.py` | Merge citation correctness, faithfulness, and veractity classification results into one file |

**4. Verdict verification.** Run from `verdict_verification/`:

| Script | Purpose |
|---|---|
| `verdict_verification.py` | Re-predict verdicts given only the top-k attributed sentences |
| `verdict_verification_metrics.py` | Evaluation metrics for verdict verification|

<br>
Example usage for attribution using ContextCite and faithfulness evaluation:

```bash
cd context_cite_metrics
python answer_attribution_methods.py \
    --dataset averitec --use_longcite \
    --attr_methods context_cite \
    --cc_num_ablations 32 256 --cc_batch_size 4 \
    --results_path results/averitec_longcite.json
python calc_cc_metrics.py --dataset averitec \ 
    --use_longcite --cc_batch_size 4 \
    --metrics log_prob_drop --results_path results/averitec_longcite.json
```

## Acknowledgements

This code builds on:
- **[AVeriTeC](https://github.com/MichSchli/AVeriTeC)** (Schlichtkrull et al., 2023) as fact-checking dataset with real-world claims with web evidence and veracity labels
- **[ContextCite](https://github.com/MadryLab/context-cite)** (Cohen-Wang et al., 2024) for contributive attribution and faithfulness metrics
- **[LongCite](https://github.com/THUDM/LongCite)** (Zhang et al., 2024) and **[SelfCite](https://github.com/facebookresearch/SelfCite)** (Chuang et al., 2025) for direct attribution and the citation correctness evaluation

