import json 
import streamlit as st
from huggingface_hub import hf_hub_download 

#--- citation recall colors ---#
SUPPORTED_COLOR = "#c8f7c5"
PARTIAL_SUPPORT_COLOR = "#fdf7c3"
UNSUPPORTED_COLOR = "#f7c5c5"
SUPPORT_SCORE2COLOR = {
    0: UNSUPPORTED_COLOR,
    0.5: PARTIAL_SUPPORT_COLOR,
    1: SUPPORTED_COLOR,
}

#--- citation precision colors ---#
RELEVANT_COLOR = "#1b8d36"
IRRELEVANT_COLOR = "#cc1414"
RELEVANT_SCORE2COLOR = {
    0: IRRELEVANT_COLOR,
    1: RELEVANT_COLOR,
}

def load_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data

def get_attr_method(path: str):
    """Extract the attribution method name from the results filepath name."""

    attr_method_names = ["context_cite_32", "context_cite_64", "context_cite_128", "context_cite_256", "semantic_similarity", "leave_one_out", "nli_post_hoc_naive", 
     "nli_post_hoc_sliding_window_3", "nli_post_hoc_sliding_window_5", "nli_post_hoc_greedy_sampling", "llm_post_hoc", "longcite_llm_direct"]
    
    for attr_method_name in attr_method_names:
        if attr_method_name in path:
            return attr_method_name
    
# prettier label names for the different attribution methods
METH2LABEL = {
    "context_cite_256": "ContextCite (256 calls)",
    "context_cite_128": "ContextCite (128 calls)",
    "context_cite_64": "ContextCite (64 calls)",
    "context_cite_32": "ContextCite (32 calls)",
    "semantic_similarity": "Similarity",
    "leave_one_out": "Leave-one-out",
    "nli_post_hoc_naive": "NLI (window size: 1)",
    "nli_post_hoc_sliding_window_3": "NLI (window size: 3)",
    "nli_post_hoc_sliding_window_5": "NLI (window size: 5)",
    "nli_post_hoc_greedy_sampling": "NLI greedy sampling",
    "llm_post_hoc": "LLM post-hoc (Llama-3.1-8B-Instruct)",
    "longcite_llm_direct": "LLM direct attribution (LongCite-8B)",
}

# get attribution method from the label name
LABEL2METH = {
    "Contributive attribution (ContextCite)": "context_cite_256",
    "Direct attribution (LongCite-8B)": "longcite_llm_direct",
    "Post-hoc attribution (NLI: DeBERTa-v3)": "nli_post_hoc_sliding_window_3",
    "Post-hoc attribution (Semantic similarity)": "semantic_similarity",
}

def process_text(text):
    """From https://github.com/THUDM/LongCite/blob/main/demo.py"""
    special_char={
        '&': '&amp;',
        '\'': '&apos;',
        '"': '&quot;',
        '<': '&lt;',
        '>': '&gt;',
        '\n': '<br>',
        '$': '&#36;',
    }
    for x, y in special_char.items():
        text = text.replace(x, y)
    return text

def get_mapping_spans2cite_strs(results):

    cite_strs = []
    for sc in results["statements"]:
        citation = sc["citation"]

        sent_cite_strs = []
        if citation:
            spans = [c["span"] for c in citation]
            for span in spans:
                start, end = span
                if start == end:
                    cite_str = f"[{start}]"
                else:
                    cite_str = f"[{start}-{end}]"
                sent_cite_strs.append(cite_str)

        cite_strs.extend(sent_cite_strs)

    span2citestr = dict.fromkeys(cite_strs) 
    for i, key in enumerate(span2citestr, start=1):
        span2citestr[key] = f"[{i}]"

    return span2citestr

HF_REPO = "jonaspeschel/fact-checking-attribution-results"

@st.cache_data
def load_json_from_hub(path_in_repo):
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=path_in_repo,
        repo_type="dataset",
        token=st.secrets.get("HF_TOKEN"),
    )
    return load_json(local_path)


# hand-picked feature examples to display as the first examples in the app
FEATURED_EXAMPLE_IDXS = [14, 15, 11, 22, 19, 12, 0, 5, 18, 34]

def get_order_with_featured_examples(n_claims, featured_example_idxs):
    return featured_example_idxs + [i for i in range(n_claims) if i not in featured_example_idxs]

@st.dialog("How to read this page", width="large")
def show_explanations():
    st.markdown(f"""
    This app shows the outputs of a RAG pipeline for the task of automated fact-checking, where an LLM predicts the veracity of claims based on retrieved evidence. Different attribution methods are used to attribute the LLM-generated statements to individual evidence snippets. 
                
    **What is shown**
    - **Left**: the given claim, the predicted and ground-truth verdicts, and a model-generated justification for its verdict (the claims are taken from the [AVeriTeC dataset](https://arxiv.org/abs/2305.13117))
    - **Right**: the evidence snippets cited for each statement with the source URL, and corresponding evaluation metrics

    **Evaluation metrics**
    - **Citation correctness**: measures whether the content of the cited sources *factually supports* a generated statement
        - **Citation precision**: per citation; whether the citation is <span style='color: {RELEVANT_COLOR};'>relevant</span> or <span style='color: {IRRELEVANT_COLOR};'>irrelevant</span> to its corresponding statement (shown as colors of the citation markers)
        - **Citation recall**: per statement; whether the content of all cited sources combined <span style='background-color: {SUPPORTED_COLOR};'>fully supports</span>, <span style='background-color: {PARTIAL_SUPPORT_COLOR};'>partially supports</span>, or <span style='background-color: {UNSUPPORTED_COLOR};'>does not support</span> the statement (shown as background color of the statement)
    - **Citation faithfulness**: whether the LLM actually *used* the cited content to generate the statement (*causal* relationship: how much less likely is the model to generate the same statement when the cited snippets are removed from its context)

    Citation correctness is measured via LLM-as-a-judge using DeepSeek-V3. Citation faithfulness is measured as the *top-k log-probability drop* introduced by the [ContextCite paper](https://arxiv.org/abs/2409.00729).

    **Settings**
    - **Evidence sources**: whether the LLM used the gold evidence provided by the AVeriTeC dataset or custom retrieved web evidence (see [here](https://huggingface.co/datasets/jonaspeschel/AVeriTeC-with-custom-web-evidence) for retrieval setup and results)
    - **Attribution methods**: different attribution methods were used to attribute the LLM-generated statements
        - **Contributive attribution**: using [ContextCite](https://arxiv.org/abs/2409.00729) to cite the evidence snippets that most likely *caused* the LLM to generate its answer by ablating different parts of the evidence context and fitting a sparse linear model to estimate how important each part of the evidence is
        - **Direct attribution**: use an LLM fine-tuned for citation generation ([LongCite-8B](https://arxiv.org/abs/2409.02897)) to generate the citations directly together with the answer       
        - **Post-hoc attribution**: attribute a pre-generated answer to supporting evidence snippets using either an NLI model or simply cosine similarity                  
    """, unsafe_allow_html=True)