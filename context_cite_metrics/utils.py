import json
from datasets import load_dataset
import numpy as np
import torch
from copy import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
from context_cite import ContextCiter 
from longcite_utils import LongCiteContextCiter
import nltk
from nltk import sent_tokenize
nltk.download("punkt_tab")

# json loading and saving
def load_json(filepath):

    with open(filepath) as f:
        data = json.load(f)

    return data

def save_json(filepath, content):

    with open(filepath, "w") as f:
        json.dump(content, f, indent=4)

def split_model_answer(cc: ContextCiter):
    """Split model response into statements/sentences and get corresponding start and end indices.
    Different behaviour for standard ContextCiter vs. LongCite ContextCiter.
    """
    response = cc.response
    if isinstance(cc, LongCiteContextCiter):
        longcite_result = cc.response_dict 
        statements, start_idxs, end_idxs = [], [], []
        prev_end_idx = 0
        for statement in longcite_result["all_statements"]:
            statement_text = statement["statement"]
            if statement_text.strip():
                statements.append(statement_text)
                start_idx = response.find(statement_text, prev_end_idx)
                start_idxs.append(start_idx)
                prev_end_idx = start_idx + len(statement_text)
                end_idxs.append(prev_end_idx)
        spans = list(zip(start_idxs, end_idxs))
        return statements, spans
    else:
        sentences = sent_tokenize(response)
        start_idxs, end_idxs = [], []
        prev_end_idx = 0
        for sent in sentences:
            start_idx = response.find(sent, prev_end_idx)
            start_idxs.append(start_idx)
            prev_end_idx = start_idx + len(sent)
            end_idxs.append(prev_end_idx)
        spans = list(zip(start_idxs, end_idxs))
        return sentences, spans

#--- dataset helper methods ---#
def load_data(dataset_name, n_samples=-1, start_idx=0, seed=0):

    assert(n_samples <= 1_000), "Max. 1,000 samples."

    # Dataset 1: CNN DailyMail
    if dataset_name == "cnn_daily_mail":
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")   # TODO: should better use validation split like in ContextCite paper for next runs

    # Dataset 2: DRUID
    if dataset_name == "druid":
        dataset = load_dataset("copenlu/druid", "DRUID", split="train")  # there is only a train split for this dataset
        if n_samples == -1:
            n_samples = len(dataset)

        # for calculating ContextCite metrics only use examples where the evidence is sufficient and where verdict is True or False
        dataset = dataset.filter(lambda example: (example["evidence_stance"] == "supports" or example["evidence_stance"] == "refutes") and (example["factcheck_verdict"] == "False" or example["factcheck_verdict"] == "True"))

        # use only instances where the context is not extremly short (at least 5 sentences), otherwise the LDS score will probably be quite biased
        dataset = dataset.filter(lambda example: len(sent_tokenize(example["evidence"])) >= 5)

    # Dataset 3: AVeriTeC with ground truth evidence
    if dataset_name == "averitec" or dataset_name == "averitec_short_ans":
        dataset = load_dataset("jonaspeschel/AVeriTeC-with-scraped-gold-evidence", split="train")

    # Dataset 4: MultiFieldQA-en
    if dataset_name == "multifieldqa_en":
        dataset = load_dataset("jonaspeschel/MultiFieldQA-en-capped-context", split="train")
        
    if n_samples == -1:
        n_samples = len(dataset)

    # sample max 1000 samples and take the first n_samples
    # that way, results from different runs with different n_samples will use the same datapoints in the beginning
    np.random.seed(seed)
    idxs = np.random.choice(len(dataset), min(1000, len(dataset)), replace=False)
    idxs = idxs[start_idx:start_idx+n_samples]
    dataset_sampled = dataset.select(idxs)
   
    return dataset_sampled

def load_datapoint(datapoint, dataset_name, use_longcite):
    """Load context and query from a datapoint depending on the given dataset."""

    # Dataset 1: CNN DailyMail
    if dataset_name == "cnn_daily_mail":
        context = datapoint["article"]
        if use_longcite:
            query = "Please summarize the article in up to three statements."
        else:
            query = "Please summarize the article in up to three sentences."

    # Dataset 2: DRUID
    if dataset_name == "druid":
        context = datapoint["evidence"]

        # fact-checking query + claim
        query = "You are an expert fact-checker. You are provided with a claim and related evidence. Based only on the provided evidence, determine if the given claim is either supported or refuted."
        query += " Write a paragraph that justifies your decision and the reasons why you decided to classify the claim in the way that you did."
        query += f"\n\nClaim: {datapoint["claim"]}"

    # Dataset 3: AVeriTeC
    if dataset_name == "averitec":
        context = "\n\n".join(datapoint["scraped_evidences"])

        # fact-checking query + claim
        query = "You are an expert fact-checker. You are provided with a claim and related evidence. Based only on the provided evidence, determine if the given claim is either supported, refuted, has conflicting evidence, or has not enough evidence to determine its veracity. Write a paragraph of about 4-6 sentences that states your verdict and the main reason for it, referencing and synthesizing the most relevant pieces of evidence that support your verdict. Do not copy sentences from the evidence verbatim. Always paraphrase and synthesize the evidence in your own words."
        query += f"\n\nClaim: {datapoint["claim"]}"

    if dataset_name == "averitec_short_ans":
        context = "\n\n".join(datapoint["scraped_evidences"])

        ## fact-checking query + claim
        # # medium length prompt
        # query = "You are an expert fact-checker. You are provided with a claim and related evidence. Based only on the provided evidence, determine if the given claim is either supported, refuted, has conflicting evidence, or has not enough evidence to determine its veracity. Write a single concise statement of maximally 2-3 sentences. Your answer should state your verdict and the main reason for it, referencing and synthesizing the most relevant pieces of evidence that support your verdict. Do not copy sentences from the evidence verbatim. Always paraphrase and synthesize the evidence in your own words."
        # short length prompt
        query = "You are an expert fact-checker. You are provided with a claim and related evidence. Based only on the provided evidence, determine if the given claim is either supported, refuted, has conflicting evidence, or has not enough evidence to determine its veracity. Write a single sentence that states your verdict and the main reason for it, referencing and synthesizing the most relevant pieces of evidence that support your verdict. Do not copy sentences from the evidence verbatim. Always paraphrase and synthesize the evidence in your own words."
        query += f"\n\nClaim: {datapoint["claim"]}"

    # Dataset 4: MultiFieldQA-en
    if dataset_name == "multifieldqa_en":
        context = datapoint["context"]
        query = datapoint["query"]

    return context, query

def load_cc_prompt_template(dataset_name):

    # Dataset 1: CNN DailyMail
    if dataset_name == "cnn_daily_mail":
        return "Context: {context}\n\nQuery: {query}"
    
    # Dataset 2: DRUID
    if dataset_name == "druid":
        return "Query: {query}\n\nEvidence: {context}"

    # Dataset 3: AVeriTeC
    if dataset_name == "averitec" or dataset_name == "averitec_short_ans":
        return "Query: {query}\n\nEvidence: {context}"
    
    # Dataset 4: MultiFieldQA-en
    if dataset_name == "multifieldqa_en":
        return "Context: {context}\n\nQuery: {query}"

#--- dataset helper methods end ---#

def load_model(model_name, is_quantize, use_model=True):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if is_quantize else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left" # set padding side to left for batch inference with ContextCite
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_model:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16, 
                                                    attn_implementation="flash_attention_2", quantization_config=quantization_config, 
                                                    trust_remote_code=True) 
    else:
        model = None

    # check if flash attention is used
    print("\n\n##### Flash Attention #####")
    print(f"Is available: {is_flash_attn_2_available()}")
    print(f"Model attention implementation: {model.config._attn_implementation}")
    print(f"Attention class: {type(model.model.layers[0].self_attn)}")
    print("##### Flash Attention #####\n\n")

    return model, tokenizer, device


CC_GENERATE_KWARGS = {"do_sample": False, "max_new_tokens": 512}

#--- atttribution methods helper functions ---#
def get_nli_entailment_probs(nli_tokenizer, nli_model, dataloader):

    entailment_probs = []
    for answer_sent, context_sents in dataloader:

        input = nli_tokenizer(
            list(context_sents),
            list(answer_sent),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(nli_model.device)

        with torch.inference_mode():
            output = nli_model(**input)

        probs = torch.softmax(output.logits, axis=1)   # convert to probabilities
        entailment_prob = probs[:,0]    # only use predicted prob for entailment
        entailment_probs.append(entailment_prob.cpu())

    entailment_probs = np.concat(entailment_probs, axis=0)  # concatenate results from all batches

    return entailment_probs
#--- atttribution methods helper functions end---#

#--- Plotting utils ---#

# plot colors for the different attribution methods
METH2COL = {
    "context_cite_256": '#1f77b4',
    "context_cite_128": '#5392c0',
    "context_cite_64": '#78add2',
    "context_cite_32": '#accde5',
    "semantic_similarity": '#2ca02c',
    "leave_one_out": '#9467bd',
    "nli_post_hoc_naive": '#d3d3d3',
    "nli_post_hoc_sliding_window_3": '#7f7f7f',
    "nli_post_hoc_sliding_window_5": "#484848",
    "nli_post_hoc_greedy_sampling": '#333333',
    "llm_post_hoc": '#bcbd22',
    "longcite_llm_direct": '#e377c2',
}

# prettier label names for the different attribution methods
METH2LABEL = {
    "context_cite_256": "ContextCite (256 calls)",
    "context_cite_128": "ContextCite (128 calls)",
    "context_cite_64": "ContextCite (64 calls)",
    "context_cite_32": "ContextCite (32 calls)",
    "semantic_similarity": "Semantic similarity",
    "leave_one_out": "Leave-one-out",
    "nli_post_hoc_naive": "NLI (window size: 1)",
    "nli_post_hoc_sliding_window_3": "NLI (window size: 3)",
    "nli_post_hoc_sliding_window_5": "NLI (window size: 5)",
    "nli_post_hoc_greedy_sampling": "NLI greedy sampling",
    "llm_post_hoc": "LLM post-hoc (Llama-3.1-8B-Instruct)",
    "longcite_llm_direct": "LLM direct attribution (LongCite-8B)",
}

# prettier label names for the different datasets
DATASET2LABEL = {
    "cnn_daily_mail": "CNN DailyMail",
    "averitec": "AVeriTeC (gold evidence)",
    "averitec_short_ans": "AVeriTeC (gold evidence, short answers)",
    "averitec_web_evidence": "AVeriTeC (web evidence)", 
    "averitec_web_evidence_short_ans": "AVeriTeC (web evidence, short answers)",
}

def order_results(mean_results, std_results, labels):

    true_order = ["context_cite_256", "context_cite_128", "context_cite_64", "context_cite_32", "longcite_llm_direct", 
                  "llm_post_hoc", "semantic_similarity", "leave_one_out", "nli_post_hoc_naive", 
                  "nli_post_hoc_sliding_window_3", "nli_post_hoc_sliding_window_5", "nli_post_hoc_greedy_sampling"]

    adapted_true_order = copy(true_order)
    for method in true_order:
        if method not in labels:
            adapted_true_order.remove(method)

    idxs = []
    for method in adapted_true_order:
        idxs.append(labels.index(method))

    ordered_mean = mean_results[idxs]
    ordered_std = std_results[idxs]
    ordered_labels = [labels[idx] for idx in idxs]

    return ordered_mean, ordered_std, ordered_labels