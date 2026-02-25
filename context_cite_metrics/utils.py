import json
from datasets import load_dataset
import numpy as np
import torch
from copy import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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


def split_text(text):
    """Split model response into sentences and return sentences and corresponding start indices."""

    sentences = sent_tokenize(text)

    prev_end_idx = 0
    start_idxs, end_idxs = [], []
    for sent in sentences:

        start_idx = text.find(sent, prev_end_idx)
        start_idxs.append(start_idx)
        prev_end_idx = start_idx + len(sent)
        end_idxs.append(prev_end_idx)

    return sentences, start_idxs, end_idxs

#--- dataset helper methods ---#
def load_data(dataset_name, n_samples, seed=0):

    assert n_samples <= 1000, "Max. 1000 samples"

    # Dataset 1: CNN DailyMail
    if dataset_name == "cnn_daily_mail":
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")   # TODO: should better use validation split like in ContextCite paper for next runs

        # sample max 1000 samples and take the first n_samples
        # that way, results from different runs with different n_samples will use the same datapoints in the beginning
        np.random.seed(seed)
        idxs = np.random.choice(len(dataset), 1000, replace=False)
        idxs = idxs[:n_samples]
        dataset_sampled = dataset.select(idxs)

    # Dataset 2: DRUID
    if dataset_name == "druid":
        dataset = load_dataset("copenlu/druid", "DRUID", split="train")  # there is only a train split for this dataset

        # for calculating ContextCite metrics only use examples where the evidence is sufficient and where verdict is True or False
        dataset = dataset.filter(lambda example: (example["evidence_stance"] == "supports" or example["evidence_stance"] == "refutes") and (example["factcheck_verdict"] == "False" or example["factcheck_verdict"] == "True"))

        # use only instances where the context is not extremly short (at least 5 sentences), otherwise the LDS score will probably be quite biased
        dataset = dataset.filter(lambda example: len(sent_tokenize(example["evidence"])) >= 5)

        # sample max 1000 samples and take the first n_samples
        # that way, results from different runs with different n_samples will use the same datapoints in the beginning
        np.random.seed(seed)
        idxs = np.random.choice(len(dataset), 1000, replace=False)
        idxs = idxs[:n_samples]
        dataset_sampled = dataset.select(idxs)
    
    return dataset_sampled

def load_datapoint(datapoint, dataset_name):
    """Load context and query from a datapoint depending on the given dataset."""

    # Dataset 1: CNN DailyMail
    if dataset_name == "cnn_daily_mail":

        context = datapoint["article"]
        query = "Please summarize the article in up to three sentences."

    # Dataset 2: DRUID
    if dataset_name == "druid":

        context = datapoint["evidence"]

        # fact-checking query + claim
        query = "You are an expert fact-checker. You are provided with a claim and related evidence. Based only on the provided evidence, determine if the given claim is either supported or refuted."
        query += " Write a paragraph that justifies your decision and the reasons why you decided to classify the claim in the way that you did."
        query += f"\n\nClaim: {datapoint["claim"]}"

    return context, query

def load_cc_prompt_template(dataset_name):

    # Dataset 1: CNN DailyMail
    if dataset_name == "cnn_daily_mail":
        return "Context: {context}\n\nQuery: {query}"
    
    # Dataset 2: DRUID
    if dataset_name == "druid":
        return "Query: {query}\n\nEvidence: {context}"

#--- dataset helper methods end ---#

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

def load_model(model_name, is_quantize):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if is_quantize else None

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left" # set padding side to left for batch inference with ContextCite
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=torch.bfloat16, 
                                                 quantization_config=quantization_config, trust_remote_code=True) 
    device = model.device

    return model, tokenizer, device

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
    }

def order_results(mean_results, std_results, labels):

    true_order = ["context_cite_256", "context_cite_128", "context_cite_64", "context_cite_32", 
                  "semantic_similarity", "leave_one_out", "nli_post_hoc_naive", "nli_post_hoc_sliding_window_3", 
                  "nli_post_hoc_sliding_window_5", "nli_post_hoc_greedy_sampling", "llm_post_hoc"]

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