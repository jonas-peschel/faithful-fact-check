import argparse
import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
import torch
import numpy as np 
import warnings
from scipy.stats import spearmanr, ConstantInputWarning
from dotenv import load_dotenv 
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from context_cite import ContextCiter
from context_cite.utils import _get_response_logit_probs, aggregate_logit_probs
from tqdm.auto import tqdm
import nltk
from nltk import sent_tokenize
nltk.download("punkt_tab")
from typing import List 
from numpy.typing import NDArray

def parse_args():

    parser = argparse.ArgumentParser(description="Calculate top-k log-prob drop (k=1,3,5) and linear datamodeling score for different attribution methods.") 
    parser.add_argument("--attr_methods", type=str, nargs="+", choices=["context_cite_32", "context_cite_64", "context_cite_128", "context_cite_256", "semantic_similarity", "leave_one_out", "nli_post_hoc_naive"], default=None, help="Which answer attribution methods to calculate the metrics for.")
    parser.add_argument("--metrics", type=str, nargs="+", choices=["log_prob_drop", "LDS"], default=["log_prob_drop", "LDS"], help="Which metric(s) to compute.")
    parser.add_argument("--dataset", type=str, choices=["cnn_daily_mail", "druid"], required=True, help="Which dataset to use.")
    parser.add_argument("--model_name", type=str, choices=["meta-llama/Llama-3.1-8B-Instruct"], default="meta-llama/Llama-3.1-8B-Instruct", help="Huggingface name of model to use.")
    parser.add_argument("--results_path", type=str, default="Results/results.json", help="Path to the file where attribution scores and experiment results (metrics) are stored.")
    parser.add_argument("--n_samples", type=int, default=20, help="For how many data points to compute the metrics.")
    parser.add_argument("--m", type=int, default=128, help="How many random ablation vectors to sample for LDS calculation.")
    parser.add_argument("--cc_batch_size", type=int, default=8, help="Batch size to use in ContextCiter for performing inference using ablated contexts.")

    return parser.parse_args()

def load_json(filepath):

    with open(filepath) as f:
        data = json.load(f)

    return data

def save_json(filepath, content):

    with open(filepath, "w") as f:
        json.dump(content, f, indent=4)

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

def load_model(model_name, is_quantize):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if is_quantize else None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" # set padding side to left for batch inference with ContextCite
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, quantization_config=quantization_config) 
    device = model.device

    return model, tokenizer, device

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

#--- Metric Functions ---#
def calc_top_k_log_prob_drop(cc: ContextCiter, res: dict, attr_methods: List[str]):
    """
    Calculate top-k log-probability drop metric for a single data point.
    Calculate metric each for k=1,3,5 and for each sentence in the answer.
    """

    def create_mask(attr_scores: List, k: int) -> NDArray[np.bool_]:
        """Create a boolean mask based on given attribution scores and number of top-sources to ablate (k)."""

        mask = np.ones(len(attr_scores), dtype=np.bool_)

        if k == 0:  # full mask without ablations
            pass
        
        else:
            top_k_idxs = np.argsort(attr_scores)[-k:]
            mask[top_k_idxs] = False

        return mask
    
    def create_dataset(cc: ContextCiter, masks: List[NDArray[np.bool_]]):
        """
        Create "dataset" with input tokens and output tokens (labels) for different ablations.
        Based on ContextCite utils._create_regression_dataset function.
        """

        data_dict = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        response_ids = cc._response_ids # token indices of the complete response
        for mask in masks:
            prompt_ids = cc._get_prompt_ids(mask=mask)  # token indices of the prompt with sources ablated according to the mask
            input_ids = prompt_ids + response_ids

            data_dict["input_ids"].append(input_ids)
            data_dict["attention_mask"].append([1] * len(input_ids))
            data_dict["labels"].append([-100] * len(prompt_ids) + response_ids) # label only for response part

        return Dataset.from_dict(data_dict)
    
    # how many sources to ablate
    ks = [0,1,3,5]

    answer = cc.response
    sentences, start_idxs, end_idxs = split_text(answer)

    # loop through attribution methods and sentences, then for each sentence compute
    # top-k log-prob drop for k=1,3,5
    attr_methods = res["methods"].keys() if attr_methods is None else attr_methods
    for attr_method in attr_methods:

        # prepare results dict
        if "metrics" not in res["methods"][attr_method].keys():
            res["methods"][attr_method]["metrics"] = {}
        if "top_k_drop" not in res["methods"][attr_method]["metrics"].keys():
            res["methods"][attr_method]["metrics"]["top_k_drop"] = {}
        for k in ks[1:]:
            res["methods"][attr_method]["metrics"]["top_k_drop"][f"top_{k}_drop"] = []  # for each method and k, there is a list of drops (drop for each sentence)


        for sent_idx, start_idx, end_idx in zip(range(len(sentences)), start_idxs, end_idxs):

            # 0. load attribution scores
            attr_scores = res["methods"][attr_method]["attr_scores"][sent_idx] 

            # 1. create masks for k=1,3,5 and create one full mask where no sources are ablated (k=0) for computing the difference 
            masks = []
            for k in ks:
                masks.append(create_mask(attr_scores, k))

            # 2. create "dataset" with input tokens for different ablations and output tokens (labels) 
            dataset = create_dataset(cc, masks)

            # 3. calculate logit probabilities for all answer tokens for each of the context ablations
            # shape: (n_masks, n_answer_tokens)
            logit_probs = _get_response_logit_probs(
                dataset, cc.model, cc.tokenizer, len(cc._response_ids), cc.batch_size
            )

            # 4. aggregate logit_probs and convert to log_prob for the current sentence
            ids_start_idx, ids_end_idx = cc._indices_to_token_indices(start_idx, end_idx)
            logit_probs_sent = logit_probs[:, ids_start_idx:ids_end_idx]

            log_probs = []  # final log probabilites for the answer with context ablations according to k=0,1,3,5
            for i in range(len(ks)):
                log_probs.append(aggregate_logit_probs(logit_probs_sent[i:i+1,:], output_type="log_prob"))

            # 5. compute differences with full context, normalize the result by number of answer tokens (in the sentence)
            n_answer_tokens_sent = logit_probs_sent.shape[1]
            for i,k in enumerate(ks[1:], start=1):
                diff = (log_probs[0] - log_probs[i]) / n_answer_tokens_sent # diff full_context - ablated_context

                # write result to the results dict (append to list for sentences)
                res["methods"][attr_method]["metrics"]["top_k_drop"][f"top_{k}_drop"].append(diff.item())
                
    return res

def calc_linear_datamodeling_score(cc: ContextCiter, res: dict, attr_methods: List[str]):
    """Calculate linear datamodeling score metric for each sentence for a single data point."""

    def prepare_results_dict(res: dict, attr_methods: List[str]):
        """Add key-structure to the results dict for saving the LDS scores."""

        for attr_method in attr_methods:
            if "metrics" not in res["methods"][attr_method].keys():
                res["methods"][attr_method]["metrics"] = {}
            res["methods"][attr_method]["metrics"]["LDS"] = []  # list of scores (one LDS score per sentence)

        return res


    answer = cc.response
    sentences, start_idxs, end_idxs = split_text(answer)

    attr_methods = res["methods"].keys() if attr_methods is None else attr_methods
    res = prepare_results_dict(res, attr_methods)

    # get logit probs for each context ablation and answer token
    logit_probs = cc._logit_probs  # (m, n_answer_tokens)
    masks = cc._masks   # (m, n_sources)

    # loop through sentences and attribution methods, then for each sentence compute LDS
    for sent_idx, start_idx, end_idx in zip(range(len(sentences)), start_idxs, end_idxs):

        # aggregate logit_probs for the current sentence to get the true response generation probabilities f
        ids_start_idx, ids_end_idx = cc._indices_to_token_indices(start_idx, end_idx)
        logit_probs_sent = logit_probs[:, ids_start_idx:ids_end_idx]        # (m, n_tokens_sent)
        f = aggregate_logit_probs(logit_probs_sent, output_type="log_prob") # (m,)

        # compare true probabilities with predicted probabilities for each attribution method
        for attr_method in attr_methods:

            # load attribution scores for the current sentence and method
            attr_scores = np.array(res["methods"][attr_method]["attr_scores"][sent_idx])

            # calculate predicted probs by summing the attribution scores corresponding to the non-ablated sources
            f_hat = masks @ attr_scores # (m, n_sources) x (n_sources) -> (m,)

            # compute Spearman rank correlation
            # NOTE: catch warning for when any input is constant, i.e. answer sentence does not depend at all on
            # the context sentences; then set the LDS score to None and ignore it for calculating the mean later
            # (None will be converted to null when saving to json, back to None when loading from json and to np.nan
            # when converting the data to np.array when setting dtype=float where it can then be ignored using np.nanmean()
            # OR the entire sentence could be excluded)
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConstantInputWarning)

                try:
                    LDS = spearmanr(f, f_hat).statistic.item()
                except ConstantInputWarning:
                    LDS = None
    
            # write result to the results dict (append to list for sentences)
            res["methods"][attr_method]["metrics"]["LDS"].append(LDS)

    return res
#--- Metric Functions ---#


def main(config=None):

    if config is None:
        config = parse_args()

    # load results file
    results_path = Path(config.results_path)
    results = load_json(results_path)

    # check to use same dataset and model as for computing the attribution scores
    assert results["metadata"]["dataset"] == config.dataset, "Existing results should come from the same dataset as new results to compute."
    assert results["metadata"]["model"] == config.model_name, "Existing results should use the same language model as new results to compute."

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    # log into huggingface
    login(token=HF_TOKEN)

    # load data and model
    model, tokenizer, device = load_model(config.model_name, True) # load veracity classification and justification model (Llama-8B-Instruct)
    data = load_data(config.dataset, n_samples=config.n_samples, seed=0)

    CC_PROMPT_TEMPLATE = load_cc_prompt_template(config.dataset)
    CC_GENERATE_KWARGS = {"do_sample": False, "max_new_tokens": 512}

    for idx, data_point in tqdm(enumerate(data), total=len(data)):

        data_point_results = results["results"][idx]
        context, query = load_datapoint(data_point, config.dataset) # depends on the given dataset

        # instantiate ContextCiter for probability calculations
        cc = ContextCiter(
            model = model,
            tokenizer = tokenizer,
            context = context,
            query = query,
            num_ablations = config.m,   # number of ablations for LDS calculation; doesn't matter for log-prob drop
            batch_size = config.cc_batch_size,
            prompt_template = CC_PROMPT_TEMPLATE,
            generate_kwargs = CC_GENERATE_KWARGS,
        )

        # check that the same answer was generated
        answer = cc.response
        assert answer == data_point_results["model_answer"], "Model answer must be always identical."

        if "log_prob_drop" in config.metrics:
            # calculate top-k log-probability drop metric
            data_point_results = calc_top_k_log_prob_drop(cc, data_point_results, config.attr_methods)

        if "LDS" in config.metrics:
            # calculate linear datamodeling score metric
            data_point_results = calc_linear_datamodeling_score(cc, data_point_results, config.attr_methods)

        # add results for the data point to the results
        results["results"][idx] = data_point_results

    # save results to result file
    save_json(results_path, results)

if __name__ == "__main__":
    main()