import argparse
import os
from tqdm.auto import tqdm
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, BadRequestError
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from utils import load_json, save_json, load_data, load_datapoint, load_model
from longcite_utils import LongCiteContextPartitioner
import time
from typing import List, Union
import numpy as np 
import torch
from pathlib import Path
from context_cite.context_partitioner import BaseContextPartitioner, SimpleContextPartitioner

def int_or_str(value: str):
    try:
        return int(value)
    except ValueError:
        return value

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment for verdict verification using pruned context.")
    parser.add_argument("--metrics_results_path", type=str, help="Path to the file where attribution scores and experiment results (metrics) are stored.")
    parser.add_argument("--verification_results_path", type=str, help="Path to the file where verdict verification experiment results are stored.")
    parser.add_argument("--pred_labels_results_path", type=str, help="Path to the file where the originally predicted verdicts are stored.")
    parser.add_argument("--use_longcite", action="store_true", help="Whether to use ContextPartitioner from LongCite model.")
    parser.add_argument("--attr_methods", type=str, nargs="+", choices=["context_cite_32", "context_cite_64", "context_cite_128", "context_cite_256", "semantic_similarity", "leave_one_out", "nli_post_hoc_naive", "nli_post_hoc_sliding_window_3", "nli_post_hoc_sliding_window_5", "nli_post_hoc_greedy_sampling", "llm_post_hoc", "longcite_llm_direct"], help="Which attribution method to use.")
    parser.add_argument("--ks", type=int_or_str, nargs="+", help="Numbers k of how many source sentences to give the model for verification.")
    parser.add_argument("--model", type=str, choices=["Llama-3.1-8B-Instruct", "DeepSeek"])
    parser.add_argument("--add_additional_context", action="store_true", help="Whether to add additional context around the originally cited content.")
    parser.add_argument("--start_idx", type=int, default=0, help="Claim to start with")
    parser.add_argument("--end_idx", type=int, default=None, help="Claim to end with")
    return parser.parse_args()

LABELS = ["Supported", "Conflicting Evidence/Cherrypicking", "Refuted", "Not Enough Evidence"]
PRED_CLASS_NAMES = ["Supported", "Conflicting Evidence/Cherrypicking", "Refuted", "Insufficient Evidence"]

def query_api_model(client, model, sys_prompt, user_prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            # get model response with structured output
            response = client.chat.completions.create(
                model=model,
                temperature=1.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )
            return response
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if attempt == max_retries-1:
                raise 
            time.sleep(5)
        except BadRequestError:
            return None 
        except Exception as e:
            if attempt == max_retries-1:
                raise

def get_label_tokens_api_model():
    """First tokens of class names for deepseek-chat model."""

    verdict_label_tokens = ["Supported", "Conf", "Ref", "In"]

    return verdict_label_tokens

def get_pred_distribution_api_model(response, label_tokens: List[str]):

    logprobs = response.choices[0].logprobs.content[0].top_logprobs
    logprobs_dict = {entry.token: entry.logprob for entry in logprobs}
    pred_logprobs = [logprobs_dict.get(token, -np.inf) for token in label_tokens]
    pred_label_distribution = torch.softmax(torch.tensor(pred_logprobs), axis=0)
    return pred_label_distribution 

def query_hf_model(model, tokenizer, device, sys_prompt, user_prompt):

    messages=[
        {
            "role": "system",
            "content": sys_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]
    input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.inference_mode():
        output = model(**input)

    return output

def get_label_ids_hf_model(tokenizer):

    verdict_label_ids = []
    for label in PRED_CLASS_NAMES:
        verdict_label_ids.append(tokenizer.encode(label, add_special_tokens=False)[0])

    return verdict_label_ids

def get_pred_distribution_hf_model(response, label_ids: List[int]):

    pred_logits = response.logits.cpu().squeeze()[-1,:]
    pred_label_logits = pred_logits[label_ids]
    pred_label_distribution = torch.softmax(pred_label_logits, axis=0)
    return pred_label_distribution

def merge_citation_spans(citation_idxs):
    spans = []
    i = 0
    while(i < len(citation_idxs)):
        if i == 0:
            span = [citation_idxs[0]]
        else:
            if citation_idxs[i-1] == citation_idxs[i] - 1:
                span.append(citation_idxs[i])
            else:
                spans.append(span)
                span = [citation_idxs[i]]
        i += 1
        if i == len(citation_idxs):
            spans.append(span)
    return spans

def get_citation_text(cite_span: List[int], partitioner: BaseContextPartitioner):
    """Get corresponding citation source text for given citation indices.
    Get 2 sentences precedent and subsequent to the cited text to display 
    as additional context.
    """
    n = 2  # number of context sentences
    cite_mask, pre_mask, post_mask = (np.zeros(partitioner.num_sources, dtype=bool), 
                                      np.zeros(partitioner.num_sources, dtype=bool), 
                                      np.zeros(partitioner.num_sources, dtype=bool))
    
    pre_span = list(range(max(0, cite_span[0]-n), cite_span[0]))
    post_span = list(range(cite_span[-1]+1, min(partitioner.num_sources-1, cite_span[-1]+1+n)))

    cite_mask[cite_span] = True 
    pre_mask[pre_span] = True 
    post_mask[post_span] = True 

    citation_text = partitioner.get_context(cite_mask)
    pre_text = partitioner.get_context(pre_mask)
    post_text = partitioner.get_context(post_mask)

    return citation_text, pre_text, post_text

def build_evidence_context(citations: List[int], partitioner: BaseContextPartitioner, add_additional_context: bool):

    citations_full_answer = []
    for citations_sent in citations:
        citations_full_answer.extend(citations_sent)
    citations_full_answer = list(set(citations_full_answer))

    evidence_snippets = []
    for cite_span in merge_citation_spans(citations_full_answer):
        citation_text, pre_context, post_context = get_citation_text(cite_span, partitioner)
        if add_additional_context:
            evidence_snippet = " ".join([pre_context.split("\n\n")[-1], citation_text, post_context.split("\n\n")[0]])  # add pre- and post-context (split at paragraphs)
        else:
            evidence_snippet = citation_text  # use cited passage only
        evidence_snippets.append(evidence_snippet)

    return "\n\n".join(evidence_snippets)

def get_verification_prompts(claim: str, pred_label: str, evidence: str):

    # rename "Not Enough Evidence" into "Insufficient Evidence"
    if pred_label == "Not Enough Evidence":
        pred_label = "Insufficient Evidence"
        
    sys_prompt = "You are an expert fact-checker. You are provided with a claim, a verdict for the claim's veracity from another fact-checker, and corresponding evidence snippets that were used to arrive at the given verdict. Based only on the provided evidence snippets, verify the verdict from the other fact-checker by classifying the veracity of the given claim yourself and by assessing whether you agree or disagree with the given verdict. You must classify the claim by reasoning about whether the claim is either supported, refuted, has conflicting evidence, or has insufficient evidence to classify the veracity. Answer only with exactly one of the four possible verdicts: 'Supported', 'Conflicting Evidence', 'Refuted', 'Insufficient Evidence'. The verdicts have the following meanings:\nSupported: the claim is mostly supported by the information in the evidence snippets\nConflicting Evidence: there is both substantial supporting and refuting information in the evidence snippets\nRefuted: the claim is mostly refuted by the information in the evidence snippets\nInsufficient Evidence: the evidence snippets are completely irrelevant to the claim and contain absolutely no information that allows to classify the claim's veracity\n\nFor verifying the verdict from the other fact-checker you should think about whether the verdict aligns with the information in the provided evidence snippets. If not, you can provide a different verdict than the given one that you find more suitable given the evidence snippets. Please use the 'Insufficient Evidence' verdict sparingly and only if you really can not provide any of the other verdicts. IMPORTANT: Do not respond with any additional text and use only the provided evidence snippets to come up with your answer. Do not use your own knowledge or any other external sources than the ones provided."
    user_prompt = f"Verify the verdict from the other fact-checker for the following claim using the provided evidence snippets.\nClaim: {claim}\nVerdict from the other fact-checker: {pred_label}\n\nEvidence snippets:\n{evidence}\n\nYour own verdict:"

    return sys_prompt, user_prompt

def get_verification_preds(
        attr_method: str, 
        k: str | int, 
        partitioner: BaseContextPartitioner,
        add_additional_context: bool, 
        data_point_metrics_results: dict, 
        claim: str, 
        pred_label: str, 
        model_name: str, 
        model: PreTrainedModel = None, 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None, 
        device: str = None, 
        verdict_label_ids: List[int] = None, 
        client: OpenAI = None, 
        verdict_label_tokens: List[str] = None
    ):

    if k == "all":
        evidence = partitioner.get_context()
    elif k == "cite":
        # use discrete citations to give as context
        citations = data_point_metrics_results["methods"][attr_method]["citations"]
        evidence = build_evidence_context(citations, partitioner, add_additional_context)

    # get prompts & perform model inference
    sys_prompt_veri, user_prompt_veri = get_verification_prompts(claim, pred_label, evidence)

    if model_name == "meta-llama/Llama-3.1-8B-Instruct":

        response_veri = query_hf_model(model, tokenizer, device, sys_prompt_veri, user_prompt_veri)
        verdict_pred_dist = get_pred_distribution_hf_model(response_veri, verdict_label_ids)

    elif model_name == "deepseek-chat":
        
        response_veri = query_api_model(client, model_name, sys_prompt_veri, user_prompt_veri)
        if not response_veri: # e.g. BadRequestError
            return None
        verdict_pred_dist = get_pred_distribution_api_model(response_veri, verdict_label_tokens)

    return verdict_pred_dist


def main(config=None):

    if config is None:
        config = parse_args() 

    load_dotenv()

    # load data
    metrics_results = load_json(config.metrics_results_path)
    dataset = metrics_results["metadata"]["dataset"]
    n_samples = len(metrics_results["results"])
    data = load_data(dataset_name=dataset, n_samples=n_samples, start_idx=0).select(range(config.start_idx, config.end_idx)).to_list()
    pred_labels_results = load_json(config.pred_labels_results_path)[:-1]
    pred_labels = [res["pred_label"] for res in pred_labels_results]

    if config.model == "Llama-3.1-8B-Instruct":
        # load model
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model, tokenizer, device = load_model(model_name, True)
        model = model.eval()

        # get (first) token IDs for labels (class names) and yes/no
        verdict_label_ids = get_label_ids_hf_model(tokenizer)

    elif config.model == "DeepSeek":

        model_name = "deepseek-chat"
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

        # get (first) tokens for labels (class names) and yes/no
        verdict_label_tokens = get_label_tokens_api_model()

    # load or make verification results file
    verification_results_path = Path(config.verification_results_path)
    if not verification_results_path.exists():
        # make new results file and save metadata
        verification_results_path.parent.mkdir(parents=True, exist_ok=True)
        verification_results_path.touch()
        verification_results = {}

        verification_results["metadata"] = {
            "dataset": dataset,
            "model": model_name,
        }
        verification_results["results"] = []
    else:
        # load existing results file to append the new results to
        verification_results = load_json(verification_results_path)

        # check that the old experiment used the same dataset and model
        assert verification_results["metadata"]["dataset"] == dataset, "Existing results should come from the same dataset as new results to compute."
        assert verification_results["metadata"]["model"] == model_name, "Existing results should use the same model as new results to compute."

    for data_point_metrics_results, data_point, pred_label in tqdm(zip(metrics_results["results"][config.start_idx:config.end_idx], data, 
                                                                       pred_labels[config.start_idx:config.end_idx]), total=len(data), desc="Claims"):

        idx = data_point_metrics_results["instance_idx"]
        idxs = np.array([res["instance_idx"] for res in verification_results["results"]])  # indices for already computed results
        if idx in idxs:  # exists already
            data_point_verification_results = verification_results["results"][(idx == idxs).nonzero()[0].item()]  # copy old results to extend/overwrite
        else:
            # make new entry
            data_point_verification_results = {
                "instance_idx": idx,
            }
            verification_results["results"].append(data_point_verification_results)

        claim = data_point["claim"]
        label = data_point["label"]
        data_point_verification_results["label"] = label

        context, _ = load_datapoint(data_point, dataset, config.use_longcite) # depends on the given dataset
        partitioner = LongCiteContextPartitioner(context=context) if config.use_longcite else SimpleContextPartitioner(context=context)

        # ground truth distribution
        if label == "Not Enough Evidence":
            y_gt = None
        else:
            y_gt = np.zeros(len(LABELS[:-1])).tolist()
            y_gt[LABELS[:-1].index(label)] = 1.0
        data_point_verification_results["ground_truth_distribution"] = y_gt
        data_point_verification_results["pred_distributions"] = {}

        # compute predicted distributions for baseline k=all once per claim (does not have to be re-computed for all attribution methods)
        if config.model == "Llama-3.1-8B-Instruct":
            verdict_pred_dist_baseline = get_verification_preds(None, "all", partitioner, config.add_additional_context, data_point_metrics_results, 
                                                                claim, pred_label, model_name, model=model, tokenizer=tokenizer, device=device, 
                                                                verdict_label_ids=verdict_label_ids)
        elif config.model == "DeepSeek":
            verdict_pred_dist_baseline = get_verification_preds(None, "all", partitioner, config.add_additional_context, data_point_metrics_results, 
                                                                claim, pred_label, model_name, client=client, verdict_label_tokens=verdict_label_tokens)

        # compute predicted distributions for all other k and all attribution methods
        for attr_method in config.attr_methods:
            data_point_verification_results["pred_distributions"][attr_method] = {}
            for k in config.ks:
                if k == "all":
                    verdict_pred_dist = verdict_pred_dist_baseline
                else:
                    if config.model == "Llama-3.1-8B-Instruct":
                        verdict_pred_dist = get_verification_preds(attr_method, k, partitioner, config.add_additional_context, data_point_metrics_results, 
                                                                   claim, pred_label, model_name, model=model, tokenizer=tokenizer, device=device, 
                                                                   verdict_label_ids=verdict_label_ids)
                    elif config.model == "DeepSeek":
                        verdict_pred_dist = get_verification_preds(attr_method, k, partitioner, config.add_additional_context, data_point_metrics_results, 
                                                                   claim, pred_label, model_name, client=client, verdict_label_tokens=verdict_label_tokens)
            
                # save the results (i.e., the predicted distribution)
                if verdict_pred_dist is not None:
                    data_point_verification_results["pred_distributions"][attr_method][f"k={k}"] = {
                        "verdict_dist_3_classes": (verdict_pred_dist[:-1]/verdict_pred_dist[:-1].sum()).tolist(),
                        "verdict_dist_4_classes": verdict_pred_dist.tolist(),
                    }
                elif verdict_pred_dist is None:
                    data_point_verification_results["pred_distributions"][attr_method][f"k={k}"] = {
                        "verdict_dist_3_classes": None,
                        "verdict_dist_4_classes": None,
                    }

        # save every claim
        save_json(config.verification_results_path, verification_results)

if __name__ == "__main__": 
    main()
