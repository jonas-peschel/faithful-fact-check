import argparse
from tqdm.auto import tqdm
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from utils import load_json, save_json, load_data, load_datapoint, load_model
from longcite_utils import LongCiteContextPartitioner
import time
from typing import List 
import numpy as np 
import torch
from pathlib import Path
from context_cite.context_partitioner import BaseContextPartitioner, SimpleContextPartitioner

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment for verdict verification using pruned context.")
    parser.add_argument("--metrics_results_path", type=str, help="Path to the file where attribution scores and experiment results (metrics) are stored.")
    parser.add_argument("--verification_results_path", type=str, help="Path to the file where verdict verification experiment results are stored.")
    parser.add_argument("--use_longcite", action="store_true", help="Whether to use ContextPartitioner from LongCite model.")
    parser.add_argument("--attr_method", type=str, choices=["context_cite_32", "context_cite_64", "context_cite_128", "context_cite_256", "semantic_similarity", "leave_one_out", "nli_post_hoc_naive", "nli_post_hoc_sliding_window_3", "nli_post_hoc_sliding_window_5", "nli_post_hoc_greedy_sampling", "llm_post_hoc", "longcite_llm_direct"], help="Which attribution method to use.")
    parser.add_argument("--model", type=str, choices=["Llama-3.1-8B-Instruct", "DeepSeek"])
    return parser.parse_args()

CLASS_NAMES = ["Supported", "Conflicting Evidence/Cherrypicking", "Refuted"]

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
        except Exception as e:
            if attempt == max_retries-1:
                raise

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

    label_ids = []
    for label in CLASS_NAMES:
        label_ids.append(tokenizer.encode(label, add_special_tokens=False)[0])
    return label_ids 

def get_pred_distribution_hf_model(response, label_ids: List):

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

def build_evidence_context(citations: List[int], partitioner: BaseContextPartitioner):

    citations_full_answer = []
    for citations_sent in citations:
        citations_full_answer.extend(citations_sent)
    citations_full_answer = list(set(citations_full_answer))

    evidence_snippets = []
    for cite_span in merge_citation_spans(citations_full_answer):
        citation_text, pre_context, post_context = get_citation_text(cite_span, partitioner)
        evidence_snippets.append(citation_text)

    return "\n\n".join(evidence_snippets)

def get_prompts(claim: str, evidence: str):

    sys_prompt = "You are an expert fact-checker. You are provided with a claim and corresponding evidence snippets. Based only on the provided evidence snippets, classify the veracity of the given claim, i.e., whether the claim is supported, refuted, or has conflicting evidence. Answer only with exactly one of the three possible classes: 'Supported', 'Conflicting Evidence', 'Refuted'. IMPORTANT: Do not respond with any additional text and use only the provided evidence snippets to come up with your answer. Do not use your own knowledge or any other external sources than the ones provided."
    user_prompt = f"Classify the veracity of the following claim using the provided evidence.\nClaim: {claim}\n\nEvidence snippets:\n{evidence}"

    return sys_prompt, user_prompt


def main(config=None):

    if config is None:
        config = parse_args() 

    load_dotenv()

    # load data
    metrics_results = load_json(config.metrics_results_path)
    dataset = metrics_results["metadata"]["dataset"]
    n_samples = len(metrics_results["results"])
    data = load_data(dataset_name=dataset, n_samples=n_samples, start_idx=0)

    if config.model == "Llama-3.1-8B-Instruct":
        # load model
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model, tokenizer, device = load_model(model_name, True)
        model = model.eval()

        # get token IDs for labels (class names)
        label_ids = get_label_ids_hf_model(tokenizer)

    elif config.model == "DeepSeek":
        # TODO
        model_name = "deepseek-chat"

    # TODO: add ks as cli argument later
    ks = ["all", "cite"]

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

    for data_point_metrics_results, data_point in tqdm(zip(metrics_results["results"], data), total=len(data), desc="Claims"):

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
            y_gt = np.zeros(len(CLASS_NAMES)).tolist()
            y_gt[CLASS_NAMES.index(label)] = 1.0
        data_point_verification_results["class_distributions"] = {
            "ground_truth": y_gt
        }

        for k in ks:
            if k == "all":
                evidence = partitioner.get_context()
            elif k == "cite":
                # use discrete citations to give as context
                citations = data_point_metrics_results["methods"][config.attr_method]["citations"]
                evidence = build_evidence_context(citations, partitioner)

            # get prompts & perform model inference
            sys_prompt, user_prompt = get_prompts(claim, evidence)

            if model_name == "meta-llama/Llama-3.1-8B-Instruct":
                response = query_hf_model(model, tokenizer, device, sys_prompt, user_prompt)
                y_preds = get_pred_distribution_hf_model(response, label_ids)
            elif model_name == "deepseek-chat":
                # TODO
                response = query_api_model(...)

            # save the results (i.e., the predicted distribution)
            data_point_verification_results["class_distributions"][f"k={k}"] = y_preds.tolist()

        # save every iteration
        save_json(config.verification_results_path, verification_results)

if __name__ == "__main__": 
    main()
