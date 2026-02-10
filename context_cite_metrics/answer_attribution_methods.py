import argparse
import os
import json
from pathlib import Path
from datasets import load_dataset
import torch
import numpy as np 
from dotenv import load_dotenv 
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from context_cite import ContextCiter
from tqdm.auto import tqdm
import nltk
from nltk import sent_tokenize
nltk.download("punkt_tab")
# from context_cite.utils import _get_response_logit_probs

def parse_args():

    parser = argparse.ArgumentParser(description="Calculate answer attribution scores using different attribution methods.")
    parser.add_argument("--dataset", type=str, choices=["cnn_daily_mail"], default=None, required=True, help="Which dataset to use.")
    parser.add_argument("--model_name", type=str, choices=["meta-llama/Llama-3.1-8B-Instruct"], default="meta-llama/Llama-3.1-8B-Instruct", help="Huggingface name of model to use.")
    parser.add_argument("--attr_methods", type=str, nargs="+", choices=["context_cite"], required=True, help="Which answer attribution methods to use.")
    parser.add_argument("--cc_num_ablations", type=int, nargs="+", choices=[32, 64, 128, 256], help="How many ablations to use if ContextCite is used as attribution method.")
    parser.add_argument("--results_path", type=str, default="Results/results.json", help="Path to the file where attribution scores and experiment results are stored.")
    parser.add_argument("--n_samples", type=int, default=20, help="How many data points to sample from the dataset.")

    return parser.parse_args()

def load_json(filepath):

    with open(filepath) as f:
        data = json.load(f)

    return data

def save_json(filepath, content):

    with open(filepath, "w") as f:
        json.dump(content, f, indent=4)

def load_data(dataset_name, n_samples, seed=0):

    # Dataset 1: CNN Daily Mail
    if dataset_name == "cnn_daily_mail":
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")

        # sample
        np.random.seed(seed)
        idxs = np.random.choice(len(dataset), n_samples, replace=False)
        dataset_sampled = dataset.select(idxs)

    return dataset_sampled

def load_datapoint(datapoint, dataset_name):
    """Load context and query from a datapoint depending on the given dataset."""

    # Dataset 1: CNN Daily Mail
    if dataset_name == "cnn_daily_mail":

        context = datapoint["article"]
        query = "Please summarize the article in up to three sentences."

    return context, query

def load_cc_prompt_template(dataset_name):

    # Dataset 1: CNN Daily Mail
    if dataset_name == "cnn_daily_mail":
        return "Context: {context}\n\nQuery: {query}"


def load_model(model_name, is_quantize):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ) if is_quantize else None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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

#--- Answer Attribution Methods ---#
def compute_attributions_context_cite(cc: ContextCiter, res: dict):
    """Compute attributions for each sentence in the model answer and write it to the results dict."""

    answer = cc.response
    if "model_answer" in res.keys():
        assert answer == res["model_answer"], "Model answer must be always identical."

    sentences, start_idxs, end_idxs = split_text(answer)
    attr_scores = []

    for start_idx, end_idx in zip(start_idxs, end_idxs):

        attr_scores_sent = cc.get_attributions(start_idx=start_idx, end_idx=end_idx, as_dataframe=False)
        attr_scores.append(attr_scores_sent)

    #--- write results to results dict ---#
    if "query" not in res.keys():
        res["query"] = cc.query
    if "model_answer" not in res.keys():
        res["model_answer"] = answer
    if "num_sources" not in res.keys():
        res["num_sources"] = len(cc.sources)
    if "num_answer_sentences" not in res.keys():
        res["num_answer_sentences"] = len(sentences)
    # attribution scores
    if "methods" not in res.keys():
        res["methods"] = {}
    
    # add attribution scores for the specific method (always overwrite if there were existing ones)
    method_name = f"context_cite_{cc.num_ablations}"    # for context cite include the number of ablations in the method name
    res["methods"][method_name] = {
        "attr_scores": attr_scores,
    }

    return res
#--- Answer Attribution Methods ---#


def main(config=None):

    if config is None:
        config = parse_args()

    # load or make results file
    results_path = Path(config.results_path)
    if not results_path.exists():

        # make new results file and save metadata
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.touch()
        results = {}

        results["metadata"] = {
            "dataset": config.dataset,
            "model": config.model_name,
        }
        results["results"] = []
    else:

        # load existing results file to append the new results to
        results = load_json(results_path)

        # check that the old experiment used the same dataset and model 
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

        # check if other results for this data point exist already, if not add new entry
        if len(results["results"]) > idx:   # exists already
            data_point_results = results["results"][idx]    # copy old results to extend/overwrite
        else:  
            # make new entry 
            data_point_results = {  
                "instance_idx": idx,
            }
            results["results"].append(data_point_results)

        context, query = load_datapoint(data_point, config.dataset) # depends on the given dataset

        # for ContextCite as attribution method we need a need ContextCiter instance per cc_num_ablations setting
        # then we can use simply the last instantiated ContextCiter for the other attribution methods calculations
        # if ContextCite is not specified in the arguments to use as attribution method, we have to instantiate a
        # single ContextCiter object (num_ablations does not matter there)

        if "context_cite" in config.attr_methods:

            if len(config.cc_num_ablations) == 0:   # check that at least one option for num_ablations is given
                raise ValueError("Number(s) of ablations must be specified when using ContextCite for attribution.")
            
            for n_ablations in config.cc_num_ablations: # instantiate new ContextCiter for each num_ablations

                cc = ContextCiter(
                    model = model,
                    tokenizer = tokenizer,
                    context = context,
                    query = query,
                    num_ablations = n_ablations,
                    prompt_template = CC_PROMPT_TEMPLATE,
                    generate_kwargs = CC_GENERATE_KWARGS,
                )

                # perform answer attributions with ContextCite method
                data_point_results = compute_attributions_context_cite(cc, data_point_results)

        # instantiate one ContextCiter if we don't use ContextCite for attribution
        else:   

            cc = ContextCiter(
                model = model,
                tokenizer = tokenizer,
                context = context,
                query = query,
                prompt_template = CC_PROMPT_TEMPLATE,
                generate_kwargs = CC_GENERATE_KWARGS,
            )

        # perform remaining methods for answer attribution
        # TODO


        # add results for the data point to the results
        results["results"][idx] = data_point_results

    # save results to result file
    save_json(results_path, results)

if __name__ == "__main__":
    main()