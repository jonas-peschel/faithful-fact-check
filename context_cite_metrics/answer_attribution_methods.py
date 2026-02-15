import argparse
import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from dotenv import load_dotenv 
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from context_cite import ContextCiter
from context_cite.utils import _get_response_logit_probs, aggregate_logit_probs
from tqdm.auto import tqdm
import nltk
from nltk import sent_tokenize
nltk.download("punkt_tab")
from typing import List, Union
from numpy.typing import NDArray

def parse_args():

    parser = argparse.ArgumentParser(description="Calculate answer attribution scores using different attribution methods.")
    parser.add_argument("--dataset", type=str, choices=["cnn_daily_mail", "druid"], default=None, required=True, help="Which dataset to use.")
    parser.add_argument("--model_name", type=str, choices=["meta-llama/Llama-3.1-8B-Instruct"], default="meta-llama/Llama-3.1-8B-Instruct", help="Huggingface name of model to use.")
    parser.add_argument("--attr_methods", type=str, nargs="+", choices=["context_cite", "semantic_similarity", "leave_one_out", "nli_post_hoc_naive"], required=True, help="Which answer attribution methods to use.")
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

        # for calculating ContextCite metrics only use examples where the evidence is sufficient
        dataset = dataset.filter(lambda example: example["evidence_stance"] == "supports" or example["evidence_stance"] == "refutes")

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

        # format claim + evidence as context
        context = f"Claim: {datapoint["claim"]}\n\nEvidence: {datapoint["evidence"]}"

        # fact-checking query
        query = "You are an expert fact-checker. You are provided with a claim and related evidence. Based only on the provided evidence, determine if the given claim is either supported or refuted."
        query += " Write a paragraph that justifies your decision and the reasons why you decided to classify the claim in the way that you did."

    return context, query

def load_cc_prompt_template(dataset_name):

    # Dataset 1: CNN DailyMail
    if dataset_name == "cnn_daily_mail":
        return "Context: {context}\n\nQuery: {query}"
    
    # Dataset 2: DRUID
    if dataset_name == "druid":
        return "Query: {query}\n\n{context}"

#--- dataset helper methods end ---#

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

def prepare_results_dict(cc: ContextCiter, res: dict):
    """Add key-structure to the results dict for saving the attribution scores."""

    #--- write meta information about the data point results ---#
    if "query" not in res.keys():
        res["query"] = cc.query
    if "model_answer" not in res.keys():
        res["model_answer"] = cc.response
    if "num_sources" not in res.keys():
        res["num_sources"] = len(cc.sources)
    if "num_answer_sentences" not in res.keys():
        res["num_answer_sentences"] = len(sent_tokenize(cc.response))

    # attribution scores
    if "methods" not in res.keys():
        res["methods"] = {}

    return res

#--- Answer Attribution Methods ---#
def compute_attributions_context_cite(cc_kwargs: dict, res: dict, cc_num_ablations: List[int]):
    """Compute ContextCite attributions for each given number of ablations."""

    def cc_compute_attributions(cc: ContextCiter, res: dict):
        """Compute attributions for each sentence in the model answer and write it to the results dict."""

        answer = cc.response
        if "model_answer" in res.keys():
            assert answer == res["model_answer"], "Model answer must be always identical."

        sentences, start_idxs, end_idxs = split_text(answer)
        attr_scores = []

        for start_idx, end_idx in zip(start_idxs, end_idxs):

            attr_scores_sent = cc.get_attributions(start_idx=start_idx, end_idx=end_idx, as_dataframe=False)
            attr_scores.append(attr_scores_sent.tolist())   # convert to list for json serialization


        # add attribution scores for the specific method (always overwrite if there were existing ones)
        method_name = f"context_cite_{cc.num_ablations}"    # for context cite include the number of ablations in the method name
        res["methods"][method_name] = {
            "attr_scores": attr_scores,
        }

        return res
    

    if len(cc_num_ablations) == 0:   # check that at least one option for num_ablations is given
        raise ValueError("Number(s) of ablations must be specified when using ContextCite for attribution.")

    for num_ablations in cc_num_ablations: 

        # instantiate new ContextCiter for each num_ablations
        cc = ContextCiter(**cc_kwargs, num_ablations=num_ablations)

        # check that the same answer is generated if there are previous results
        answer = cc.response
        if "model_answer" in res.keys():
            assert answer == res["model_answer"], "Model answer must be always identical."

        # compute attribution scores
        res = cc_compute_attributions(cc, res)

    return res

def compute_attributions_semantic_similarity(cc: ContextCiter, embedding_model: SentenceTransformer, res: dict):
    """
    Compute answer attribution scores based on semantic similarity for each sentence in the model answer.
    The sentences in the model answer and the context are embedded using a sentence embedding model and
    their cosine similarities are used as attribution scores.
    """

    answer_sentences, _, _ = split_text(cc.response)
    context_sentences = cc.sources

    # embed sentences
    answer_embeddings = embedding_model.encode(answer_sentences)
    context_embeddings = embedding_model.encode(context_sentences) 

    # compute cosine similarities for each pair of answer and context sentence
    embedding_model.similarity_fn_name = "cosine" 
    cos_similarities = embedding_model.similarity(answer_embeddings, context_embeddings)    # (n_sentences, n_sources)

    # write to results dict
    res["methods"]["semantic_similarity"] = {
        "attr_scores": cos_similarities.tolist(),
    }

    return res

def compute_attributions_leave_one_out(cc: ContextCiter, res: dict):
    """Compute answer attribution scores based on leave-one-out baseline for each sentence in the model answer."""

    def create_masks(cc: ContextCiter) -> List[NDArray[np.bool_]]:
        """
        Set up boolean masks for calculating the log-prob drop for each context source.
        
        Returns:
            masks (List[NDArray[np.bool_]]):
                List of boolean masks. First element is a full masks with all entries equal to 1.
                For the following entries exactly one entry is set to 0 and all others remain 1;
                one entry per context source. Shape: (n_sources+1, n_sources)
        """

        masks = []
        masks.append(np.ones(len(cc.sources), dtype=np.bool_))  # full mask 
        for i in range(len(cc.sources)):
            mask = np.ones(len(cc.sources), dtype=np.bool_)
            mask[i] = False
            masks.append(mask)

        return masks
    
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
    

    answer = cc.response
    sentences, start_idxs, end_idxs = split_text(answer)

    # create masks: one full mask and one per context source where exactly one source is ablated
    masks = create_masks(cc)

    # create "dataset" with input tokens for the different ablations and output tokens (labels) 
    dataset = create_dataset(cc, masks)

    # calculate logit probabilities for all answer tokens for each of the context ablations
    # shape: (n_masks, n_answer_tokens)
    logit_probs = _get_response_logit_probs(
        dataset, cc.model, cc.tokenizer, len(cc._response_ids), cc.batch_size
    )

    attr_scores = []
    for start_idx, end_idx in zip(start_idxs, end_idxs):

        # aggregate logit_probs and convert to log_prob for the current sentence
        ids_start_idx, ids_end_idx = cc._indices_to_token_indices(start_idx, end_idx)
        logit_probs_sent = logit_probs[:, ids_start_idx:ids_end_idx]

        log_probs = []  # final log probabilites for the answer with context ablations
        for i in range(logit_probs_sent.shape[0]):
            log_probs.append(aggregate_logit_probs(logit_probs_sent[i:i+1,:], output_type="log_prob").item())

        # 5. compute differences with full context, normalize the result by number of answer tokens (in the sentence)
        n_answer_tokens_sent = logit_probs_sent.shape[1]
        log_prob_drops = (np.array(log_probs[0]) - np.array(log_probs[1:])) / n_answer_tokens_sent

        attr_scores.append(log_prob_drops.tolist())


    # write to results dict
    res["methods"]["leave_one_out"] = {
        "attr_scores": attr_scores,
    }

    return res

def compute_attributions_post_hoc_naive(cc: ContextCiter, nli_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
                                        nli_model: PreTrainedModel, res: dict):
    """
    Compute answer attribution scores using post-hoc attribution with the DeBERTa model for natural language 
    inference for each sentence in the model answer. We take as attribution scores the predicted probability
    for entailment from the NLI model using the answer sentence to attribute as premise and each of the 
    context sentences as premise, respectively. This is a naive version because we only consider individual
    source sentences, disregarding that they might need context from adjacent sentences to be understandable.
    """

    class NLIDataset(Dataset):
        """Contains all pairs of sentences between a single answer sentence and all context sentences."""

        def __init__(self, answer_sent: str, context_sents: List[str]):
            self.answer_sent = answer_sent
            self.context_sents = context_sents 

        def __len__(self):
            return len(self.context_sents) 

        def __getitem__(self, idx):
            return self.answer_sent, self.context_sents[idx]


    answer = cc.response
    sentences, _, _ = split_text(answer)

    attr_scores = []
    for sent in sentences:

        # use torch dataloader for batch processing
        dataset = NLIDataset(sent, cc.sources)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=32)

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

        attr_scores.append(entailment_probs.tolist())   # save results as list for json compatibility

    # write to results dict
    res["methods"]["nli_post_hoc_naive"] = {
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

    # load sentence embedding model if semantic similarity is used for attribution
    if "semantic_similarity" in config.attr_methods:
        sentence_embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # load DeBERTa NLI model for post-hoc answer attribution
    if "nli_post_hoc_naive" in config.attr_methods:
        model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

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

        # instantiate ContextCiter to use for answer attribution (used by baselines; for attribution with ContextCite 
        # we have to instantiate new ContextCiter objects with corresponding num_ablations).
        cc_kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "context": context,
                "query": query,
                "prompt_template": CC_PROMPT_TEMPLATE,
                "generate_kwargs": CC_GENERATE_KWARGS,
            }
        cc = ContextCiter(**cc_kwargs)

        # save meta information for the data point and add key structure to the dict to save attribution scores later
        data_point_results = prepare_results_dict(cc, data_point_results)

        #--- perform answer attribution using the different methods ---#
        if "context_cite" in config.attr_methods:

            data_point_results = compute_attributions_context_cite(cc_kwargs, data_point_results, config.cc_num_ablations)

        if "semantic_similarity" in config.attr_methods:

            data_point_results = compute_attributions_semantic_similarity(cc, sentence_embedding_model, data_point_results)

        if "leave_one_out" in config.attr_methods:

            data_point_results = compute_attributions_leave_one_out(cc, data_point_results)

        if "nli_post_hoc_naive" in config.attr_methods:

            data_point_results = compute_attributions_post_hoc_naive(cc, nli_tokenizer, nli_model, data_point_results)


        # add results for the data point to the results
        results["results"][idx] = data_point_results

    # save results to result file
    save_json(results_path, results)

if __name__ == "__main__":
    main()