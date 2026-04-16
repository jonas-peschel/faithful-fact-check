import argparse
import numpy as np
from typing import List 
from transformers import AutoTokenizer
from context_cite.context_partitioner import SimpleContextPartitioner, BaseContextPartitioner
from utils import load_json, save_json, load_data, load_datapoint
from longcite_utils import LongCiteContextPartitioner

def parse_args():
    parser = argparse.ArgumentParser(description="Utils script for formatting the ContextCite metrics results (for single model and attribution method but for possibly multiple datasets) for citation & correctness evaluation.")
    parser.add_argument("--results_paths", type=str, default=None, nargs="+", help="Paths to the ContextCite metrics results files.")
    parser.add_argument("--attr_method", type=str, choices=["context_cite_32", "context_cite_64", "context_cite_128", "context_cite_256", "semantic_similarity", "leave_one_out", "nli_post_hoc_naive", "nli_post_hoc_sliding_window_3", "nli_post_hoc_sliding_window_5", "nli_post_hoc_greedy_sampling", "llm_post_hoc", "longcite_llm_direct"], required=True, help="Which answer attribution methods to use.")
    parser.add_argument("--use_longcite", action="store_true", help="Whether to use answer statements and context partioning from LongCite.")
    parser.add_argument("--save_file_name", type=str, default=None, help="Extension to the save file name.")

    return parser.parse_args()

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

def format_results(results_paths, attr_method, use_longcite, save_file_name):
    """Format the results from ContextCite metrics calculations for a single attribution method but potentially for
    multiple different datasets to be suitable for citation quality and correctness evaluation. 
    Save the formatted results in scores_cite/ folder.
    """

    formatted_data_point_results = []
    for results_path in results_paths:
        # load ContextCiter to use its response_dict property for context and citation formatting (if use_longcite) 
        # else use the cc.sources and build the cited contexts manually
        results = load_json(results_path)
        dataset = results["metadata"]["dataset"]
        model_name = results["metadata"]["model"]
        n_samples = len(results["results"])
        data = load_data(dataset_name=dataset, n_samples=n_samples, start_idx=0)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = "left" # set padding side to left for batch inference with ContextCite
        tokenizer.pad_token_id = tokenizer.eos_token_id

        for data_point_results, data_point in zip(results["results"], data):
            idx = data_point_results["instance_idx"]
            context, query = load_datapoint(data_point, dataset, use_longcite) # depends on the given dataset
            partitioner = LongCiteContextPartitioner(context=context) if use_longcite else SimpleContextPartitioner(context=context)

            formatted_data_point_result = {
                "idx": idx,
                "dataset": dataset,
                "query": query,
                "prediction": data_point_results["model_answer"],
                "answer": data_point.get("answer"),  # reference answer for MultiFieldQA-en, else None
                "few_shot_scores": None,
            }

            # add claim veracity label for fact-checking datasets
            if (dataset == "averitec" or dataset == "averitec_short_ans" 
                or dataset == "averitec_web_evidence" or dataset == "averitec_web_evidence_short_ans"):
                formatted_data_point_result["claim"] = data_point.get("claim")
                formatted_data_point_result["label"] = data_point.get("label")
                formatted_data_point_result["pred_label"] = None
                formatted_data_point_result["justification"] = data_point.get("justification")

            formatted_data_point_result["decomposed_model_answer"] = data_point_results["decomposed_model_answer"]

            answer_statements = data_point_results["answer_statements"]
            citations = data_point_results["methods"][attr_method]["citations"]
            citation_scores = data_point_results["methods"][attr_method].get("citation_spans_scores")

            statements = []
            if citation_scores:
                for statement, citations_sent, citation_scores_sent in zip(answer_statements, citations, citation_scores):
                    citation_texts = []
                    for cite_span, score in zip(merge_citation_spans(citations_sent), citation_scores_sent):
                        citation_text, pre_context, post_context = get_citation_text(cite_span, partitioner)
                        citation_texts.append({
                            "cite": citation_text, 
                            "span": (cite_span[0], cite_span[-1]), 
                            "score": score,
                            "pre_context": pre_context,
                            "post_context": post_context,
                        }) 
                    statements.append({"statement": statement, "citation": citation_texts})
                formatted_data_point_result["statements"] = statements
                formatted_data_point_results.append(formatted_data_point_result)
            else:
                for statement, citations_sent in zip(answer_statements, citations):
                    citation_texts = []
                    for cite_span in merge_citation_spans(citations_sent):
                        citation_text, pre_context, post_context = get_citation_text(cite_span, partitioner)
                        citation_texts.append({
                            "cite": citation_text, 
                            "span": (cite_span[0], cite_span[-1]), 
                            "score": None,
                            "pre_context": pre_context,
                            "post_context": post_context,
                        }) 
                    statements.append({"statement": statement, "citation": citation_texts})
                formatted_data_point_result["statements"] = statements
                formatted_data_point_results.append(formatted_data_point_result)

    # save formatted data
    save_name = f"results_formatted/{model_name.split("/")[-1]}_{attr_method}"
    save_path = f"{save_name}_{save_file_name}.json" if save_file_name else f"{save_name}.json"
    save_json(save_path, formatted_data_point_results)
    print(f"Saved results to: {save_path}")

def main(config=None):

    if config is None:
        config = parse_args() 

    format_results(config.results_paths, config.attr_method, config.use_longcite, config.save_file_name)
 
if __name__ == "__main__":
    main()