import argparse
import numpy as np
from transformers import AutoTokenizer
from context_cite import ContextCiter
from context_cite.context_partitioner import SimpleContextPartitioner
from utils import load_json, save_json, load_data, load_datapoint, load_cc_prompt_template, CC_GENERATE_KWARGS
from longcite_utils import LongCiteContextCiter, LongCiteContextPartitioner, LONGCITE_GENERATE_KWARGS, LONGCITE_PROMPT_TEMPLATE

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

def get_citations(res, attr_method, use_longcite):
    """Use same number of citations as LongCite model generated for
    LLM-based post-hoc attribution and NLI-based post-hoc greedy sampling method.
    """

    if attr_method == "llm_post_hoc":
        llm_citations = res["methods"][attr_method]["citations"]
        if use_longcite:
            longcite_citations = res["methods"]["longcite_llm_direct"]["citations"]
            citations = []
            for llm_citations_sent, longcite_citations_sent in zip(llm_citations, longcite_citations):
                k_longcite = len(longcite_citations_sent)
                if k_longcite == 0:
                    citations.append([])
                else:
                    citations_sent = llm_citations_sent[str(k_longcite)]
                    if citations_sent:
                        citations.append(citations_sent)
                    else:
                        citations.append([])
        else:
            raise Exception("Can't determine how many sources to cite without LongCite reference.")
    elif attr_method == "nli_post_hoc_greedy_sampling":
        greedy_sampling_citations = res["methods"][attr_method]["citations"]
        if use_longcite:
            longcite_citations = res["methods"]["longcite_llm_direct"]["citations"]
            citations = []
            for greedy_sampling_citations_sent, longcite_citations_sent in zip(greedy_sampling_citations, longcite_citations):
                k_longcite = len(longcite_citations_sent)
                citations.append(greedy_sampling_citations_sent[:k_longcite])
        else:
            raise Exception("Can't determine how many sources to cite without LongCite reference.")
    else:
        citations = res["methods"][attr_method]["citations"]

    return citations

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
        PROMPT_TEMPLATE = LONGCITE_PROMPT_TEMPLATE if use_longcite else load_cc_prompt_template(dataset)  # LongCite prompt template is not actually used
        GENERATE_KWARGS = LONGCITE_GENERATE_KWARGS if use_longcite else CC_GENERATE_KWARGS

        for data_point_results, data_point in zip(results["results"], data):
            idx = data_point_results["instance_idx"]
            context, query = load_datapoint(data_point, dataset, use_longcite) # depends on the given dataset
            partitioner = LongCiteContextPartitioner(context=context) if use_longcite else SimpleContextPartitioner(context=context)

            cc_kwargs = {
                "model": None,  # don't need model inference here
                "tokenizer": tokenizer,
                "context": context,
                "query": query,
                "prompt_template": PROMPT_TEMPLATE,
                "generate_kwargs": GENERATE_KWARGS,
                "partitioner": partitioner,
            }
            cc = LongCiteContextCiter(**cc_kwargs) if use_longcite else ContextCiter(**cc_kwargs)

            formatted_data_point_result = {
                "idx": idx,
                "dataset": dataset,
                "query": query,
                "prediction": data_point_results["model_answer"],
                "decomposed_model_answer": data_point_results["decomposed_model_answer"],
                "answer": data_point.get("answer"),  # reference answer for MultiFieldQA-en, else None
                "few_shot_scores": None,
            }

            # add claim veracity label for fact-checking datasets
            if dataset == "averitec":
                formatted_data_point_result["claim"] = data_point.get("claim")
                formatted_data_point_result["label"] = data_point.get("label")
                formatted_data_point_result["pred_label"] = None
                formatted_data_point_result["justification"] = data_point.get("justification")

            answer_statements = data_point_results["answer_statements"]
            citations = data_point_results["methods"][attr_method]["citations"]
            citation_scores = data_point_results["methods"][attr_method].get("citation_spans_scores")

            statements = []
            if citation_scores:
                for statement, citations_sent, citation_scores_sent in zip(answer_statements, citations, citation_scores):
                    citation_texts = []
                    for cite_span, score in zip(merge_citation_spans(citations_sent), citation_scores_sent):
                        mask = np.zeros(len(cc.sources), dtype=bool)
                        mask[cite_span] = True 
                        citation_text = partitioner.get_context(mask)
                        citation_texts.append({"cite": citation_text, "span": (cite_span[0], cite_span[-1]), "score": score}) 
                    statements.append({"statement": statement, "citation": citation_texts})
                formatted_data_point_result["statements"] = statements
                formatted_data_point_results.append(formatted_data_point_result)
            else:
                for statement, citations_sent in zip(answer_statements, citations):
                    citation_texts = []
                    for cite_span in merge_citation_spans(citations_sent):
                        mask = np.zeros(len(cc.sources), dtype=bool)
                        mask[cite_span] = True 
                        citation_text = partitioner.get_context(mask)
                        citation_texts.append({"cite": citation_text, "span": (cite_span[0], cite_span[-1]), "score": None}) 
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