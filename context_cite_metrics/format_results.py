import argparse
from pathlib import Path 
from transformers import AutoTokenizer
from context_cite import ContextCiter 
from utils import load_json, save_json, load_data, load_datapoint, load_cc_prompt_template, CC_GENERATE_KWARGS
from longcite_utils import LongCiteContextCiter, LongCiteContextPartitioner, LONGCITE_GENERATE_KWARGS, LONGCITE_PROMPT_TEMPLATE

def parse_args():
    parser = argparse.ArgumentParser(description="Utils script for formatting the ContextCite metrics results (for single model and attribution method but for possibly multiple datasets) for citation & correctness evaluation.")
    parser.add_argument("--results_paths", type=str, default=None, nargs="+", help="Paths to the ContextCite metrics results files.")
    parser.add_argument("--attr_method", type=str, choices=["longcite_llm_direct"])
    parser.add_argument("--use_longcite", action="store_true", help="Whether to use answer statements and context partioning from LongCite.")

    return parser.parse_args()

def format_results(results_paths, attr_method, use_longcite):
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
            partitioner = LongCiteContextPartitioner(context=context) if use_longcite else None

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
            
            if use_longcite:
                # if use_longcite we can set the full model output and use the LongCite results dict
                # we have to set the citations in the full model output according to the citations for
                # the corresponding attribution method (or leave it as it was if attr_method is LongCite)
                if attr_method == "longcite_llm_direct":
                    cc._cache["output"] = data_point_results["model_output_full"]
                else:
                    pass  # TODO

                longcite_results = cc.response_dict
                formatted_data_point_result = {
                    "idx": idx,
                    "dataset": dataset,
                    "query": query,
                    "prediction": cc._cache["output"],
                    "answer": None,
                    "few_shot_scores": None,
                }
                # format statements with citations
                statements =[]
                for statement in longcite_results["all_statements"]:
                    if not statement["statement"].strip():
                        continue
                    citation = []
                    if statement["citation"]:
                        for citation_dict in statement["citation"]:
                            citation.append({"cite": citation_dict["cite"]})
                    statements.append({"statement": statement["statement"], "citation": citation})
                formatted_data_point_result["statements"] = statements 

            else:
                pass  # TODO maybe

            formatted_data_point_results.append(formatted_data_point_result)

    # save formatted data
    save_path = Path(f"results_formatted/{model_name.split("/")[-1]}_{attr_method}.json")
    save_json(save_path, formatted_data_point_results)

def main(config=None):

    if config is None:
        config = parse_args() 

    format_results(config.results_paths, config.attr_method, config.use_longcite)
 
if __name__ == "__main__":
    main()