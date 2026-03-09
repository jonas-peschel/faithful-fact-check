import argparse 
from utils import load_json, save_json
from copy import copy
from transformers import AutoTokenizer
from tqdm.auto import tqdm 
from pathlib import Path 

def parse_args():
    parser = argparse.ArgumentParser(description="Merge evaluation results for answer correctness, citation quality, and citation length into one file.")
    parser.add_argument("--cite_file_path", type=str, default=None, help="Filepath with citation quality results.")
    parser.add_argument("--correct_file_path", type=str, default=None, help="Filepath with answer correctness results.")

    return parser.parse_args() 

def count_citation_lengths(results):
    "Count lengths (number of tokens) of citations. Compute average per statement and per citation (i.e. cited span)"

    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)  # use same tokenizer as in LongCite & SelfCite paper 
    n_statements, n_spans = 0, 0
    total_cite_len = 0

    for res in tqdm(results):
        for sc in res["statements"]:
            if sc["citation"]:
                n_statements += 1
                for c in sc["citation"]:
                    n_spans += 1 
                    total_cite_len += len(tokenizer.encode(c["cite"], add_special_tokens=False))

    citation_lengths = {
        "avg_len_per_statement": total_cite_len/n_statements,
        "avg_len_per_span": total_cite_len/n_spans,
    }   
    return citation_lengths

def main(config=None):

    if config is None:
        config = parse_args() 

    cite_results_file = load_json(config.cite_file_path)
    correct_results_file = load_json(config.correct_file_path)
    # extract overall scores from the end of the file
    cite_results, cite_scores = cite_results_file[:-1], cite_results_file[-1]
    correct_results, correct_scores = correct_results_file[:-1], correct_results_file[-1]

    dataset = cite_results[0]["dataset"]
    results = []
    for cite_res, correct_res in zip(cite_results, correct_results):
        assert cite_res["idx"] == correct_res["idx"]
        assert cite_res["dataset"] == correct_res["dataset"] == dataset

        res = copy(cite_res)
        res["pred_label"] = correct_res["pred_label"]

        results.append(res)

    # count citation lengths
    citation_length_scores = count_citation_lengths(results)

    # add overall results
    results.append({
        "eval_metrics": {
            "answer_correctness": correct_scores,
            "citation_quality": cite_scores, 
            "citation length": citation_length_scores,
        }
    })

    # save merged results 
    save_path = "./results_final/" + Path(config.cite_file_path).name 
    save_json(save_path, results)


if __name__ == "__main__":
    main()