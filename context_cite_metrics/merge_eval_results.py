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
    """Count lengths (number of tokens) of citations. Compute average per statement and per citation (i.e. cited span).
    Count number of citations (all individual source sentences, not just spans) per statement. 
    Citation length is averaged over number of statements with at least one citation to make it comparable to the results
    from LongCite & SelfCite papers. Number of citations is averaged over all statements (with or without citations).
    """

    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)  # use same tokenizer as in LongCite & SelfCite paper 
    n_statements, n_statements_with_citation, n_spans = 0, 0, 0
    total_cite_len = 0
    n_citations = 0

    for res in tqdm(results):
        for sc in res["statements"]:
            n_statements += 1
            if sc["citation"]:
                n_statements_with_citation += 1
                for c in sc["citation"]:
                    n_spans += 1
                    n_citations += c["span"][-1] - c["span"][0] + 1
                    total_cite_len += len(tokenizer.encode(c["cite"], add_special_tokens=False))

    citation_lengths = {
        "avg_len_per_statement": total_cite_len/n_statements_with_citation,
        "avg_len_per_span": total_cite_len/n_spans,
        "avg_len_per_statement_all": total_cite_len/n_statements, 
        "avg_num_citations_per_statement": n_citations/n_statements_with_citation, 
        "avg_num_citations_per_span": n_citations/n_spans, 
        "avg_num_citations_per_statement_all": n_citations/n_statements,
    }   
    return citation_lengths

def count_answer_lengths(results):
    """Count lengths of model answers (number of tokens and number of statements)"""

    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)  # use same tokenizer as in LongCite & SelfCite paper 
    n_statements = 0 
    total_ans_length = 0

    for res in tqdm(results):
        for sc in res["statements"]:
            n_statements += 1
            total_ans_length += len(tokenizer.encode(sc["statement"], add_special_tokens=False))

    answer_lengths = {
        "avg_num_statements": n_statements/len(results),
        "avg_answer_length": total_ans_length/len(results),
    }
    return answer_lengths

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

    # count citation lengths and answer lengths
    citation_lengths = count_citation_lengths(results)
    answer_lengths = count_answer_lengths(results)

    # add overall results
    results.append({
        "eval_metrics": {
            "answer_correctness": correct_scores,
            "citation_quality": cite_scores, 
            "citation_length": citation_lengths,
            "answer_length": answer_lengths,
        }
    })

    # save merged results 
    save_path = "./results_final/" + Path(config.cite_file_path).name 
    save_json(save_path, results)
    print(f"Saved results to: {save_path}")

if __name__ == "__main__":
    main()