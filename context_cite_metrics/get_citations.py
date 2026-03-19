import argparse 
from utils import load_json, save_json
import numpy as np
from scipy import special 
from tqdm.auto import tqdm
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Convert attribution scores into discrete citations using thresholding and filtering.")
    parser.add_argument("--results_paths", type=str, default=None, nargs="+", help="Paths to the ContextCite metrics results files.")
    parser.add_argument("--attr_methods", type=str, default=None, nargs="+", choices=["context_cite_32", "context_cite_64", 
        "context_cite_128", "context_cite_256", "semantic_similarity", "nli_post_hoc_naive", "nli_post_hoc_sliding_window_3", "nli_post_hoc_sliding_window_5"])

    return parser.parse_args() 

def get_params(attr_method):

    if attr_method in ["context_cite_32", "context_cite_64", "context_cite_128", "context_cite_256"]:
        params = {
            "t": 1.5,
            "p": 0.7, 
            "k": 4, 
            "n": math.inf,
        }
    elif attr_method == "semantic_similarity":
        params = {
            "t": 0.7,
            "p": 0.7, 
            "k": 4, 
            "n": math.inf,
        }
    elif attr_method == "nli_post_hoc_naive":
        params = {
            "t": 0.65,
            "p": 0.7, 
            "k": 4, 
            "n": math.inf,
        }
    elif attr_method == "nli_post_hoc_sliding_window_3":
        params = {
            "t": 0.75,
            "p": 0.3, 
            "k": 2, 
            "n": 5,
        }
    elif attr_method == "nli_post_hoc_sliding_window_5":
        params = {
            "t": 0.75,
            "p": 0.3, 
            "k": 2, 
            "n": 5,
        }

    return params

def filter_and_thresh_citations(scores, t, p, k, n):
    """Extract citations from ContextCite attribution scores according to the
    method described in SelfCite paper Appendix B.2
    """
    scores = np.array(scores)
    idxs = np.array(range(len(scores)))

    # 1. filtering
    mask = scores >= t
    if not mask.any():
        return [], []
    idxs = idxs[mask] 

    # 2. merging adjacent scores 
    spans = []
    i = 0
    while(i < len(idxs)):
        if i == 0:
            span = [idxs[0]]
        else:
            if idxs[i-1] == idxs[i] - 1 and len(span) < n:
                span.append(idxs[i])
            else:
                spans.append(span)
                span = [idxs[i]]
        i += 1
        if i == len(idxs):
            spans.append(span)

    for i, span in enumerate(spans):
        spans[i] = np.array(span)

    merged_scores = np.array([np.max(scores[span]) for span in spans])

    # 3. softmax normalization + added max normalization before to make softmax less sensitive to the scale of the scores
    norm_scores = special.softmax(merged_scores / np.max(merged_scores))

    # 4. top-p selection 
    order = np.argsort(norm_scores)[::-1]
    spans = [spans[i] for i in order]
    norm_scores = norm_scores[order]

    selected_spans = []
    summed_scores = 0
    for span, score in zip(spans, norm_scores):
        selected_spans.append(span)
        summed_scores += score 
        if summed_scores >= p:
            break 

    # 5. top-k filtering 
    selected_spans = selected_spans[:k]

    # return citations as list
    citations = []
    for span in selected_spans:
        citations.extend(span.tolist())

    # additionally return corresponding attribution scores (score per span)
    citation_scores = merged_scores[order][:len(selected_spans)].tolist() 

    return citations, citation_scores

def get_citations(results_paths, attr_methods):

    for results_path in results_paths:
        results = load_json(results_path)
        for i, data_point_results in tqdm(enumerate(results["results"]), total=len(results["results"])):
            for attr_method in attr_methods:
                attr_scores = data_point_results["methods"][attr_method]["attr_scores"]

                # different thresholding and filtering parameters for different attribution methods
                filter_and_thresh_params = get_params(attr_method)
                citations, citation_scores = [], []
                for attr_scores_sent in attr_scores:
                    c, cs = filter_and_thresh_citations(attr_scores_sent, **filter_and_thresh_params) 
                    citations.append(c) 
                    citation_scores.append(cs)

                data_point_results["methods"][attr_method]["citations"] = citations
                data_point_results["methods"][attr_method]["citation_spans_scores"] = citation_scores
            results["results"][i] = data_point_results
        save_json(results_path, results)

def main(config=None):

    if config is None:
        config = parse_args()

    get_citations(config.results_paths, config.attr_methods)


if __name__ == "__main__":
    main()