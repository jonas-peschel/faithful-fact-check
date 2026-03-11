import json 

def load_json(filepath):

    with open(filepath) as f:
        data = json.load(f)

    return data

def get_attr_method(path: str):
    """Extract the attribution method name from the results filepath name."""

    attr_method_names = ["context_cite_32", "context_cite_64", "context_cite_128", "context_cite_256", "semantic_similarity", "leave_one_out", "nli_post_hoc_naive", 
     "nli_post_hoc_sliding_window_3", "nli_post_hoc_sliding_window_5", "nli_post_hoc_greedy_sampling", "llm_post_hoc", "longcite_llm_direct"]
    
    for attr_method_name in attr_method_names:
        if attr_method_name in path:
            return attr_method_name
    

