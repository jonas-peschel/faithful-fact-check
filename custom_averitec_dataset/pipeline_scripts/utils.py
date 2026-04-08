import json
import numpy as np 
from rank_bm25 import BM25Okapi 
from nltk import word_tokenize 
from typing import List 

# json loading and saving
def load_json(filepath):

    with open(filepath) as f:
        data = json.load(f)

    return data

def save_json(filepath, content):

    with open(filepath, "w") as f:
        json.dump(content, f, indent=4)

# BM25 for evidence ranking
class BM25:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus 
        tok_corpus = [word_tokenize(paragraph) for paragraph in corpus]
        self.retriever = BM25Okapi(tok_corpus)
    
    def get_scores(self, query: str | List[str]):
        if isinstance(query, str):
            tok_query = word_tokenize(query) 
            scores = self.retriever.get_scores(tok_query)
        elif isinstance(query, List):
            scores = []
            for q in query:
                tok_query = word_tokenize(q)
                scores.append(self.retriever.get_scores(tok_query))
            scores = np.array(scores)
            scores = np.max(scores, axis=0)

        return scores