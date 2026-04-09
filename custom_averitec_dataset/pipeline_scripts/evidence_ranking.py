import argparse 
from dotenv import load_dotenv
from pathlib import Path
import math
import numpy as np
from tqdm.auto import tqdm 
import torch 
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
from mxbai_rerank import MxbaiRerankV2
from sentence_transformers.util import cos_sim
from sklearn.cluster import DBSCAN
from utils import load_json, save_json, BM25 
import nltk 
from nltk import sent_tokenize, word_tokenize
nltk.download("punkt_tab")
from typing import List, Dict
from numpy.typing import NDArray


def parse_args():
    parser = argparse.ArgumentParser(description="Ranking the relevance of retrieved evidence chunks with respect to the individual claims.")
    parser.add_argument("--results_path", type=str, default=None, help="Path to the AVeriTeC data file.")
    parser.add_argument("--store_folder", type=str, default="store_folder", help="Folder path with stored web evidence")
    parser.add_argument("--start_idx", type=int, default=0, help="Claim to start with")
    parser.add_argument("--end_idx", type=int, default=None, help="Claim to end with")
    parser.add_argument("--n_1", type=int, default=50, help="Number of documents to keep after first dense-sparse hybrid ranking step")
    parser.add_argument("--n_2", type=int, default=10, help="Number of documents to keep after second generative re-ranking step")

    return parser.parse_args()

MAX_CHUNK_LEN = 600 
SPLIT_CUTOFF_LEN = 800
DUPLICATE_COS_SIM = 0.85
BATCH_SIZE_EMBEDDING = 256
BATCH_SIZE_RERANKING = 8

def load_paragraphs(dir: Path, n: int):
    """
    Load content from evidence files in the given directory and split
    it into paragraphs. Return a list of lists of paragraphs per file.
    """
    paragraphs = []
    for i in range(1, n+1):
        evidence_file_path = dir / f"search_result_{i}.txt"
        with open(evidence_file_path, "r") as f:
            content = f.read()
            paragraphs.append(content.split("\n\n"))
    return paragraphs

def load_and_chunk_evidence(dir: Path, max_chunk_len: int, split_cutoff_len: int, tokenizer: PreTrainedTokenizerFast):

    def split_sentence(sent, length):

        def split_evenly(n: int, k: int):
            q = n // k
            r = n % k
            return [q+1]*r + [q]*(k-r)

        chunks = []
        n_chunks = math.ceil(length / max_chunk_len)
        words = word_tokenize(sent)
        n_words_per_split = split_evenly(len(words), n_chunks)
        cum_n_words = list(np.cumsum(n_words_per_split))
        for i,j in zip([0]+cum_n_words[:-1], cum_n_words):
            chunks.append([" ".join(words[i:j])])
        return chunks 

    def split_paragraph(text):
        sents = sent_tokenize(text)
        lengths = [len(ids) for ids in tokenizer(sents, add_special_tokens=False)["input_ids"]]
        chunks = []
        curr_chunk = [] 
        curr_len = 0

        for sent, length in zip(sents, lengths):
            # 1. sentence is too long on its own and has to be partitioned
            if length >= split_cutoff_len:
                # append the current chunk
                if curr_chunk:
                    chunks.append(curr_chunk)
                curr_chunk = [] 
                curr_len = 0

                # split the sentence and append the chunks
                sent_chunks = split_sentence(sent, length)
                chunks.extend(sent_chunks)

            # 2. sentence does not fit in the current chunk anymore 
            elif curr_len + length > max_chunk_len:
                # append the current chunk
                if curr_chunk:
                    chunks.append(curr_chunk)
                
                # start new chunk
                curr_chunk = [sent]
                curr_len = length 

            # 3. paragraph fits in the current chunk
            else: 
                curr_chunk.append(sent)
                curr_len += length 

        # append final chunk
        if curr_chunk:
            chunks.append(curr_chunk) 

        # concatenate all sentences per chunk with a single whitespace
        chunk_texts = [[" ".join(chunk)] for chunk in chunks]
        return chunk_texts

    search_infos = load_json(dir / "search_infos.json")

    # load retrieved web content and split into paragraphs (based on double newlines "\n\n")
    paragraphs = load_paragraphs(dir, len(search_infos))

    # count lengths of the paragraphs
    lengths = []
    for ps_doc in paragraphs:
        paragraph_ids = tokenizer(ps_doc, add_special_tokens=False)["input_ids"]
        lengths.append([len(ids) for ids in paragraph_ids])

    # split into chunks
    chunks = []
    infos = []
    for ps_doc, lengths_doc, search_info in zip(paragraphs, lengths, search_infos):
        curr_chunk = [] 
        curr_len = 0 

        for p, length in zip(ps_doc, lengths_doc):
            # 1. paragraph is too long on its own and has to be partitioned
            if length >= split_cutoff_len:
                # append the current chunk
                if curr_chunk:
                    chunks.append(curr_chunk)
                    infos.append(search_info)
                curr_chunk = []
                curr_len = 0

                # split paragraph and append the chunks
                p_chunks = split_paragraph(p)
                chunks.extend(p_chunks)
                infos.extend([search_info] * len(p_chunks))

            # 2. paragraph does not fit in the current chunk anymore
            elif curr_len + length > max_chunk_len:
                # append the current chunk
                if curr_chunk:
                    chunks.append(curr_chunk)
                    infos.append(search_info)
                
                # start new chunk
                curr_chunk = [p]
                curr_len = length 

            # 3. paragraph fits in the current chunk
            else: 
                curr_chunk.append(p)
                curr_len += length 

        # append final chunk
        if curr_chunk:
            chunks.append(curr_chunk) 
            infos.append(search_info)

    # remove all leading and trailing newlines and concatenate all paragraphs per chunk with a single newline
    chunks = [[text.strip("\n ") for text in chunk] for chunk in chunks]
    chunk_texts = ["\n".join(chunk) for chunk in chunks]

    print(f"Successfully loaded {len(chunks)} evidence chunks.")

    return chunk_texts, infos

def dense_sparse_hybrid_ranking(chunks: List[str], queries: List[str], chunks_metadata: List[Dict], embedding_model: SentenceTransformer, n: int):

    def get_ranking(scores: NDArray):
        sorted_idxs = np.argsort(scores)[::-1].copy()
        ranking = np.empty(scores.shape)
        ranking[sorted_idxs] = np.arange(1, scores.shape[0]+1)
        return ranking

    # 1. semantic similarity based dense retrieval 
    query_embds = embedding_model.encode(queries, prompt_name="query", batch_size=BATCH_SIZE_EMBEDDING, 
                                         convert_to_tensor=True, show_progress_bar=True)
    chunk_embds = embedding_model.encode(chunks, batch_size=BATCH_SIZE_EMBEDDING, 
                                         convert_to_tensor=True, show_progress_bar=True)

    sim_matrix = cos_sim(query_embds, chunk_embds)
    similarities = torch.max(sim_matrix, axis=0).values.cpu().numpy()

    # 2. BM25-based sparse retrieval
    bm25 = BM25(chunks)
    scores = bm25.get_scores(queries)

    # 3. combine dense and sparse similarities using RRF
    dense_ranking = get_ranking(similarities)
    sparse_ranking = get_ranking(scores)

    rrf_k = 60  # default that is often used
    rrf_scores = 1/(rrf_k + dense_ranking) + 1/(rrf_k + sparse_ranking)

    # retain only top-n chunks
    top_n_idxs = np.argsort(rrf_scores)[::-1][:n].copy()
    top_n_chunks = [chunks[idx] for idx in top_n_idxs]
    top_n_chunks_metadata = [chunks_metadata[idx] for idx in top_n_idxs]
    embds = chunk_embds[top_n_idxs].cpu().numpy()

    return top_n_chunks, top_n_chunks_metadata, embds

def remove_duplicates(chunk_embds: torch.Tensor, chunks: List[str], chunks_metadata: List[Dict]):
    """Remove duplicate chunks with high cosine similarity"""

    cluster_alg = DBSCAN(eps = 1-DUPLICATE_COS_SIM, metric = "cosine", min_samples = 2)
    clustering = cluster_alg.fit(chunk_embds)

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    unique_idxs = []
    for label, idxs in clusters.items():
        if label == -1:
            unique_idxs.extend(idxs)
        else:
            # choose longest doc in the cluster
            longest_doc_idx = sorted(idxs, key=lambda idx: len(chunks[idx]))[-1]
            unique_idxs.append(longest_doc_idx)

    unique_chunks = [chunks[idx] for idx in unique_idxs]
    unique_chunks_metadata = [chunks_metadata[idx] for idx in unique_idxs]

    print(f"Dropped {len(chunks)-len(unique_idxs)}/{len(chunks)} duplicate chunks.")

    return unique_chunks, unique_chunks_metadata

def generative_reranking(chunks: List[str], queries: List[str], chunks_metadata: List[Dict], reranking_model: MxbaiRerankV2, n: int):
    
    scores = []
    for query in queries:
        results = reranking_model.rank(
            query, 
            chunks, 
            top_k=len(chunks), 
            batch_size=BATCH_SIZE_RERANKING,
            sort=False,
            show_progress=True,
        )
        scores.append([res.score for res in results])
    scores = np.array(scores)
    scores = np.max(scores, axis=0)  # max pooling again

    # retain only top-n chunks
    top_n_idxs = np.argsort(scores)[::-1][:n].copy()
    top_n_chunks = [chunks[idx] for idx in top_n_idxs]
    top_n_chunks_metadata = [chunks_metadata[idx] for idx in top_n_idxs]

    return top_n_chunks, top_n_chunks_metadata

def main(config=None):

    if config is None:
        config = parse_args() 

    load_dotenv()

    results = load_json(config.results_path)
    web_evidence_folder = Path(config.store_folder)

    # load models
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)  # use same tokenizer as in LongCite & SelfCite paper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device=device)
    reranking_model = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-large-v2")  # device is determined automatically
    print(f"DEVICES:\nEmbedding model: {embedding_model.device}\nRe-ranking model: {reranking_model.device}")

    for claim_idx, claim_results in tqdm(enumerate(results[config.start_idx:config.end_idx], start=config.start_idx), 
                                   desc="Claims", total=len(results[config.start_idx:config.end_idx])):
        
        web_evidence_claim_folder =  web_evidence_folder / f"claim_{claim_idx}"

        ## 1. split the retrieved web evidence content into chunks of about 500-800 tokens (preserving paragraphs when possible)
        chunks, chunks_metadata = load_and_chunk_evidence(web_evidence_claim_folder, max_chunk_len=MAX_CHUNK_LEN, 
                                                          split_cutoff_len=SPLIT_CUTOFF_LEN, tokenizer=tokenizer)
        queries = list(set([d["search_string"] for d in chunks_metadata]))

        ## 2. Evidence Ranking
        # 2.1 dense-sparse hybrid ranking: BM25 + semantic similarity combined via reciprocal rank fusion
        top_n1_chunks, chunks_metadata, top_n1_chunk_embds = dense_sparse_hybrid_ranking(chunks, queries, chunks_metadata,  embedding_model, config.n_1)

        # # 2.2 de-duplication
        top_chunks, chunks_metadata = remove_duplicates(top_n1_chunk_embds, top_n1_chunks, chunks_metadata) 

        # 2.3 generative re-ranking using LLM-based ranking model
        top_n2_chunks, chunks_metadata = generative_reranking(top_chunks, queries, chunks_metadata, reranking_model, config.n_2)

        ## save the results
        claim_results["evidences_metadata"] = chunks_metadata
        claim_results["evidences_content"] = top_n2_chunks
        save_json(config.results_path, results) 

if __name__ == "__main__":
    main()