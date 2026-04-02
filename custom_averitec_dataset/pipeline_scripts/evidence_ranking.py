import argparse 
from pathlib import Path
import os 
import sys
from tqdm.auto import tqdm 
import json 
import torch 
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from utils import load_json, save_json 

def parse_args():
    parser = argparse.ArgumentParser(description="Ranking the relevance of retrieved evidence chunks with respect to the individual claims using sentence transformer.")
    parser.add_argument("--input_path", type=str, default=None, help="Path to the input file with claims (in AVeriTeC format)")
    parser.add_argument("--store_folder", type=str, default="store_folder", help="Folder path with stored evidence")
    parser.add_argument("--start_idx", type=int, default=0, help="Claim to start with")
    parser.add_argument("--end_idx", type=int, default=None, help="Claim to end with")
    parser.add_argument("--n_1", type=int, default=50, help="Number of documents to keep after first ranking with BM25")
    parser.add_argument("--n_2", type=int, default=10, help="Number of documents to keep after second ranking with BERT")

    return parser.parse_args()