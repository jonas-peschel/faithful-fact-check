import argparse
from tqdm.auto import tqdm
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from pydantic import BaseModel, ValidationError
from json import JSONDecodeError
from utils import load_json, save_json
import re
import time
from pathlib import Path
from typing import List 
import instructor 
import os 

def parse_args():
    parser = argparse.ArgumentParser(description="Decompose generated answer statements into atomic facts using deepseek-chat.")
    parser.add_argument("--results_path", type=str, default=None, help="Path to the unformatted json results file.")
    parser.add_argument("--model", type=str, choices=["deepseek-chat"], default="deepseek-chat")
    return parser.parse_args()

def extract_claim_text(query: str):
    pattern = r"\n\nClaim: (.+)"
    claim = re.search(pattern, query, re.DOTALL).group(1)
    return claim

def get_decomposition_prompts(dataset_name: str, claim: str, answer_statements: List[str]):

    if dataset_name == "averitec" or dataset_name == "averitec_short_ans":
        sys_prompt = "You are an expert text analysis assistant. Your task is to decompose statements from an answer text into atomic facts. The answer text is a fact-checking justification based on a given document or pieces of evidence about a given claim. An atomic fact is a single, self-contained statement or claim that can be verified independently.\n\nInput format: The input is a list of strings. Each element in the list represents one statement from the answer.\n\nOutput format: Your output must be in valid JSON format. Your output must be a JSON list of lists of strings. The outer list must have the exact same length as the input list. Each element in your output list is a list of atomic facts from the corresponding input statement. Respond strictly only with the JSON list and do not output any additional text.\n\nRules for decomposition:\n- Each atomic fact should contain exactly one verifiable statement or claim\n- Preserve all specific details like names, numbers, and dates\n- Each atomic fact should be understandable without context\n- Do not add information that is not in the original statement\n- Do not include atomic facts for information that is present in the answer only for introduction, conclusion, or summarization.\n- When the atomic facts are put back together, it should convey the same factual information as the original answer, excluding introductory, conclusive, or summarizing statements\n- Strip all meta-references to the evidence or documents, such as 'based on the provided evidence', 'the evidence provided in the document states that', 'the evidence shows' etc.\n- When an answer statement does not contain any atomic facts, i.e., it does not contain any verifiable factual claim or statement, return an empty list for that statement\n\nExample:\nAnswer statements:\n['Based on the provided evidence, the claim that \\'Before he was mayor, Pete Buttigieg ran statewide in Indiana and lost by 20 points\\' is **refuted**. The evidence provided in the document states that Buttigieg did run for statewide office in Indiana in 2010, but he did not lose by 20 points. Instead, he lost the race for Indiana state treasurer to incumbent treasurer and eventual governor Eric Holcomb by a margin of 37.5% to 62.5%. This was a significant defeat, but not by the 20 point margin claimed in the original statement.', 'The document also corrects an earlier version of the article that had incorrectly stated that Buttigieg had never run for office outside of South Bend. The correct statement is that he has only won public office outside of South Bend once, when he was elected as a member of the Indiana House of Representatives in 2004.', 'Therefore, based on the evidence provided, the claim that Buttigieg lost a statewide race in Indiana by 20 points is refuted.']\n\nOutput:\n[['Buttigieg ran for statewide office in Indiana in 2010.', 'Buttigieg did not lose the 2010 Indiana statewide race by 20 points.', 'Buttigieg lost the race for Indiana state treasurer.', 'The incumbent treasurer Buttigieg ran against was Eric Holcomb.', 'Eric Holcomb eventually became governor of Indiana.', 'Buttigieg received 37.5% of the vote in the Indiana state treasurer race.', 'Eric Holcomb received 62.5% of the vote in the Indiana state treasurer race.'], ['An earlier version of the article incorrectly stated that Buttigieg had never run for office outside of South Bend.', 'Buttigieg has only won public office outside of South Bend once.', 'Buttigieg was elected as a member of the Indiana House of Representatives in 2004.'], []]"

        user_prompt = f"Decompose the following statements from a fact-checking justification into atomic facts. Your output must be a valid JSON list of lists of strings. The justification was written about the claim '{claim}'.\n\nAnswer statements:\n{answer_statements}\n\nOutput:"

    return sys_prompt, user_prompt

class AtomicFacts(BaseModel):
    atomic_facts: list[list[str]]

def query_model(client, model, sys_prompt, user_prompt, answer_statements, max_retries=5):
    for attempt in range(max_retries):
        try:
            # get model response with structured output
            response = client.chat.completions.create(
                model = model,
                temperature = 0.0 if attempt == 0 else 1.0,
                messages = [
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                response_model = AtomicFacts,
            )
            atomic_facts = response.atomic_facts
            assert len(atomic_facts) == len(answer_statements)
            return atomic_facts
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if attempt == max_retries-1:
                raise 
            time.sleep(5)
        except (ValidationError, JSONDecodeError, ValueError, AssertionError) as e:
            if attempt == max_retries-1:
                raise

def main(config=None):

    if config is None:
        config = parse_args()

    load_dotenv()
    results = load_json(config.results_path)
    dataset_name = results["metadata"]["dataset"]

    client = instructor.from_openai(
        OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        ),
        mode=instructor.Mode.JSON,
    )

    for data_point_results in tqdm(results["results"]):
        if data_point_results.get("decomposed_model_answer") is not None:
            continue
        claim = extract_claim_text(data_point_results["query"])
        answer_statements = data_point_results["answer_statements"]

        # query the model 
        sys_prompt, user_prompt = get_decomposition_prompts(dataset_name, claim, answer_statements)
        atomic_facts = query_model(client, config.model, sys_prompt, user_prompt, answer_statements)

        # store decomposed statements (each iteration)
        data_point_results["decomposed_model_answer"] = atomic_facts
        save_json(config.results_path, results)
    print(f"Saved results to: {config.results_path}")

if __name__ == "__main__": 
    main()