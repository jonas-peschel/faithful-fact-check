import argparse
from tqdm.auto import tqdm
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from pydantic import BaseModel
from utils import load_json, save_json
import re
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Use GPT-4o-mini model for veracity classification via structured output generation on the justification text.")
    parser.add_argument("--results_path", type=str, default=None, help="Path to the json file with results formatted using format_results.py.")
    parser.add_argument("--gpt_model", type=str, choices=['gpt-4o-mini', 'gpt-4o'], default='gpt-4o-mini')
    parser.add_argument("--model_name", type=str, default="", help="Model used for answer generation. Only used for naming the results file.")

    return parser.parse_args()

def parse_plain_answer_text(answer: str):
    answer = re.sub(r"<cite>.*?</cite>", "", answer, flags=re.DOTALL)
    answer = answer.replace('<statement>', '').replace('</statement>', '')
    answer = re.sub(r"<\|reserved_special_token_0\|>.*?<\|reserved_special_token_1\|>", " ", answer, flags=re.DOTALL)
    answer = re.sub(r"<\|reserved_special_token_\d+\|>", " ", answer, flags=re.DOTALL)
    return answer

def get_veracity_classification_prompts(dataset_name: str, claim: str, justification: str):

    if dataset_name == "averitec":
        sys_prompt = (
            "You are an expert in classifying fact-checking justification texts. You are provided with a claim and a justification written by a fact-"
            "checker, which evaluates the veracity of the claim, i.e. if the claim is either supported, refuted, has conflicting evidence, or has not "
            "enough evidence to determine its veracity. Based solely on the provided justification text, determine which veracity label the fact-checker"
            " assigned to the claim. Do not use any external knowledge. Your answer should only be the veracity label of the claim, i.e. you should "
            "only respond with either 'Supported', or 'Refuted', or 'Not Enough Evidence', or 'Conflicting Evidence/Cherrypicking'."
        )
        user_prompt = f"Claim: {claim}\n\nJustification: {justification}"

    return sys_prompt, user_prompt

def get_label_names(dataset_name):

    if dataset_name == "averitec":
        return ['Supported', 'Refuted', 'Conflicting Evidence/Cherrypicking', 'Not Enough Evidence']

class VeracityLabel(BaseModel):
    veracity_label: str

def classify_veracity(client, model, sys_prompt, user_prompt, labels, max_retries=5):

    for attempt in range(max_retries):
        try:
            # get model response with structured output
            response = client.chat.completions.create(
                model = model,
                temperature = 0.0,  # deterministic output always
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
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "veracity_classification_schema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "veracity_label": {
                                    "type": "string",
                                    "enum": labels
                                }
                            },
                            "required": ["veracity_label"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )

            veracity_label = VeracityLabel.model_validate_json(response.choices[0].message.content).veracity_label
            return veracity_label
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if attempt == max_retries-1:
                raise 
            time.sleep(5)

def main(config=None):

    if config is None:
        config = parse_args()

    load_dotenv()
    results = load_json(config.results_path)
    dataset_name = results[0]["dataset"]
    save_path = "./scores_correct/" + config.model_name + f"_{dataset_name}" + f"_pred_labels_{config.gpt_model}" + ".json"
    print(save_path)

    # Open AI model
    client = OpenAI()

    for data_point_result in tqdm(results):

        claim = data_point_result["claim"]
        model_answer = parse_plain_answer_text(data_point_result["prediction"])
        LABEL_NAMES = get_label_names(dataset_name)

        # classify justification to get predicted veracity label
        sys_prompt, user_prompt = get_veracity_classification_prompts(dataset_name, claim, model_answer)
        veracity_label = classify_veracity(client, config.gpt_model, sys_prompt, user_prompt, LABEL_NAMES)
        if config.gpt_model == "gpt-4o":
            time.sleep(5)  # sleep to avoid hitting rate limit

        # store veracity_label (each iteration)
        data_point_result["pred_label"] = veracity_label
        save_json(config.results_path, results)
        save_json(save_path, results)
        print(f"Saved results to: {save_path}")

if __name__ == "__main__": 
    main()