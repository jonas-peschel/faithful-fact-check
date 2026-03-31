import argparse 
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
import torch 
from json import JSONDecodeError 
from pydantic import RootModel, ValidationError
from pathlib import Path 
from tqdm.auto import tqdm
from typing import List
from utils import load_json, save_json 

def parse_args():
    parser = argparse.ArgumentParser(description="Generate search queries for web-based evidence retrieval.")
    parser.add_argument("--results_path", type=str, default=None, help="Path to the AVeriTeC data file.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print model output text")
    return parser.parse_args()

def load_model():
    model_name = "mistralai/Ministral-3-14B-Instruct-2512"
    tokenizer = MistralCommonBackend.from_pretrained(model_name)
    model = Mistral3ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
    device = model.device 
    return model, tokenizer, device 

def get_prompt_texts(claim, date):

    system_prompt_text = 'You are an expert investigative journalist and professional fact-checker. You are given a claim, which is a verifiable, factual statement, and you are given the date that the claim was made. Your task is to generate 3-5 distinct search queries designed to find evidence relevant for verifying the veracity of the given claim. You must follow the following rules for generating the search queries:\n1. Generate 3-5 search queries.\n2. The search queries must be diverse and cover different retrieval strategies, for instance, direct factual lookup, entity-specific queries, numeric/quantitative queries (if applicable), comparison queries (if applicable), source-seeking queries.\n3. Incorporate the date that the claim was made into the search queries. Ignore this when the date is not given, i.e., it is None.\n4. Each query must be short, concise, and optimized for search engines.\n5. Do NOT repeat the same meaning with different wording.\n6. Do NOT include meta-terms such as "fact-check", "hoax", or "debunk". Focus on retrieving the original source material, official statistics, transcripts, and contemporary news reports.\n7. Output a list of strings in valid JSON format, where each list entry corresponds to one search query.\n8. Do NOT output any additional text or explanations.'
    system_prompt_text += """\n\nExamples:

Claim: "Alex Jones claimed on the InfoWars website that Democratic-led areas told people to 'never' remove their masks during the pandemic."
Date: 21-6-2020
Queries: [
    "Alex Jones InfoWars broadcast June 2020 mask mandates",
    "Democratic city COVID-19 mask guidance June 2020",
    "Blue state public health orders 'permanent' masking June 2020",
    "InfoWars website article Democratic mask mandates June 21 2020"
]

Claim: "BJP president Amit Shah said that the days of serving biryani to terrorists are over"
Date: 20-6-2016
Queries: [
    "Amit Shah speech June 2016 'biryani to terrorists'",
    "BJP president address June 2016 terrorism biryani quote",
    "Amit Shah rally June 20 2016 transcript",
    "Indian news reports Amit Shah biryani terrorist remark June 2016"
]

Claim: "Donald Trump said that he 'did criminal justice reform, which President Obama could not get approved...'"
Date: 9-8-2019
Queries: [
    "Donald Trump speech August 9 2019 criminal justice reform",
    "Trump remarks Obama criminal justice reform failure August 2019",
    "Prison Reform Summit August 2019 Donald Trump quotes",
    "First Step Act implementation news August 2019"
]

Claim: "Italy went against the WHO instruction of not performing autopsies on COVID-19 casualties."
Date: None
Queries: [
    "WHO COVID-19 autopsy official guidelines",
    "Italy Ministry of Health protocol COVID-19 autopsies",
    "Italy coronavirus autopsy findings report",
    "World Health Organization recommendations post-mortem examinations COVID-19"
]

Claim: "To counter Deepika, BJP enlists Shaan, Tanisha"
Date: 10-1-2020
Queries: [
    "BJP celebrity support campaign January 10 2020",
    "Shaan and Tanishaa Mukerji BJP video January 2020",
    "Deepika Padukone JNU visit BJP response January 2020",
    "Bollywood actors supporting CAA BJP event January 2020"
]
"""

    user_prompt_text = f"Generate 3-5 distinct search queries designed to find evidence relevant for verifying the veracity of the given claim. Remember to use valid JSON format for your output.\n\nClaim: {claim}\nDate: {date}\nQueries: "

    return system_prompt_text, user_prompt_text

def get_prompt_ids(tokenizer, device, claim, date):

    system_prompt, user_prompt = get_prompt_texts(claim, date)
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    prompt_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        return_tensors="pt", 
        return_dict=True,
    ).to(device)
    return prompt_ids

def query_model(model, tokenizer, device, claim, date):

    generate_kwargs = {
        "max_new_tokens": 1024, 
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }

    prompt_ids = get_prompt_ids(tokenizer, device, claim, date) 
    model.eval()
    with torch.inference_mode():
        output = model.generate(**prompt_ids, **generate_kwargs)
    output_text = tokenizer.decode(output.squeeze()[prompt_ids["input_ids"].shape[1]:])
    return output_text

class SearchQueriesList(RootModel):
    root: List[str]

def parse_output_text(output_json: str, eos_token: str) -> List[str] | None:

    max_queries = 5

    try:
        clean_json = output_json.strip().replace("```json", "").replace("```", "").replace(eos_token, "")  # clean triple backticks and end-of-sequence token
        search_queries = SearchQueriesList.model_validate_json(clean_json)
    except (JSONDecodeError, ValidationError): 
        return None

    return search_queries.root[:max_queries]

def get_model_output(model, tokenizer, device, claim, date):
    n_tries = 5
    for _ in range(n_tries):
        output_text = query_model(model, tokenizer, device, claim, date)
        search_queries = parse_output_text(output_text, tokenizer.eos_token)

        # return results if they are valid
        if search_queries:
            return search_queries, output_text 
        
    return None, output_text 

def main(config=None):

    if config is None:
        config = parse_args() 

    results = load_json(config.results_path)  # load existing results

    model, tokenizer, device = load_model()

    for claim_results in tqdm(results):
        # skip claim if it already has results computed
        if claim_results.get("search_queries"):
            continue 

        claim_text = claim_results["claim"]
        date = claim_results.get("claim_date")
        search_queries, output_text = get_model_output(model, tokenizer, device, claim_text, date) 

        if config.verbose:
            print(f"Claim: {claim_text}\nModel output: {output_text}\nSearch Queries: {search_queries}\n\n")

        # add search queries to the results
        claim_results["search_queries"] = search_queries

        # save results each iteration
        save_json(config.results_path, results)

if __name__ == "__main__":
    main()