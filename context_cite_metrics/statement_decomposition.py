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

def get_decomposition_prompts(dataset_name: str, query: str, answer_statements: List[str]):

    if (dataset_name == "averitec" or dataset_name == "averitec_short_ans" 
        or dataset_name == "averitec_web_evidence"  or dataset_name == "averitec_web_evidence_short_ans"):

        claim = extract_claim_text(query)

        sys_prompt = "You are an expert text analysis assistant. Your task is to decompose statements from an answer text into atomic facts. The answer text is a fact-checking justification based on a given document or pieces of evidence about a given claim. An atomic fact is a single, self-contained statement or claim that can be verified independently.\n\nInput format: The input is a numbered list of statements. Each element in the list represents one statement from the answer.\n\nOutput format: Your output must be in valid JSON format. Your output must be a JSON list of lists of strings. The outer list must have the exact same length as the input list, i.e., there must be an entry for each of the input statements. Each element in your output list is a list of atomic facts from the corresponding input statement. Respond strictly only with the JSON list and do not output any additional text.\n\nRules for decomposition:\n- Each atomic fact should contain exactly one verifiable statement or claim\n- Preserve all specific details like names, numbers, and dates\n- Each atomic fact should be understandable without context\n- Do not add information that is not in the original statement\n- Do not include atomic facts for information that is present in the answer only for introduction, conclusion, or summarization.\n- When the atomic facts are put back together, it should convey the same factual information as the original answer, excluding introductory, conclusive, or summarizing statements\n- Strip all meta-references to the evidence or documents, such as 'based on the provided evidence', 'the evidence provided in the document states that', 'the evidence shows' etc.\n- When an answer statement does not contain any atomic facts, i.e., it does not contain any verifiable factual claim or statement, return an empty list for that statement\n\nExample:\nAnswer statements:\n1. Based on the provided evidence, the claim that \\'Before he was mayor, Pete Buttigieg ran statewide in Indiana and lost by 20 points\\' is **refuted**. The evidence provided in the document states that Buttigieg did run for statewide office in Indiana in 2010, but he did not lose by 20 points. Instead, he lost the race for Indiana state treasurer to incumbent treasurer and eventual governor Eric Holcomb by a margin of 37.5% to 62.5%. This was a significant defeat, but not by the 20 point margin claimed in the original statement.\n2. The document also corrects an earlier version of the article that had incorrectly stated that Buttigieg had never run for office outside of South Bend. The correct statement is that he has only won public office outside of South Bend once, when he was elected as a member of the Indiana House of Representatives in 2004.\n3. Therefore, based on the evidence provided, the claim that Buttigieg lost a statewide race in Indiana by 20 points is refuted.\n\nOutput:\n[['Buttigieg ran for statewide office in Indiana in 2010.', 'Buttigieg did not lose the 2010 Indiana statewide race by 20 points.', 'Buttigieg lost the race for Indiana state treasurer.', 'The incumbent treasurer Buttigieg ran against was Eric Holcomb.', 'Eric Holcomb eventually became governor of Indiana.', 'Buttigieg received 37.5% of the vote in the Indiana state treasurer race.', 'Eric Holcomb received 62.5% of the vote in the Indiana state treasurer race.'], ['An earlier version of the article incorrectly stated that Buttigieg had never run for office outside of South Bend.', 'Buttigieg has only won public office outside of South Bend once.', 'Buttigieg was elected as a member of the Indiana House of Representatives in 2004.'], []]"

        user_prompt = f"Decompose the following statements from a fact-checking justification into atomic facts. Your output must be a valid JSON list of lists of strings. Remember that your output JSON list must have as many entries as there are input answer statements. The justification was written about the claim '{claim}'.\n\nAnswer statements:\n{answer_statements}\n\nOutput:"

    elif dataset_name == "multifieldqa_en":
        
        sys_prompt = "You are an expert text analysis assistant. Your task is to decompose statements from an answer text into atomic facts. The answer text provides information or an explanation based on a source document in response to a specific question. An atomic fact is a single, self-contained statement or claim that can be verified independently.\n\nInput format: The input is a numbered list of statements. Each element in the list represents one statement from the answer.\n\nOutput format: Your output must be in valid JSON format. Your output must be a JSON list of lists of strings. The outer list must have the exact same length as the input list, i.e., there must be an entry for each of the input statements. Each element in your output list is a list of atomic facts from the corresponding input statement. Respond strictly only with the JSON list and do not output any additional text.\n\nRules for decomposition:\n- Each atomic fact should contain exactly one verifiable statement or claim\n- Preserve all specific details like names, numbers, and dates\n- Each atomic fact should be understandable without context\n- Do not add information that is not in the original statement\n- Do not include atomic facts for information that is present in the answer only for introduction, conclusion, or summarization.\n- Do not include atomic facts for conversational filler, framing phrases, or redundant summaries that simply repeat facts already listed.\n- When the atomic facts are put back together, it should convey the same factual information as the original answer, excluding introductory, conclusive, or summarizing statements\n- Strip all meta-references to the article or documents, such as 'based on the provided article', 'based on the paper', 'based on the information provided' etc.\n- When an answer statement does not contain any atomic facts, i.e., it does not contain any verifiable factual claim or statement, return an empty list for that statement\n\nExample 1:\nAnswer statements:\n1. Based on the paper, the generative interactive model used in the method is called a Coupled Generalized Dynamic Bayesian Network (C-GDBN).\n2. Specifically, the paper states:\n3. 'In this work, we propose a method to jointly detect GPS spoofing and jamming attacks in the V2X network. A coupled generalized dynamic Bayesian network (C-GDBN) is employed to learn the interaction between RF signals received by the RSU from multiple vehicles and their corresponding trajectories.'\n4. 'The coupled GDBN (C-GDBN) is composed of two GDBNs representing the two signals received at the RSU where their discrete hidden variables are stochastically coupled.'\n5. So the C-GDBN is the generative interactive model used in the method to jointly detect GPS spoofing and jamming attacks in V2X networks.\n\nOutput:\n[['The generative interactive model used in the method is called a Coupled Generalized Dynamic Bayesian Network (C-GDBN).'], [], ['The method jointly detects GPS spoofing and jamming attacks in the V2X network.', 'A coupled generalized dynamic Bayesian network (C-GDBN) learns the interaction between RF signals received by the RSU from multiple vehicles.', 'The C-GDBN learns the interaction between the RF signals and their corresponding trajectories.'], ['The coupled GDBN (C-GDBN) is composed of two GDBNs.', 'The two GDBNs represent two signals received at the RSU.', 'The discrete hidden variables of the two GDBNs are stochastically coupled.'], []]\n\nExample 2:\nAnswer statements:\n1. Based on the information provided, the symptoms of vitamin K deficiency include:\n2. - Anemia\n3. - Bruising \n4. - Nosebleeds\n5. - Bleeding of the gums \n6. - Heavy menstrual bleeding in women\n7. The deficiency can lead to coagulopathy, a bleeding disorder. Newborn infants are at an increased risk of deficiency. Other populations with an increased prevalence of vitamin K deficiency include those who suffer from liver damage or disease, cystic fibrosis, or inflammatory bowel diseases, or have recently had abdominal surgeries. Secondary vitamin K deficiency can occur in people with bulimia, those on stringent diets, and those taking anticoagulants.\n8. The deficiency can also lead to weakened bones and promote calcification of arteries and other soft tissues. Symptoms of K1 deficiency include anemia, bruising, nosebleeds and bleeding of the gums in both sexes, and heavy menstrual bleeding in women.\n9. So in summary, the key symptoms of vitamin K deficiency are bleeding/bruising issues, anemia, and weakened bones. Newborns and certain other populations are at higher risk.\n\nOutput:\n[[], ['Anemia is a symptom of vitamin K deficiency.'], ['Bruising is a symptom of vitamin K deficiency.'], ['Nosebleeds are a symptom of vitamin K deficiency.'], ['Bleeding of the gums is a symptom of vitamin K deficiency.'], ['Heavy menstrual bleeding in women is a symptom of vitamin K deficiency.'], ['Vitamin K deficiency can lead to coagulopathy.', 'Coagulopathy is a bleeding disorder.', 'Newborn infants are at an increased risk of vitamin K deficiency.', 'People who suffer from liver damage or disease have an increased prevalence of vitamin K deficiency.', 'People with cystic fibrosis have an increased prevalence of vitamin K deficiency.', 'People with inflammatory bowel diseases have an increased prevalence of vitamin K deficiency.', 'People who have recently had abdominal surgeries have an increased prevalence of vitamin K deficiency.', 'Secondary vitamin K deficiency can occur in people with bulimia.', 'Secondary vitamin K deficiency can occur in people on stringent diets.', 'Secondary vitamin K deficiency can occur in people taking anticoagulants.'], ['Vitamin K deficiency can lead to weakened bones.', 'Vitamin K deficiency can promote calcification of arteries.', 'Vitamin K deficiency can promote calcification of soft tissues.', 'Symptoms of Vitamin K1 deficiency include anemia.', 'Symptoms of Vitamin K1 deficiency include bruising.', 'Symptoms of Vitamin K1 deficiency include nosebleeds.', 'Symptoms of Vitamin K1 deficiency include bleeding of the gums in both sexes.', 'Symptoms of Vitamin K1 deficiency include heavy menstrual bleeding in women.'], []]"

        user_prompt = f"Decompose the following statements from a question-answering response into atomic facts. Your output must be a valid JSON list of lists of strings. Remember that your output JSON list must have as many entries as there are input answer statements. The answer text was written in response to the question: '{query}'.\n\nAnswer statements:\n{answer_statements}\n\nOutput:"

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

        answer_statements = data_point_results["answer_statements"]
        answer_statements_numbered = "\n".join([f"{i}. {statement}" for i, statement in enumerate(answer_statements, start=1)])

        # query the model 
        sys_prompt, user_prompt = get_decomposition_prompts(dataset_name, data_point_results["query"], answer_statements_numbered)
        atomic_facts = query_model(client, config.model, sys_prompt, user_prompt, answer_statements)

        # store decomposed statements (each iteration)
        data_point_results["decomposed_model_answer"] = atomic_facts
        save_json(config.results_path, results)
    print(f"Saved results to: {config.results_path}")

if __name__ == "__main__": 
    main()