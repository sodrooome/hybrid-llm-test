import requests
import warnings
import os
from dotenv import load_dotenv
from config import BATCH_REQUEST_BODY, BATCH_WORK_EXP_REQUEST_BODY
from bert_score import score
from concurrent.futures import ThreadPoolExecutor, as_completed


warnings.filterwarnings(
    "ignore",
    message="You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.",
)

load_dotenv()

base_url = os.getenv("BASE_URL")
x_api_key = os.getenv("X_API_KEY")


def generate_profile_summary(request_body: dict[str, str]):
    # change this base URL here with the one you have
    request_url = f"{base_url}"
    request_headers = {
        "x-api-key": f"{x_api_key}",
        "type": "profile_summary",
    }
    response = requests.post(request_url, headers=request_headers, json=request_body)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def generate_work_experience(request_body: dict[str, str]):
    # change this base URL here with the one you have
    request_url = f"{base_url}"
    request_headers = {
        "x-api-key": f"{x_api_key}",
        "type": "work_experience",
    }
    response = requests.post(request_url, headers=request_headers, json=request_body)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def generate_batch_summaries(type: str) -> list:
    results = []

    # WIP: should tidy up this function
    if type == "work_exp":
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = [
                executor.submit(generate_work_experience, input_data)
                for input_data in BATCH_WORK_EXP_REQUEST_BODY
            ]

            for future in as_completed(future_to_url):
                try:
                    results.append(future.result())
                except Exception as e:
                    raise Exception(f"There's something wrong with the request: {e}")
    elif type == "profile_summary":
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = [
                executor.submit(generate_profile_summary, input_data)
                for input_data in BATCH_REQUEST_BODY
            ]

            for future in as_completed(future_to_url):
                try:
                    results.append(future.result())
                except Exception as e:
                    raise Exception(f"There's something wrong with the request: {e}")
    else:
        raise ValueError(
            "Invalid type, for now only consists of work_exp and profile_summary"
        )

    return results


def semantic_similarity(reference, candidate) -> float:
    # semantic similarity score might not identify the best
    # result, but it is a good starting point for the
    # fuzzy matching and keyword bias detection. High score
    # also doesn't mean that the summary is similar enough,
    # at the moment, this proof-of-concept is focused on
    # QA testing and we already have anchor coverage and use-case
    # based scenarios to test the generated summaries
    (
        _,
        _,
        F1,
    ) = score(
        [candidate], [reference], lang="en", verbose=False, rescale_with_baseline=True
    )
    return F1[0].item()


def check_generated_length_skills(text: list) -> bool:
    return len(text) <= 3 or len(text) >= 10


def skills_fuzzy_match(skill: list, category: dict[str, str]) -> dict:
    for category, keywords in category.items():
        if any(keyword.lower() in skill.lower() for keyword in keywords):
            return category


def check_length_limit(text, max_length: int = 500) -> bool:
    return len(text) <= max_length


def check_incomplete_generation(text: str) -> bool:
    return not text.strip().endswith("..")


def fuzzy_match(summary: str, category: dict[str, str]) -> dict:
    summary = summary.lower()
    coverage = {}
    for category, keywords in category.items():
        match_category = any(keyword.lower() in summary for keyword in keywords)
        coverage[category] = match_category

    return coverage


def keyword_bias_match(text: str, keyword_list: list) -> list:
    text = text.lower()
    return [keyword for keyword in keyword_list if keyword in text]


def is_hallucinated(summary: str, anchors: list) -> list:
    hallucinated = []
    words = set(summary.lower().split())
    anchors_joined = " ".join(anchor.lower() for anchor in anchors)
    for word in words:
        if word.isalpha() and word not in anchors_joined:
            hallucinated.append(word)
    return hallucinated
