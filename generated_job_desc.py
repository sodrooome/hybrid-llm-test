import requests
import warnings
from config import REQUEST_BODY, WORK_EXP_REQUEST_BODY
from bert_score import score  # type: ignore


warnings.filterwarnings(
    "ignore",
    message="You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.",
)


def generate_job_desc():
    # change this base URL here with the one you have
    request_url = "base-url-here/development/genai"
    request_headers = {
        "x-api-key": "some-secret-key",
        "type": "profile_summary",
    }
    response = requests.post(request_url, headers=request_headers, json=REQUEST_BODY)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def generate_work_experience():
    # change this base URL here with the one you have
    request_url = "base-url-here/development/genai"
    request_headers = {
        "x-api-key": "some-secret-key",
        "type": "work_experience",
    }
    response = requests.post(
        request_url, headers=request_headers, json=WORK_EXP_REQUEST_BODY
    )
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


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
