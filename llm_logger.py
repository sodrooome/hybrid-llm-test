import json
import os
import uuid
from datetime import datetime


def generate_has_id() -> str:
    # the best bet we can do is to generate a unique hash ID
    # for each LLM response. This can be used to track the
    # responses and compare them with the previous responses
    # however, the viable solution is to combine between:
    # prompt version + llm model + timestamp + jobID / userId
    return uuid.uuid4().hex


def log_llm_response(
    llm_model: str,
    prompt_version: str,
    generated_summary: str,
    baseline_summary: str,
    similarity_score: float,
    length_is_ok: bool,
    incomplete_generation: bool,
    coverage: dict,
    anchors_coverage: list = None,
    bias_detected: bool = None,
    hallucinated: list = None,
) -> None:
    test_failures = []
    summary_length = len(generated_summary)

    if similarity_score < 0.92:
        # the similarity score should be customizable through arguments
        test_failures.append("similarity_score")

    if incomplete_generation:
        test_failures.append("incomplete_generation")

    if length_is_ok:
        test_failures.append("length_is_ok")

    if anchors_coverage:
        test_failures.append("missing_anchors")

    if hallucinated:
        test_failures.append("hallucinated")

    if not all(coverage.values()):
        for category, found in coverage.items():
            if not found:
                test_failures.append(category)

    test_result = "pass" if not test_failures else "fail"

    log_llm_entry = {
        "hash_id": generate_has_id(),
        "llm_model": llm_model,
        "prompt_version": prompt_version,
        "generated_summary": generated_summary,
        "baseline_summary": baseline_summary,
        "similarity_score": similarity_score,
        "length_is_ok": length_is_ok,
        "incomplete_generation": incomplete_generation,
        "coverage": coverage,
        "missing_anchors": anchors_coverage,
        "test_failures": test_failures,
        "summary_length": summary_length,
        "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "test_result": test_result,
        "bias_detected": bias_detected,
        "hallucinated": hallucinated,
    }

    if os.path.exists("llm_log.json"):
        with open("llm_log.json", "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_llm_entry)

    with open("llm_log.json", "w") as f:
        json.dump(logs, f, indent=4)

    print("Successfully logged LLM response")
