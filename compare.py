import json


def llm_log_checker(llm_log_path: dict, previous_llm_log_path: dict) -> dict:
    diffs = {}

    # assuming certain key always exists in the JSON file currently may not hold
    # will need to update this to handle the case when the key is not present
    if llm_log_path["test_result"] != previous_llm_log_path["test_result"]:
        diffs["test_result_changed"] = True

    old_failures = set(map(str, previous_llm_log_path["test_failures"]))
    new_failures = set(map(str, llm_log_path["test_failures"]))
    diffs["new_failures"] = list(new_failures - old_failures)

    diffs["similarity_score_changed"] = {
        round(llm_log_path.get("similiarity_score", 0), 3),
        round(previous_llm_log_path.get("similiarity_score", 0), 3),
    }

    old_anchors = set(previous_llm_log_path.get("missing_anchors", []))
    new_anchors = set(llm_log_path.get("missing_anchors", []))
    diffs["new_missing_anchors"] = list(new_anchors - old_anchors)

    old_hallucinated = set(previous_llm_log_path.get("hallucinated", []))
    new_hallucinated = set(llm_log_path.get("hallucinated", []))
    diffs["new_hallucinated"] = list(new_hallucinated - old_hallucinated)

    diffs["new_bias_terms"] = list(
        set(llm_log_path.get("bias_detected", []))
        - set(previous_llm_log_path.get("bias_detected", []))
    )

    category_diffs = {}
    for category in llm_log_path.get("coverage", {}):
        if llm_log_path["coverage"].get(category) != previous_llm_log_path[
            "coverage"
        ].get(category):
            category_diffs[category] = {
                "before": previous_llm_log_path["coverage"].get(category),
                "after": llm_log_path["coverage"].get(category),
            }
    diffs["category_diffs_changed"] = category_diffs

    return diffs


def compare_llm_logs(llm_log_path: str, previous_llm_log_path: str) -> dict:
    try:
        with open(llm_log_path, "r") as new, open(previous_llm_log_path, "r") as old:
            new_llm_log = json.load(new)
            previous_llm_log = json.load(old)
            diffs = llm_log_checker(new_llm_log, previous_llm_log)
    except FileNotFoundError:
        raise Exception("No LLM log found")
    except json.JSONDecodeError:
        raise Exception("Invalid JSON format")
    return diffs


if __name__ == "__main__":
    new_llm_log_path = "new_llm_log.json"
    previous_llm_log_path = "previous_llm_log.json"
    diffs = compare_llm_logs(new_llm_log_path, previous_llm_log_path)
    print(diffs)
