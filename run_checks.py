from generated_job_desc import (
    generate_job_desc,
    semantic_similarity,
    check_length_limit,
    check_incomplete_generation,
    fuzzy_match,
    keyword_bias_match,
    is_hallucinated,
    generate_work_experience,
)
from extraction import extract_summary_anchors, check_coverage_anchors
from config import (
    BASELINE_JOB_DESC_SUMMARY,
    CATEGORIES,
    BIAS_TERMS,
    REQUEST_BODY,
    BASELINE_WORK_EXP_SUMMARY,
)
from llm_logger import log_llm_response


def run_work_experience_checks():
    work_experience = generate_work_experience()["descriptions"]
    similarity_score = semantic_similarity(BASELINE_WORK_EXP_SUMMARY, work_experience)
    length_is_ok = check_length_limit(work_experience)
    incomplete_generation = check_incomplete_generation(work_experience)
    coverage = fuzzy_match(work_experience, CATEGORIES)

    if similarity_score < 0.92:
        print(
            f"⚠️ Work experience descriptions are not similar enough with score: {similarity_score:.3f}"
        )
    else:
        print(
            f"✅ Work experience descriptions are similar enough with score: {similarity_score:.3f}"
        )

    if length_is_ok:
        print(
            f"✅ Work experience description length is within the limit: {len(work_experience)}"
        )
    else:
        print(
            f"⚠️  Work experience description length is not within the limit: {len(work_experience)}"
        )

    if incomplete_generation:
        print("⚠️ Work experience description is incomplete with ending '...'")
    else:
        print("✅ Work experience description is complete without ending with '...'")

    # model version and prompt version should be parameterized, not just hardcoding
    # but again, all this specific scenarios are not covered by the current proof-of-concept
    log_llm_response(
        llm_model="gpt-4",
        prompt_version="v1",
        generated_summary=work_experience,
        baseline_summary=BASELINE_WORK_EXP_SUMMARY,
        similarity_score=similarity_score,
        length_is_ok=length_is_ok,
        incomplete_generation=incomplete_generation,
        coverage=coverage,
        anchors_coverage=None,
        bias_detected=None,
        hallucinated=None,
    )


def run_all_checks() -> None:
    new_job_desc = generate_job_desc()["descriptions"]
    anchors = extract_summary_anchors(request_body=REQUEST_BODY)
    similarity_score = semantic_similarity(BASELINE_JOB_DESC_SUMMARY, new_job_desc)
    length_is_ok = check_length_limit(new_job_desc)
    incomplete_generation = check_incomplete_generation(new_job_desc)
    coverage = fuzzy_match(new_job_desc, CATEGORIES)
    bias_match = keyword_bias_match(new_job_desc, BIAS_TERMS)
    anchors_coverage = check_coverage_anchors(new_job_desc, anchors)
    hallucinated = is_hallucinated(new_job_desc, anchors)

    # if anchors_coverage:
    #     print(f"⚠️ Job description is missing anchors: {anchors}")
    # else:
    #     print(f"✅ Job description is complete with all anchors")

    # TODO: need to tidy up the checks and logging by create reusable functions
    if hallucinated:
        print(f"⚠️ Job description contains hallucinated words: {hallucinated}")
    else:
        print(f"✅ Job description does not contain hallucinated words")

    if similarity_score < 0.92:
        print(
            f"⚠️ Job descriptions are not similar enough with score: {similarity_score:.3f}"
        )
    else:
        print(
            f"✅ Job descriptions are similar enough with score: {similarity_score:.3f}"
        )

    if length_is_ok:
        print(f"✅ Job description length is within the limit: {len(new_job_desc)}")
    else:
        print(f"⚠️  Job description length is not within the limit: {len(new_job_desc)}")

    if incomplete_generation:
        print("⚠️ Job description is incomplete with ending '...'")
    else:
        print("✅ Job description is complete without ending with '...'")

    # if bias_match:
    #     bias_match.append("bias_detected")

    # for category, found in coverage.items():
    #     if found:
    #         print(f"✅ Job description contains {category}")
    #     else:
    #         print(f"⚠️ Job description does not contain {category}")

    log_llm_response(
        llm_model="gpt-3.5-turbo",  # change this model to your preferred one or based on the APIs you have
        prompt_version="v1",
        generated_summary=new_job_desc,
        baseline_summary=BASELINE_JOB_DESC_SUMMARY,
        similarity_score=similarity_score,
        length_is_ok=length_is_ok,
        incomplete_generation=incomplete_generation,
        coverage=coverage,
        anchors_coverage=None,
        bias_detected=None,
        hallucinated=None,
    )


if __name__ == "__main__":
    run_all_checks()
    run_work_experience_checks()
