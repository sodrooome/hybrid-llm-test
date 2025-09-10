from generated_job_desc import (
    generate_profile_summary,
    semantic_similarity,
    check_length_limit,
    check_incomplete_generation,
    fuzzy_match,
    keyword_bias_match,
    is_hallucinated,
    generate_work_experience,
    generate_batch_summaries,
)
from extraction import extract_summary_anchors, check_coverage_anchors
from config import (
    BASELINE_JOB_DESC_SUMMARY,
    CATEGORIES,
    BIAS_TERMS,
    REQUEST_BODY,
    BASELINE_WORK_EXP_SUMMARY,
    WORK_EXP_REQUEST_BODY,
)
from llm_logger import log_llm_response
from tabulate import tabulate  # type: ignore
from metrics.score import bert_score


def _print_check_result(
    similarity_score: float,
    length_is_ok: bool,
    incomplete: bool,
    length_of_job_desc: int,
    hallucinated: list = None,
):
    # this is a method which is used to print the results of the checks,
    # it is used to make the output more readable and easier to understand,
    # by collecting all the results in one place rather than printing them individually
    if similarity_score < 0.92:
        print(
            f"⚠️ Job descriptions are not similar enough with score: {similarity_score:.3f}"
        )
    else:
        print(
            f"✅ Job descriptions are similar enough with score: {similarity_score:.3f}"
        )

    if length_is_ok:
        print(
            f"✅ Job description length is within the limit: {len(length_of_job_desc)}"
        )
    else:
        print(
            f"⚠️  Job description length is not within the limit: {len(length_of_job_desc)}"
        )

    if incomplete:
        print("⚠️ Job description is incomplete with ending '...'")
    else:
        print("✅ Job description is complete without ending with '...'")

    if hallucinated:
        print(f"⚠️ Job description contains hallucinated words: {hallucinated}")
    else:
        print(f"✅ Job description does not contain hallucinated words")


def run_batch_summaries_checks():
    batch_summaries = generate_batch_summaries(type="work_exp")
    result_tables = []

    for index, summary in enumerate(batch_summaries):
        generated_summary = summary["descriptions"]
        similarity_score = semantic_similarity(
            BASELINE_JOB_DESC_SUMMARY, generated_summary
        )
        length_is_ok = check_length_limit(generated_summary)
        incomplete_generation = check_incomplete_generation(generated_summary)
        coverage = fuzzy_match(generated_summary, CATEGORIES)

        result_tables.append(
            [
                index,
                f"{similarity_score:.3f}",
                "✅" if length_is_ok else "⚠️",
                "✅" if incomplete_generation else "⚠️",
                f"{len(generated_summary)}",
            ]
        )

        log_llm_response(
            llm_model="gpt-4o",
            prompt_version="v1",
            generated_summary=generated_summary,
            baseline_summary=BASELINE_JOB_DESC_SUMMARY,
            similarity_score=similarity_score,
            length_is_ok=length_is_ok,
            incomplete_generation=incomplete_generation,
            coverage=coverage,
            anchors_coverage=None,
            bias_detected=None,
            hallucinated=None,
        )

    headers = [
        "Index",
        "Similarity Score",
        "Length Is Ok",
        "Complete Summary",
        "Length",
    ]
    print("\n Batch Summary Results")
    print(
        "\n Job Title Samples: Software Engineer, Sales, Admin, Chef, Barista, Graphic Designer, Driver, Product Manager, Teacher, Retail, Technician\n"
    )
    print(tabulate(result_tables, headers=headers, tablefmt="github"))


def run_work_experience_checks():
    work_experience = generate_work_experience(request_body=WORK_EXP_REQUEST_BODY)[
        "descriptions"
    ]

    # prefer to not use the BERTScore from the provided packages,
    # but rather use the one that we've in our code base and written
    # from scratch, however, this is not a strict requirement as per now
    internal_bert_score = bert_score(BASELINE_WORK_EXP_SUMMARY, work_experience)
    similarity_score = semantic_similarity(BASELINE_WORK_EXP_SUMMARY, work_experience)
    length_is_ok = check_length_limit(work_experience)
    incomplete_generation = check_incomplete_generation(work_experience)
    coverage = fuzzy_match(work_experience, CATEGORIES)

    _print_check_result(
        similarity_score=internal_bert_score,
        length_is_ok=length_is_ok,
        incomplete=incomplete_generation,
        length_of_job_desc=work_experience,
    )

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


def run_generated_job_desc_checks():
    new_job_desc = generate_profile_summary(request_body=REQUEST_BODY)["descriptions"]
    anchors = extract_summary_anchors(request_body=REQUEST_BODY)
    similarity_score = semantic_similarity(BASELINE_JOB_DESC_SUMMARY, new_job_desc)
    length_is_ok = check_length_limit(new_job_desc)
    incomplete_generation = check_incomplete_generation(new_job_desc)
    coverage = fuzzy_match(new_job_desc, CATEGORIES)
    bias_match = keyword_bias_match(new_job_desc, BIAS_TERMS)
    anchors_coverage = check_coverage_anchors(new_job_desc, anchors)
    hallucinated = is_hallucinated(new_job_desc, anchors)

    if anchors_coverage:
        print(f"⚠️ Job description is missing anchors: {anchors}")
    else:
        print(f"✅ Job description is complete with all anchors")

    _print_check_result(
        similarity_score=similarity_score,
        length_is_ok=length_is_ok,
        incomplete=incomplete_generation,
        length_of_job_desc=new_job_desc,
        hallucinated=hallucinated,
    )

    if bias_match:
        bias_match.append("bias_detected")

    for category, found in coverage.items():
        if found:
            print(f"✅ Job description contains {category}")
        else:
            print(f"⚠️ Job description does not contain {category}")

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
    run_generated_job_desc_checks()
    run_work_experience_checks()
    run_batch_summaries_checks()
