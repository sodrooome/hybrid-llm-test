def extract_summary_anchors(request_body: dict[str, str]) -> list:
    anchors = []

    if "latest_position" in request_body:
        anchors.append(request_body["latest_position"])

    if "industry" in request_body:
        anchors.append(request_body["industry"])

    if "years_experience" in request_body:
        anchors.append(str(request_body["years_experience"]) + " years of experience")

    if "start_availability" in request_body:
        anchors.append(request_body["start_availability"])

    if "skill_list" in request_body:
        anchors += [skill.strip() for skill in request_body["skill_list"].split(",")]

    if "language_list" in request_body:
        anchors += [
            language.strip() for language in request_body["language_list"].split(",")
        ]

    return anchors


def check_coverage_anchors(summary: str, anchors: list) -> bool:
    return [anchor for anchor in anchors if anchor.lower() not in summary.lower()]
