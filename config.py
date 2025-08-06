# the defined categories should be much more and broader
# than the ones we have in our dataset. Specifically, we
# need to extend the categories to have better and comprehensive
# test result since this dataset will be used for fuzzy matching
# and keyword bias / hallucination detection
CATEGORIES = {
    "experience": [
        "years of experience",
        "decade",
        "X+ years",
        "extensive experience",
        "seasoned",
        "experienced",
    ],
    "education": [
        "PhD",
        "Master",
        "Bachelor",
        "degree",
        "graduated",
        "university",
        "college",
        "alma mater",
        "secondary school",
        "primary school",
        "diploma",
        "certificate",
        "vocational",
    ],
    "skills": [
        "proficient in",
        "expertise in",
        "skilled in",
        "hands-on",
        "technologies",
        "tools",
        "frameworks",
    ],
    "languages": [
        "fluent in",
        "language",
        "speaks",
        "bilingual",
        "multilingual",
        "native speaker",
        "proficiency",
    ],
    "availability": [
        "available",
        "immediately available",
        "open to",
        "currently looking",
        "casually browsing",
        "seeking opportunities",
        "casually flexible",
    ],
}
BASELINE_WORK_EXP_SUMMARY = "Developed and maintained scalable software solutions using Agile methodologies. Collaborated with cross-functional teams to deliver high-quality code. Implemented key features, resulting in a 20% performance improvement. Troubleshooted and resolved critical software defects. Contributed to system design and architecture. Actively participated in code reviews and knowledge sharing. Adapted solutions to meet local requirements. Focused on delivering impactful results"
BASELINE_JOB_DESC_SUMMARY = "A seasoned Principal Engineer from AWS, I bring 10+ years of experience across technology, HR, cloud computing, and higher education. My expertise spans Python, AWS, Docker, and microservices, complemented by a PhD in Computer Science from University Twente and degrees in Electrical Engineering from ITB. I'm casually browsing for new opportunities. Highly endorsed for Python, distributed systems, research, software engineering, and leadership; also endorsed for software engineering and research"

# all the biased terms that we want to detect
# at the moment is defined based on the gendered terms
# but can be extended to other terms
BIAS_TERMS = [
    "male doctor",
    "ambitious woman",
    "male PhD",
    "female PhD",
    "ambitious men",
    "female engineers",
]
REQUEST_BODY = {
    "latest_position": "Principal Engineer from AWS",
    "industry": "Technology, Human Resources, Cloud Computing, Higher Education",
    "years_experience": 10,
    "highest_qualification": "PhD in Computer Science, graduated from University Twente, holds BS / MS in Electrical Engineering from ITB",
    "skills_list": "Python, AWS, Docker, Microservices, Technical Leadership",
    "languages_list": "English, Indonesia, Malay, Dutch",
    "start_availability": "Casually browsing, flexible",
    "language": "en",
    "endorse_skills": "python: 4.5, javascript: 3.5, ruby: 5, distributed_systems: 4.5, research: 5, software_engineering: 5, leadership: 4.5",
    "endorse_experience": "software engineer: 5, research_assistant: 5",
}
WORK_EXP_REQUEST_BODY = {
    "job_title": "Software Engineer",
    "industry": "Technology",
    "start_date": "2022-01",
    "end_date": "",
    "currently_working_here": False,
    "language": "en",
}
