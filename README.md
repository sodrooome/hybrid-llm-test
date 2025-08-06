## Hybrid LLM-based Automated Test

**proof-of-concept** for a hybrid LLM-based automated test for AI generative models. This test is designed to assess the quality of a LLM-generated summary by comparing it to a baseline summary (or any kind of golden dataset that we've) and identifying any differences. 

please do consideration, this proof-of-concept will be used for QA testing of AI models not just for scientific evaluation which may not suitable for this purposes

### Overview

The proof-of-concept presents a simplistic LLM-based automated test for AI generative models. The test is designed to assess the quality of a LLM-generated summary by comparing it to a baseline summary (or any kind of golden dataset that we've) and identifying any differences

Most of the time (at least based on our cases that we had), the LLM-generated summaries are not always accurate and can contain hallucinated words, which can be a problem for recruitment platforms that rely on these summaries to match candidates with job openings

With this proof-of-concept, we aim to create a hybrid LLM-based automated test that can detect hallucinated words, bias terms and missing anchors in the generated summaries. Not just with that, we also want to ensure that the LLM-generated summaries are strictly bound with our business requirements such as:

- Generated summaries should be within the length limit (max 500 words)
- Generated summaries should be complete and not end with '...'
- Generated summaries should be semantically similar to the baseline summary

Still, the subjectivity [1] of the LLM-generated summaries is quite unique and become frontier for the LLMs in recent years, specifically for us as QA testers to evaluate all the generated summaries. With this, we aim to incorporate HITL (human-in-the-loop) testing to evaluate the quality of the summaries [2], which is performed qualitatively by QA and our stakeholders (thus, this will act as an subject-matter-expert in this case)

The initial idea of this proof-of-concept can be see by the following architecture diagram:

![Architecture Diagram](/images/initial-diagram.png)

For now, all the LLM-generated summaries are stored in a JSON file, which is then used for the evaluation with several considerations:

- Regression detection for catching when models updates or changes over time, that might lead to degrade performance or quality on previously generated summaries
- Performance evaluation for evaluating the quality of the summaries whether our models is improving, staying stable or relevant over different iterations

**additional informations**, the measurement using BERT score is used to evaluate the similarity between the generated summaries and the baseline summary. This is a simple yet effective way to measure the quality of the summaries, but it is not perfect and can be subject to biases or even the generated summaries can be identified as hallucinated words

We acknowledge and fully aware with this hindrances, however, this current proof-of-concept doesn't aim to incorporate fine-tuning our models to improve the quality of the summaries, since at the beginning of this project is used for QA testing and with that, all the upcoming fine tuning or evaluation of the generative AI models may unecessary within this proof-of-concept

### Potential considerations or limitations

The "golden dataset or baseline" component could benefit from more specificity about how we maintain and update this reference data over time. Golden datasets can become stale or biased if not carefully curated. Other than that, we might want to consider the dataset scalability specifically for human evaluation can become a bottleneck

This is also applicable to the use-case based testing scenarios, which could be a potential limitation for the scalability of this proof-of-concept for over time (assuming, there will be a major overhaul or change in the use-case, we need to update it regurlarly)

### References

[1] - [Subjective Topic meets LLMs: Unleashing Comprehensive, Reflective and Creative Thinking through the Negation of Negation](https://aclanthology.org/2024.emnlp-main.686/) (Lv et al., EMNLP 2024)

[2] - [Li, H., Chu, Y., Yang, K., Copur-Gencturk, Y., & Tang, J. (2025). LLM-based Automated Grading with Human-in-the-Loop. ArXiv, abs/2504.05239.](https://arxiv.org/abs/2504.05239)

### License

Do anything with this code, but all copyrights and authorships belong to [MauKerja Malaysia](https://www.maukerja.my/en/about)