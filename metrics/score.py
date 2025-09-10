import torch
from torch import Tensor
from transformers import BertTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutput


# in the future, we can use parameterized models here
# instead of hardcoding the model name
# for now, this is sufficient for the proof-of-concept
MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)


def _get_embedding(text: str) -> Tensor:
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)

    # we're not doing training for the model or backpropagation,
    # we just embeddings and it's intentionally used for similiraty checkers
    # for large models that we've trained, this means we need to use or 
    # at least compute hundreds of MB per batch, which is not feasible
    # for the current use case
    with torch.no_grad():
        outputs: BaseModelOutput = model(**input_ids, output_hidden_states=True)

    return outputs.last_hidden_state  # type: Tensor


def _cosine_similarity(candidate: str, reference: str) -> Tensor:
    generated_embedding = torch.nn.functional.normalize(candidate, dim=1)
    reference_embedding = torch.nn.functional.normalize(reference, dim=1)
    return torch.bmm(generated_embedding, reference_embedding.transpose(1, 2))


def _get_precision_score(similarity_score: float):
    return similarity_score.max(dim=2)[0].mean()


def _get_recall_score(similarity_score: float):
    return similarity_score.max(dim=1)[0].mean()


def _get_f1_score(precision_score: float, recall_score: float) -> float:
    return 2 * (precision_score * recall_score) / (precision_score + recall_score)


def bert_score(candidate: str, reference: str) -> float:
    # straightforward implementation of the BERTScore
    # based on the original paper: https://arxiv.org/abs/1904.09675
    candidate_embedding = _get_embedding(candidate)
    reference_embedding = _get_embedding(reference)
    similarity_matrix = _cosine_similarity(candidate_embedding, reference_embedding)
    precision_score = _get_precision_score(similarity_matrix)
    recall_score = _get_recall_score(similarity_matrix)
    f1_score = _get_f1_score(precision_score, recall_score)
    print(f"generated f1 score is: {f1_score:.3f}")
    return f1_score.item()
