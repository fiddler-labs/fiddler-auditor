from sentence_transformers import SentenceTransformer
from sentence_transformers import util as sent_util

DEFAULT_SENTENCE_XFMR = 'sentence-transformers/paraphrase-mpnet-base-v2'


def load_similarity_model(
    model_name: str = DEFAULT_SENTENCE_XFMR,
    device: str = 'cpu',
):
    sent_xfmer = SentenceTransformer(
        model_name,
        device=device,
    )
    return sent_xfmer


def compute_similarity(
    sentence_model: SentenceTransformer,
    reference_sentence: str,
    perturbed_sentence: str,
):
    ref_emb = sentence_model.encode(
        sentences=[reference_sentence],
        convert_to_tensor=True
    )
    perturbed_emb = sentence_model.encode(
        sentences=[perturbed_sentence],
        convert_to_tensor=True
    )
    score = sent_util.cos_sim(ref_emb, perturbed_emb)
    return score.cpu().numpy()[0][0]
