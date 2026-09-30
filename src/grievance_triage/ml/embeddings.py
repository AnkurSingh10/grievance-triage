from functools import lru_cache

from sentence_transformers import SentenceTransformer

from ..core.settings import get_settings


@lru_cache
def get_embedding_model():
    settings = get_settings()
    model = SentenceTransformer(settings.embedding_model)
    dim = getattr(model, "get_embedding_dimension", getattr(model, "get_sentence_embedding_dimension", None))()
    if dim != settings.embedding_dimension:
        raise ValueError("EMBEDDING_DIMENSION does not match the selected Sentence Transformer model")
    return model



def generate_embedding(text: str) -> list[float]:
    return get_embedding_model().encode(text, normalize_embeddings=True).tolist()


def generate_embeddings(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    return get_embedding_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()
