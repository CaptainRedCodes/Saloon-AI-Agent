from sentence_transformers import SentenceTransformer

_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _encoder
