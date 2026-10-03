from sentence_transformers import SentenceTransformer

#pretrained transformer that turns a sentence into 384 numbers representing its meaning
#train.py and chat.py MUST use the same encoder, or the numbers mean different things
ENCODER_NAME = "all-MiniLM-L6-v2"

_encoder = None #loaded the first time encode() is called, not on import

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(ENCODER_NAME)
    return _encoder

def encode(sentences):
    """
    sentences = ["hello there", "what music do you like?"]
    returns a numpy array of shape (2, 384)
    normalize so every vector has length 1 -> only the direction (meaning) matters, not sentence length
    """
    return get_encoder().encode(sentences, normalize_embeddings=True)
