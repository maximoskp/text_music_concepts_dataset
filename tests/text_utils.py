import tensorflow as tf
from tensorflow import keras
import keras_nlp
import wikipedia as wp
import numpy as np

# initialize tokenizer
SEQ_LENGTH = 512
vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_file,
    sequence_length=SEQ_LENGTH,
    lowercase=True,
    strip_accents=True,
)
def get_tokenized_text_part( text_part ):
    random_part = text_part[ np.random.randint( len(text_part)-SEQ_LENGTH ): ]
    return tokenizer.tokenize( random_part )
# end get_tokenized_text_part

def detokenize_part( tokenized_part ):
    d = tokenizer.detokenize( tokenized_part )
    return d.numpy().decode('utf-8')
# end detokenize_part