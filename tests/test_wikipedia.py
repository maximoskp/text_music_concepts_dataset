#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 07:43:49 2023

@author: max
"""

# %% imports

import tensorflow as tf
from tensorflow import keras
import keras_nlp

import wikipedia as wp

import numpy as np

# %% get a list of search results
search_result = wp.search('major scale')

# %% get page for first result
first_result_page = wp.page( search_result[0] )

# %% get content of page
page_content = first_result_page.content

# %% download vocabulary
# Download vocabulary data.
vocab_file = keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/keras-nlp/examples/bert/bert_vocab_uncased.txt",
)

# %% make constants
SEQ_LENGTH = 512

# %% make tokenizer
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_file,
    sequence_length=SEQ_LENGTH,
    lowercase=True,
    strip_accents=True,
)
# %% tokenize a random part of the page content
random_part = page_content[ np.random.randint( len(page_content)-SEQ_LENGTH ): ]
t = tokenizer.tokenize( random_part )
# detokenize to examine
d = tokenizer.detokenize( t )
s = d.numpy().decode('utf-8')
print(s)
print(len(s.split(' ')))