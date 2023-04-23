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
SEQ_LENGTH = 128

# %% make tokenizer
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab_file,
    sequence_length=SEQ_LENGTH,
    lowercase=True,
    strip_accents=True,
)