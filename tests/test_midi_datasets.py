# %% imports
from musicaiz.tokenizers import MMMTokenizer, MMMTokenizerArguments
from musicaiz.datasets import Maestro

# % check dataset
maestro = Maestro()

# %% path for tokens
output_path = '../data/Maestro/tokenized'

# %% prepare tokenization
args = MMMTokenizerArguments(
    prev_tokens='',
    windowing=True,
    time_unit='HUNDRED_TWENTY_EIGHT',
    num_programs=None,
    shuffle_tracks=True,
    track_density=False,
    window_size=128,
    hop_length=64,
    time_sig=True,
    velocity=True,
)

# %% tokenize dataset
maestro.tokenize(
    dataset_path='../data/Maestro',
    output_path=output_path,
    dataset_version='maestro-v3.0.0',
    output_file='token-sequences',
    args=args,
    tokenize_split="all"
)

# %% get vocabulary
vocab = MMMTokenizer.get_vocabulary(
    dataset_path=output_path
)