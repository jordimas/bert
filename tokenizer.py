#! pip install tokenizers
import os
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

#paths = [str(x) for x in Path("./corpu").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
#tokenizer.enable_truncation(max_length=512)

# Customize training

vocab_size=5000
paths=['corpus/ca_dedup.txt']
tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
directory = "models/roberta"

if not os.path.exists(directory):
    os.makedirs(directory)

tokenizer.save(directory, "roberta")
