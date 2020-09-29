#! pip install tokenizers

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

#paths = [str(x) for x in Path("./corpu").glob("**/*.txt")]

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
paths=['corpus/ca_dedup.txt']
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save files to disk
tokenizer.save_model("models", "catalan-tokenizer")
