import os
from tokenizers import ByteLevelBPETokenizer

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
#tokenizer.enable_truncation(max_length=512)

# Customize training

vocab_size=50265
path='corpus/ca_dedup.txt'
# Customize training
tokenizer.train(files=path,
                vocab_size=50265,
                min_frequency=2,
                special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

# Save files to disk
directory = "models/roberta"

if not os.path.exists(directory):
    os.makedirs(directory)

tokenizer.save("models/roberta")
