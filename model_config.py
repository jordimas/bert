
import json

config = {
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.3,
  "hidden_size": 128,
  "initializer_range": 0.02,
  "num_attention_heads": 1,
  "num_hidden_layers": 1,
  "vocab_size": vocab_size,
  "intermediate_size": 256,
  "max_position_embeddings": 256
}

with open("models/roberta/config.json", 'w') as fp:
    json.dump(config, fp, indent=4)

