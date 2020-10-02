#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
from transformers import RobertaTokenizer, RobertaForMaskedLM
logging.basicConfig(level=logging.INFO)# OPTIONAL



tokenizer = RobertaTokenizer.from_pretrained('models/roberta/weights')
model = RobertaForMaskedLM.from_pretrained('models/roberta/weights')
model.eval()
# model.to('cuda')  # if you have gpu


#https://stackoverflow.com/questions/54978443/predicting-missing-words-in-a-sentence-natural-language-processing-model
def predict_masked_sent(text, top_k=5):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    #print(tokenizer.lang2id)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))

        
#predict_masked_sent("La meva mare [MASK] molt guapa.", top_k=20)
predict_masked_sent("El meu para és una bona [MASK].", top_k=50)
