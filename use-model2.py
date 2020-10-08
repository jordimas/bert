#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

tokenizer = RobertaTokenizer.from_pretrained('models/roberta/') 
#tokenizer = RobertaTokenizer.from_pretrained('models/roberta/weights')
model = RobertaForMaskedLM.from_pretrained('models/roberta/output')

sentence = "El meu cotxe és molt millor del que molts ___ ."


def get_prediction (sent):
    
    token_ids = tokenizer.encode(sent, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position ]

    with torch.no_grad():
        output = model(token_ids)

    last_hidden_state = output[0].squeeze()

    list_of_list =[]
    for index,mask_index in enumerate(masked_pos):
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=5, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        list_of_list.append(words)
        print ("Mask ",index+1,"Guesses : ",words)
    
    best_guess = ""
    for j in list_of_list:
        best_guess = best_guess+" "+j[0]
        
    return best_guess


print ("Original Sentence: ",sentence)
sentence = sentence.replace("___","<mask>")
print ("Original Sentence replaced with mask: ",sentence)
print ("\n")

predicted_blanks = get_prediction(sentence)
print ("\nBest guess for fill in the blank :::",predicted_blanks)
