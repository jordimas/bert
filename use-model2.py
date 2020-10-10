#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

tokenizer = RobertaTokenizer.from_pretrained('models/roberta/') 
model = RobertaForMaskedLM.from_pretrained('models/roberta/output/')

sentences = [ \
    "El meu cotxe és molt millor del que molts ___.",
    "El tribunal considera provat que els acusats van ___ gairebé 24 milions d'euros.",
    "Els principals responsables de l'empresa que ___ la depuradora.",
    "El meu pare és el més ___ del grup",
    "El cotxe està ___.",
    "Tinc tanta son que a les cinc tinc ___.",
]


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
        idx = torch.topk(mask_hidden_state, k=10, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        list_of_list.append(words)
        #print ("Mask ",index+1,"Guesses : ",words)
    
    best_guess = ""
    for j in list_of_list:
        best_guess = best_guess+""+j[0]
        
    return words, best_guess

for sentence in sentences:
    sentence = sentence.replace("___","<mask>")
    print (sentence)
    words,  best_guess = get_prediction(sentence)
    print (f"Predicted words:{words}, best_guess '{best_guess}'\n")
