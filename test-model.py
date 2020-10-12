#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

tokenizer = RobertaTokenizer.from_pretrained('models/roberta/') 
model = RobertaForMaskedLM.from_pretrained('models/roberta/output/checkpoint-500000/')

sentences = None
with open('test-corpus/test.txt') as f:
    sentences = f.readlines()

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
        idx = torch.topk(mask_hidden_state, k=50, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        list_of_list.append(words)
        #print ("Mask ",index+1,"Guesses : ",words)
    
    best_guess = ""
    for j in list_of_list:
        best_guess = best_guess+""+j[0]
        
    return words, best_guess

def mask_sentence(sentence, pos):
    words = sentence.split()
    word = words[pos]
    sentence = sentence.replace(word, '<mask>', 1)

    return sentence, word

predicted = 0
not_predicted = 0

for sentence_org in sentences:

    sentence_org = sentence_org.replace("\n","")
    n_words = len(sentence_org.split())
    for pos in range(0, n_words):
        
        sentence, word = mask_sentence(sentence_org, pos)
        print(f"sentence: {sentence}, word: {word}, pos {pos}")
        words, best_guess = get_prediction(sentence)
    #    print(words)
        if word in words:
            predicted = predicted + 1
        else:
            not_predicted = not_predicted + 1

tot = predicted + not_predicted
p_predicted = predicted * 100 /tot;
p_not_predicted = not_predicted * 100 /tot;
print(f"predicted: {predicted} ({p_predicted:.2f}%)")
print(f"not_predicted: {not_predicted} ({p_not_predicted:.2f}%)")

