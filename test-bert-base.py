import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)# OPTIONAL



tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
model.eval()
# model.to('cuda')  # if you have gpu

sentences = None
with open('test-corpus/test.txt') as f:
    sentences = f.readlines()


#https://stackoverflow.com/questions/54978443/predicting-missing-words-in-a-sentence-natural-language-processing-model
def get_prediction(text, top_k=30):
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

    words = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        words.append(predicted_token)
        #print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))

    best = words[0]
    return best, words

        
#predict_masked_sent("La meva mare [MASK] molt guapa.", top_k=20)

def mask_sentence(sentence, pos):
    words = sentence.split()
    if pos >= len(words):
        pos = 0

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
        sentence = sentence.replace("<mask>", "[MASK]")

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

