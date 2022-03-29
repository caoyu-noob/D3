import json
import torch
from sys import argv
from tqdm import tqdm

from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.modeling_bert import BertForPreTraining
from transformers.tokenization_bert import BertTokenizer
from transformers import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.nn import CrossEntropyLoss

LR = 1e-5
BATCH_SIZE = 32
STEPS = 100
input_file = argv[1]
output_model = argv[2]
base_model = 'BERT'

BERT_MODEL_PATH = '../bert_model'
GPT2_MODEL_PATH = '../gpt2-small'

with open(input_file, 'r') as f:
    sentences = json.load(f)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if base_model == 'GPT2':
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH)
elif base_model == 'BERT':
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    tokenizer.pad_token = '[PAD]'
    model = BertForPreTraining.from_pretrained(BERT_MODEL_PATH)
optimizer = AdamW(model.parameters(), lr=LR, correct_bias=True)
model.to(device)

all_inputs = tokenizer(sentences, return_tensors='pt', padding=True).data
dataset = TensorDataset(all_inputs['input_ids'], all_inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
steps = 0
s2s_loss = 0
while True:
    tqdm_data = tqdm(dataloader)
    for batch in tqdm_data:
        optimizer.zero_grad()
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        if base_model == 'GPT2':
            loss = model(**inputs, labels=inputs['input_ids'])[0]
        else:
            logits = model(**inputs)[0]
            loss_fct = CrossEntropyLoss()
            labels = inputs['input_ids']
            labels = labels.masked_fill((labels==0).long(), -100)
            loss = loss_fct(logits.view(-1, model.config.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        tqdm_data.set_postfix({'s2s_loss': loss.item()})
        steps += 1
        if steps > STEPS:
            break
    if steps > STEPS:
        break
torch.save(model.state_dict(), output_model)