import json
import torch
import math
import torch.nn as nn
from tqdm import tqdm
from sys import argv

from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers.tokenization_gpt2 import GPT2Tokenizer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import InputExample

input_file = argv[1]
output_file = argv[2]
BATCH_SIZE = 16

def calculate_ppls(responses, device):
    print('calculate PPL scores...')
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_utterance_model')
    model = GPT2LMHeadModel.from_pretrained('./gpt2_utterance_model')
    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    ppls = []
    for r in tqdm(responses):
        inputs = tokenizer(r, return_tensors='pt').data
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        loss = model(**inputs, labels=inputs['input_ids'])[0]
        ppls.append(math.exp(loss.item()))
    return ppls

def calculate_nli_scores(sentences1, sentences2, model, tokenizer, device):
    input_examples = []
    cnt = 0
    for i in range(len(sentences1)):
        input_examples.append(InputExample(str(cnt), sentences1[i], sentences2[i], '0'))
        cnt += 1
    features = convert_examples_to_features(
        input_examples,
        tokenizer,
        label_list=['0', '1', '2'],
        max_length=128,
        output_mode='classification',
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
    dataset = TensorDataset(all_input_ids, all_attention_mask)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    model.eval()
    all_probs = None
    const = torch.tensor([-1, 0, 1], device=device)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = model(**inputs)
            cur_scores = torch.sum(nn.Softmax(dim=-1)(outputs[0].detach()) * const, dim=-1)
            if all_probs is None:
                all_probs = cur_scores.cpu()
            else:
                all_probs = torch.cat((all_probs, cur_scores.cpu()), dim=0)
    scores = all_probs.tolist()
    return scores

def calculate_entailment_scores(personas, responses, device):
    print('calculate entailment scores...')
    tokenizer = AutoTokenizer.from_pretrained('../roberta_mnli')
    model = AutoModelForSequenceClassification.from_pretrained('../roberta_mnli')
    entailment_scores = calculate_nli_scores(personas, responses, model, tokenizer, device)
    return entailment_scores

def calculate_coherence_scores(history, responses, device):
    print('calculate coherence scores...')
    tokenizer = AutoTokenizer.from_pretrained('../coherence_nli_model')
    model = AutoModelForSequenceClassification.from_pretrained('../coherence_nli_model')
    coherence_scores = calculate_nli_scores(history, responses, model, tokenizer, device)
    return coherence_scores

if __name__ == '__main__':
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    responses = [d[2] for d in input_data]
    personas = [d[0] for d in input_data]
    history = [d[1][-1] for d in input_data]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ppls = calculate_ppls(responses, device)
    entailment_scores = calculate_entailment_scores(personas, responses, device)
    coherence_scores = calculate_coherence_scores(history, responses, device)
    with open(output_file, 'w') as f:
        json.dump({'ppls': ppls, 'entailment_scores': entailment_scores, 'coherence_scores': coherence_scores}, f)