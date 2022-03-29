import json
import pickle
from sys import argv

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import InputExample

NLI_MODEL_PATH = './persona_nli'

# The original train file
input_file = argv[1]
# The output file that saves the NLI logits given the train samples
output_file = argv[2]

def get_dataloader(input_examples, tokenizer, device):
    features = convert_examples_to_features(
        input_examples,
        tokenizer,
        label_list=['0', '1'],
        max_length=128,
        output_mode='classification',
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
    dataset = TensorDataset(all_input_ids, all_attention_mask)
    dataloader = DataLoader(dataset, batch_size=6)
    return dataloader


def read_txt_data(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    cur_persona1, cur_persona2 = [], []
    cur_dialogs1, cur_dialogs2 = [], []
    personas, dialogs = [], []
    sentence_pairs = []
    start = True
    for line in lines:
        if 'your persona:' in line or 'partner\'s persona' in line:
            if start and len(cur_persona1) > 0:
                personas.append(cur_persona1)
                personas.append(cur_persona2)
                dialogs.append(cur_dialogs1)
                dialogs.append(cur_dialogs2)
                cur_persona1, cur_persona2 = [], []
                cur_dialogs1, cur_dialogs2 = [], []
            start = False
            if 'your persona:' in line:
                persona_index = line.find('your persona:')
                persona = line[persona_index + 14: -1]
                cur_persona1.append(persona)
            elif 'partner\'s persona' in line:
                persona_index = line.find('partner\'s persona:')
                persona = line[persona_index + 19: -1]
                cur_persona2.append(persona)
        else:
            start = True
            space_index = line.find(' ')
            sents = line[space_index + 1:].split('\t')
            cur_dialogs1.append(sents[1])
            cur_dialogs2.append(sents[0])
    return personas, dialogs


def read_json_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    examples = []
    cnt = 0
    for d in data:
        examples.append(InputExample(str(cnt), d[0], d[2], '0'))
        cnt += 1
    return examples

tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()
pred_results = []
if '.json' in input_file:
    all_logits = None
    input_examples = read_json_data(input_file)
    train_dataloader = get_dataloader(input_examples, tokenizer, device)
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            outputs = model(**inputs)
            if all_logits is None:
                all_logits = outputs[0].cpu().detach()
            else:
                all_logits = torch.cat((all_logits, outputs[0].cpu().detach()), dim=0)
    all_logits = all_logits.numpy()
    with open(output_file, 'wb') as f:
        pickle.dump(all_logits, f)
else:
    personas, dialogs = read_txt_data(input_file)
    entailed_results = []
    with torch.no_grad():
        for i in tqdm(range(len(personas))):
            cur_persona = personas[i]
            cur_dialogs = dialogs[i]
            cnt = 0
            cur_pred_results = []
            for persona in cur_persona:
                input_examples = []
                for dialog in cur_dialogs:
                    input_examples.append(InputExample(str(cnt), persona, dialog, '0'))
                    cnt += 1
                train_dataloader = get_dataloader(input_examples, tokenizer, device)
                all_logits = None
                for batch in train_dataloader:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                    outputs = model(**inputs)
                    if all_logits is None:
                        all_logits = outputs[0].detach()
                    else:
                        all_logits = torch.cat((all_logits, outputs[0].detach()), dim=0)
                results = torch.argmax(all_logits, dim=1)
                for j, r in enumerate(results):
                    if r == 2:
                        entailed_results.append((persona, cur_dialogs[j]))
                cur_pred_results.append(all_logits.cpu())
            pred_results.append(cur_pred_results)
    with open('entailed_sentences.json', 'w') as f:
        json.dump(entailed_results, f)
    torch.save(pred_results, 'entailment_scores.bin')
