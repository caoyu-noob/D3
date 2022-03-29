import json
import numpy as np

from transformers.tokenization_gpt2 import GPT2Tokenizer

from itertools import chain

IGNORE_TOKENS = ['i', 'my', 'he', 'she', '.', 'am', 'was', 'is', 'are', 'have', 'has', 'had']

def read_txt_data(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue

        space_idx = line.find(' ')
        if space_idx == -1:
            dialog_idx = int(line)
        else:
            dialog_idx = int(line[:space_idx])

        if int(dialog_idx) == 1:
            data.append({'persona_info': [], 'dialog': []})

        dialog_line = line[space_idx + 1:].split('\t')
        dialog_line = [l.strip() for l in dialog_line]

        if dialog_line[0].startswith('your persona:'):
            persona_info = dialog_line[0].replace('your persona: ', '')
            if persona_info[-1] == '.' and persona_info[-2] != ' ':
                persona_info = persona_info[:-1] + ' .'
            data[-1]['persona_info'].append(persona_info)
        elif len(dialog_line) > 1:
            data[-1]['dialog'].append(dialog_line[0])
            data[-1]['dialog'].append(dialog_line[1])
    return data

def get_dataset_and_history_positions(data):
    dataset = []
    positions = []
    for chat in data:
        persona_info = [s.split() for s in chat['persona_info']]
        dialog = []
        for i, replica in enumerate(chat['dialog'], 1):
            dialog.append(replica.split())
            if not i % 2:
                dataset.append((persona_info, dialog[:], []))
                persona_len = [len(x) for x in persona_info]
                dialog_len = [len(x) for x in dialog]
                persona_pos, history_pos = [], []
                p = 1
                for l in persona_len:
                    p = p + l
                    persona_pos.append(p)
                for j in range(max(len(dialog_len) - 6, 0), len(dialog_len) - 1):
                    p = p + 1 + dialog_len[j]
                    history_pos.append(p)
                positions.append([persona_pos, history_pos])
    for i, data in enumerate(dataset):
        dataset[i] = [[' '.join(p) for p in data[0]], [' '.join(u) for u in data[1]]]
    return dataset, positions

def _get_entail_index_matched_token_positions(entail_data, tokenizer):
    all_attention_positions = []
    matched_token_positions = []
    for i in range(len(entail_data)):
        entail_sample = entail_data[i]
        raw_idx = raw_sample_idx[i]
        if tokenizer is None:
            persona = entail_sample[0].split()
            response = entail_sample[2].split()
        else:
            persona = [t[1:] if t[0] == 'Ġ' else t for t in tokenizer.tokenize(entail_sample[0])]
            response = [t[1:] if t[0] == 'Ġ' else t for t in tokenizer.tokenize(entail_sample[2])]
        target_positions = []
        for i in range(len(persona)):
            for j in range(len(response)):
                if persona[i] not in IGNORE_TOKENS and persona[i] == response[j]:
                    target_positions.append([i, j])
        all_attention_positions.append([raw_idx, entail_sample[0]])
        matched_token_positions.append(target_positions)
    return all_attention_positions, matched_token_positions

def get_dataset_and_persona_positions(raw_data, entail_data, tokenizer):
    all_attention_positions, all_matched_token_positions = _get_entail_index_matched_token_positions(entail_data, tokenizer)
    raw_sample_idx_set = set([s[0] for s in all_attention_positions])
    all_dataset = []
    index = 0
    for chat in raw_data:
        persona_info = [s for s in chat['persona_info']]
        dialog = []
        for i, replica in enumerate(chat['dialog'], 1):
            dialog.append(replica)
            if not i % 2:
                all_dataset.append((persona_info, dialog[:], []))
    new_dataset = [all_dataset[i] for i in [x[0] for x in all_attention_positions]]

    all_target_persona_sentence_positions = []
    all_persona_positions = []
    for i in range(len(all_attention_positions)):
        target_persona = all_attention_positions[i][1]
        cur_data = new_dataset[i]
        target_persona_text_index = 0
        while target_persona != cur_data[0][target_persona_text_index]:
            target_persona_text_index += 1
        if tokenizer is None:
            tokenized_personas = [p.split() for p in cur_data[0]]
        else:
            tokenized_personas = [tokenizer.tokenize(p) for p in cur_data[0]]
        target_persona_start_token_index = 1 + sum([len(x) for x in tokenized_personas[:target_persona_text_index]])
        target_persona_end_token_index = target_persona_start_token_index + \
                                         len(tokenized_personas[target_persona_text_index])
        all_target_persona_sentence_positions.append([target_persona_start_token_index, target_persona_end_token_index])
        all_persona_length = 1 + sum([len(x) for x in tokenized_personas])
        all_persona_positions.append([1, all_persona_length])
        if len(all_matched_token_positions[i]) > 0:
            for j, positions in enumerate(all_matched_token_positions[i]):
                all_matched_token_positions[i][j][0] = all_matched_token_positions[i][j][0] + \
                                                       target_persona_start_token_index
    return new_dataset, all_target_persona_sentence_positions, all_persona_positions, all_matched_token_positions

with open('base_data/th0.99_dev_self.json', 'r') as f:
    entail_data = json.load(f)
with open('base_data/th0.99_dev_raw_sample_idx.json', 'r') as f:
    raw_sample_idx = json.load(f)
data = read_txt_data('../datasets/ConvAI2/valid_self_original.txt')
gpt2_special_sumbol = 'Ġ'

MODE = 'persona'
MODEL = 'gpt2'

if MODE == 'persona':
    tokenizer = None
    if MODEL == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('../gpt2-small')
    new_dataset, target_persona_sentence_position, all_persona_positions, matched_token_positions = \
            get_dataset_and_persona_positions(data, entail_data, tokenizer)
    with open('th0.99_consistent_dataset.json', 'w') as f:
        json.dump(new_dataset, f)
    with open('th0.99_consistent_positions.json', 'w') as f:
        json.dump({'token_positions': matched_token_positions,
                   'target_persona_positions': target_persona_sentence_position,
                   'whole_persona_positions': all_persona_positions}, f)
else:
    data, positions = get_dataset_and_history_positions(data)
    with open('attention_dev_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)
    with open('attention_dev_position.json', 'w', encoding='utf-8') as f:
        json.dump(positions, f)

print('111')