import json
import pickle
from sys import argv

import numpy as np

TH = 0.99

def get_raw_data(original_lines):
    raw_data = []
    all_personas = set()
    for i, line in enumerate(original_lines):
        line = line.strip()

        if len(line) == 0:
            continue

        space_idx = line.find(' ')
        if space_idx == -1:
            dialog_idx = int(line)
        else:
            dialog_idx = int(line[:space_idx])

        if int(dialog_idx) == 1:
            raw_data.append({'persona': [], 'revised_persona': [], 'dialog': []})

        dialog_line = line[space_idx + 1:].split('\t')
        dialog_line = [l.strip() for l in dialog_line]

        if dialog_line[0].startswith('your persona:'):
            persona_info = dialog_line[0].replace('your persona: ', '')
            all_personas.add(persona_info[:-1] + ' .')
            raw_data[-1]['persona'].append(persona_info[:-1] + ' .')
        elif len(dialog_line) > 1:
            raw_data[-1]['dialog'].append(dialog_line[0])
            raw_data[-1]['dialog'].append(dialog_line[1])
    return raw_data, list(all_personas)

# The original train data file
input_file = argv[1]
# The logits given by NLI model obtained before
logits_file = argv[2]
# The output json file that contains all distilled samples that were determined as entailed by the NLI model
output_file = argv[3]

if '.json' in input_file:
    with open(input_file, 'r') as f:
        data = json.load(f)
    cnt1, cnt2, cnt3 = 0, 0, 0
    with open(logits_file, 'rb') as f:
        logits = pickle.load(f)
    entail = np.argmax(logits, axis=-1)
    entail_result = []
    for i, d in enumerate(data):
        cur_logit = logits[i]
        if entail[i] == 0:
            cnt1 += 1
        elif entail[i] == 1:
            cnt2 += 1
        elif entail[i] == 2:
            softmax = np.exp(logits[i]) / np.sum(np.exp(logits[i]))
            if softmax[2] < TH:
                cnt2 += 1
            else:
                cnt3 += 1
                entail_result.append(data[i])
    print(cnt1)
    print(cnt2)
    print(cnt3)
    with open(output_file, 'w') as f:
        json.dump(entail_result, f)
else:
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    raw_data, all_personas = get_raw_data(lines)

    with open(logits_file, 'rb') as f:
        d = pickle.load(f)
    cnt1, cnt2, cnt3 = 0, 0, 0
    entail_result = []
    for dialog in d:
        cur_entail_result = []
        for p in dialog:
            res = np.argmax(p, axis=-1)
            res = list(res)
            for i, r in enumerate(res):
                if r == 2:
                    softmax = np.exp(p[i]) / np.sum(np.exp(p[i]))
                    if softmax[2] < TH:
                        res[i] = 1
            cur_entail_result.append(list(res))
        entail_result.append(cur_entail_result)
    contradict_list = []
    entail_list = []
    neutral_list = []
    entail_sample_idx = []
    sample_idx = 0
    for i, dialog in enumerate(entail_result):
        for j, p in enumerate(dialog):
            for k, u in enumerate(p):
                if u == 0:
                    cnt1 += 1
                    contradict_list.append((i, j, k))
                if u == 1:
                    cnt2 += 1
                if u == 2:
                    cnt3 += 1
                    entail_list.append((i, j, k))
                    entail_sample_idx.append(sample_idx + k)
        sample_idx += len(p)
    print(cnt1)
    print(cnt2)
    print(cnt3)

    entail_data = []
    for idx in entail_list:
        persona = raw_data[idx[0]]['persona'][idx[1]]
        response_idx = idx[2] * 2 + 1
        response = raw_data[idx[0]]['dialog'][response_idx]
        history = raw_data[idx[0]]['dialog'][max(0, response_idx - 3): response_idx]
        if len(history) == 0:
            history = ['__SILENCE__']
        entail_data.append([persona, history, response])
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entail_data, f)
    with open(output_file + 'raw_idx', 'w', encoding='utf-8') as f:
        json.dump(entail_sample_idx, f)
