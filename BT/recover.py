import json
from sys import argv

from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

bt_file = argv[1]
original_file = argv[2]
index_file = argv[3]
output_file = argv[4]

SYMBOLS = ['.', ',', '!', '?', ';', '"']
SPECIAL1 = '’'

def find_the_most_different_replace(original_sentence, candidates):
    min_bleu = 100
    res = candidates[0]
    for c in candidates:
        bleu = sentence_bleu(original_sentence.lower(), c.lower())
        if bleu < min_bleu and abs(len(original_sentence) - len(c)) < len(original_sentence) * 0.4:
            res = c.lower()
            min_bleu = bleu
    return res

def clean_data_core(line):
    line_list = []
    for c in line:
        if c in SYMBOLS or c == SPECIAL1:
            if len(line_list) > 0 and line_list[-1] != ' ':
                line_list.append(' ')
        if c == SPECIAL1:
            c = '\''
        line_list.append(c)
    new_sentence = ''.join(line_list)
    end_idx = len(new_sentence)
    for i in range(len(new_sentence)):
        cur_idx = len(new_sentence) - 1 - i
        if new_sentence[cur_idx] not in SYMBOLS and new_sentence[cur_idx] != ' ':
            break
        else:
            if new_sentence[cur_idx] in SYMBOLS:
                end_idx = cur_idx + 1
    new_line = new_sentence[:end_idx]
    return new_line

def recover_persona_end(cleaned_persona):
    if len(cleaned_persona) >= 2 and cleaned_persona[-1] in SYMBOLS and cleaned_persona[-2] == ' ':
        j = len(cleaned_persona) - 2
        while cleaned_persona[j] == ' ':
            j -= 1
        cleaned_persona = cleaned_persona[:j + 1] + '.'
    elif len(cleaned_persona) > 0 and cleaned_persona[-1] not in SYMBOLS and cleaned_persona[-1] != ' ':
        cleaned_persona = cleaned_persona + '.'
    return cleaned_persona

def clean_data(data, is_json=False):
    if is_json:
        for i, sample in enumerate(data):
            persona, history, response = sample[0], sample[1], sample[2]
            cleaned_persona = clean_data_core(persona)
            cleaned_persona = recover_persona_end(cleaned_persona)
            for j, h in enumerate(history):
                if h != '__SILENCE__':
                   history[j] = clean_data_core(h)
            cleaned_response = clean_data_core(response)
            data[i] = [cleaned_persona, history, cleaned_response]
    else:
        for i, line in enumerate(data):
            line = line.lower().strip()
            if 'your persona: ' in line:
                cleaned_persona = clean_data_core(line)
                if cleaned_persona[-1] in SYMBOLS and cleaned_persona[-2] == ' ':
                    j = len(cleaned_persona) - 2
                    while cleaned_persona[j] == ' ':
                        j -= 1
                    cleaned_persona = cleaned_persona[:j + 1] + '.'
                elif cleaned_persona[-1] not in SYMBOLS and cleaned_persona[-1] != ' ':
                    cleaned_persona = cleaned_persona + '.'
                data[i] = cleaned_persona + '\n'
            else:
                items = line.split('\t')
                for j, s in enumerate(items[:2]):
                    items[j] = clean_data_core(s)
                data[i] = '\t'.join(items) + '\n'
    return data

beam_size = 25
bt_sentences = []
with open(bt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
i = 0
while i < len(lines):
    cur_sentences = [s.strip() for s in lines[i: i + beam_size]]
    i += beam_size
    bt_sentences.append(cur_sentences)
with open(index_file, 'r') as f:
    indices = json.load(f)
if '.txt' in original_file:
    with open(original_file, 'r') as f:
        original_lines = f.readlines()
    prev_line_idx = -1
    for i, line_idx in enumerate(tqdm(indices)):
        cur_bt_sentences = bt_sentences[i]
        line = original_lines[line_idx].strip()
        if 'your persona: ' in line:
            start_index = line.find('your persona: ')
            original_sentence = line[start_index + 14:]
            replace_start, replace_end = start_index + 14, len(line)
        else:
            space_index = line.find(' ')
            items = line[space_index + 1:].split('\t')
            if line_idx != prev_line_idx:
                original_sentence = items[0]
                replace_start, replace_end = space_index + 1, space_index + len(original_sentence)
            else:
                original_sentence = items[1]
                t_index = line.find('\t')
                replace_start, replace_end = t_index + 1, t_index + len(original_sentence) + 1
        replace_sentence = find_the_most_different_replace(original_sentence, cur_bt_sentences)
        original_lines[line_idx] = line[:replace_start] + replace_sentence + line[replace_end:] + '\n'
        prev_line_idx = line_idx
    cleaned_lines = clean_data(original_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(original_lines)
if '.json' in original_file:
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    for i, data_index in enumerate(tqdm(indices)):
        cur_bt_sentences = bt_sentences[i]
        sample_index = data_index[0]
        original_sample = original_data[sample_index]
        if data_index[1] < 3:
            original_sentence = original_sample[data_index[1]]
            replace_sentence = find_the_most_different_replace(original_sentence, cur_bt_sentences)
            original_sample[data_index[1]] = replace_sentence
        else:
            original_sentence = original_sample[1][data_index[1] - 3]
            if original_sentence != '__SILENCE__':
                replace_sentence = find_the_most_different_replace(original_sentence, cur_bt_sentences)
                original_sample[1][data_index[1] - 3] = replace_sentence
        original_data[sample_index] = original_sample
    cleaned_data = clean_data(original_data, is_json=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(original_data, f)
