import json

INPUT_FILE = '../../datasets/train_self_original.txt'

with open(INPUT_FILE, 'r') as f:
    lines = f.readlines()
personas, utterances = [], []
for line in lines:
    line = line.strip()
    if len(line) == 0:
        continue
    space_idx = line.find(' ')
    if space_idx == -1:
        dialog_idx = int(line)
    else:
        dialog_idx = int(line[:space_idx])
    dialog_line = line[space_idx + 1:].split('\t')
    dialog_line = [l.strip() for l in dialog_line]

    if dialog_line[0].startswith('your persona:'):
        persona_info = dialog_line[0].replace('your persona: ', '')
        if persona_info[-1] == '.' and persona_info[-2] != ' ':
            persona_info = persona_info[:-1] + ' .'
        personas.append(persona_info)
    elif len(dialog_line) > 1:
        utterances.append(dialog_line[1])

with open('personas.json', 'w') as f:
    json.dump(list(set(personas)), f)
with open('responses.json', 'w') as f:
    json.dump(list(set(utterances)), f)