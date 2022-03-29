import json
from sys import argv

# The input file of the distilled data (json file) or the original file
input_file = argv[1]
# The output file that only save the utterances of the input data for back-translation
output_file = argv[2]

new_lines = []
line_idx = []
if '.txt' in input_file:
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        persona_idx = line.find('your persona: ')
        if persona_idx != -1 and persona_idx < 5:
            new_lines.append(line[persona_idx + 14:])
            line_idx.append(i)
        else:
            space_idx = line.find(' ')
            history = line[space_idx + 1:].split('\t')
            new_lines.append(history[0].strip() + '\n')
            new_lines.append(history[1].strip() + '\n')
            line_idx.append(i)
            line_idx.append(i)
if '.json' in input_file:
    with open(input_file, 'r') as f:
        data = json.load(f)
    for i, d in enumerate(data):
        new_lines.append(d[0].strip() + '\n')
        line_idx.append([i, 0])
        for j, u in enumerate(d[1]):
            new_lines.append(u.strip() + '\n')
            line_idx.append([i, 3 + j])
        new_lines.append(d[2].strip() + '\n')
        line_idx.append([i, 2])
with open(output_file, 'w') as f:
    f.writelines(new_lines)
with open(output_file + '_idx.json', 'w') as f:
    json.dump(line_idx, f)