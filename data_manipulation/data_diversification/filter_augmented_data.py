import json
from sys import argv
from metrics import cal_novelty

PPL_WEIGHT = 0.2
ENTAILMENT_WEIGHT = 0.6
COHERENCE_WEIGHT = 0.2
PPL_NORMALIZER = 50
TH = 0.15

def cal_new_token(data):
    token_set = set()
    for d in data:
        for t in d[0].split():
            token_set.add(t)
        for s in d[1]:
            for t in s.split():
                token_set.add(t)
        for t in d[2].split():
            token_set.add(t)
    return len(token_set)

def get_novelty(original_data, new_data):
    original_sentence_persona, original_sentence_utterances = set(), set()
    new_persona, new_utterances = [], []
    for d in original_data:
        original_sentence_persona.add(d[0])
        original_sentence_utterances.add(d[2])
        original_sentence_utterances.add(d[1][-1])
    for d in new_data:
        new_persona.append(d[0])
        new_utterances.append(d[2])
        new_utterances.append(d[1][-1])
    n1, n2, n3, n4 = cal_novelty(list(original_sentence_persona), new_persona)
    un1, un2, un3, un4 = cal_novelty(list(original_sentence_utterances), new_utterances)
    print('111')

PUNCS = [',', '.', ';', '!', '?', ':']
TYPE_MAP = {(0, 0, 0): 'M_P_G', (0, 0, 1): 'M_P_R', (0, 1, 0): 'M_O_G', (0, 1, 1): 'M_O_R',
            (1, 0, 0): 'P_P_G', (1, 0, 1): 'P_P_R', (1, 1, 0): 'P_O_G', (1, 1, 1): 'P_O_R'}


input_prefix = argv[1]
with open(input_prefix + '_with_replace_response.json', 'r') as f:
    replace_data = json.load(f)
with open(input_prefix + '_with_replace_response_type.json', 'r') as f:
    replace_types = json.load(f)
replace_raw_idx = [d for d in replace_types['raw_data_idx']]
replace_types = [d + [1] for d in replace_types['data_type']]
with open(input_prefix + '_with_generated_response.json', 'r') as f:
    generated_data = json.load(f)
with open(input_prefix + '_without_response_type.json', 'r') as f:
    generated_types = json.load(f)
generated_raw_idx = [d for d in generated_types['raw_data_idx']]
generated_types = [d + [0] for d in generated_types['data_type']]
with open(input_prefix + '_with_replace_response_scores.json', 'r') as f:
    replace_scores = json.load(f)
with open(input_prefix + '_with_generated_response_scores.json', 'r') as f:
    generated_scores = json.load(f)

# generated_scores = {'ppls':[], 'entailment_scores': [], 'coherence_scores': []}
all_data = replace_data[:len(replace_data)] + generated_data[:len(generated_data)]
all_types = replace_types[:len(replace_data)] + generated_types[:len(generated_data)]
all_raw_idx = replace_raw_idx[:len(replace_data)] + generated_raw_idx[:len(generated_data)]
all_scores= {}
for k, v in replace_scores.items():
    all_scores[k] = replace_scores[k] + generated_scores[k]

weighted_scores = []
for i in range(len(all_data)):
    weighted_scores.append(-all_scores['ppls'][i] / PPL_NORMALIZER * PPL_WEIGHT +
                           all_scores['entailment_scores'] * ENTAILMENT_WEIGHT +
                               all_scores['coherence_scores'][i] * COHERENCE_WEIGHT)

res = []
res_types = []
raw_idx_map = {}
selected_raw_idx = []
for i in range(len(weighted_scores)):
    if not raw_idx_map.__contains__(all_raw_idx[i]):
        raw_idx_map[all_raw_idx[i]] = [[], []]
    if weighted_scores[i] > TH:
        res.append(all_data[i])
        res_types.append(all_types[i])
        if all_types[i] == [1,0,0]:
            raw_idx_map[all_raw_idx[i]][0].append(all_data[i])
        selected_raw_idx.append(all_raw_idx[i])
    else:
        raw_idx_map[all_raw_idx[i]][1].append(all_data[i])
type_cnt = {}
for res_type in res_types:
    if not type_cnt.__contains__(TYPE_MAP[tuple(res_type)]):
        type_cnt[TYPE_MAP[tuple(res_type)]] = 0
    type_cnt[TYPE_MAP[tuple(res_type)]] += 1
print('The new augmented data number is ' + str(len(res)))
with open('base_data/th0.99_model_entail_train_self.json', 'r') as f:
    original_data = json.load(f)
# get_novelty(original_data, res)
with open('th0.99_self_augmented_no_persona_filter.json', 'w') as f:
    json.dump(original_data + res, f)
print('111')
