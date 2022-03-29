import json
from metrics import cal_novelty

with open('original_utterances.json', 'r') as f:
    original_utterances = json.load(f)
with open('new_utterances.json', 'r') as f:
    new_utterances = json.load(f)
un1, un2, un3, un4 = cal_novelty(original_utterances, new_utterances[40000:50000])
print(un1)
print(un2)
print(un3)
print(un4)