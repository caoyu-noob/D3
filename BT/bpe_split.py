import sentencepiece as spm
import sys

split_num = int(sys.argv[1])
sp=spm.SentencePieceProcessor()
sp.load('sentence.bpe.model')

with open('th0.99_entail_dev.txt', 'r') as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    nl = sp.EncodeAsPieces(line.strip())
    new_lines.append(' '.join(nl) + '\n')
file_name = 'th0.99_entail_dev_bpe.txt'
avg_size = len(new_lines) // split_num
start = 0
for i in range(split_num):
    if i != split_num - 1:
        sub_lines = new_lines[start:start + avg_size]
    else:
        sub_lines = new_lines[start:]
    start += avg_size
    with open(file_name + str(i), 'w') as f:
        f.writelines(sub_lines)

