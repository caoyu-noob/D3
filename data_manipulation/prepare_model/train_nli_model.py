import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import json
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers.data.processors.utils import InputExample
from transformers import glue_convert_examples_to_features as convert_examples_to_features

EPOCHS = 2
LR = 1e-5
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.05
EVAL_INTERVAL = 1000
BATCH_SIZE = 32
MAX_GRAD_NORM = 1.0

INPUT_MODEL_PATH = './roberta_mnli'
OUTPUT_MODEL_FILE = 'best_model.bin'

def get_input_examples(data):
    input_examples = []
    label_dict = {'negative': '0', 'neutral': '1', 'positive': '2'}
    for d in data:
        input_examples.append(InputExample(d['id'], d['sentence1'], d['sentence2'], label_dict[d['label']]))
    return input_examples

def eval_model(model, dev_dataloader, prev_best, step):
    dev_tqdm_data = tqdm(dev_dataloader, desc='Evaluation (step #{})'.format(step))
    eval_loss = 0
    model.eval()
    preds, out_label_ids = None, None
    eval_step = 0
    with torch.no_grad():
        for batch in dev_tqdm_data:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_step += 1
            eval_loss += tmp_eval_loss.mean().item()
            dev_tqdm_data.set_postfix({'loss': eval_loss / eval_step})
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        preds = np.argmax(preds, axis=1)
        accuracy = (preds == out_label_ids).astype(np.float32).mean().item()
    if accuracy > prev_best:
        print('Current model BEATS the previous best model, previous best is {:.3f}, current is {:.3f}'.format(prev_best, accuracy))
        torch.save(model.state_dict(), OUTPUT_MODEL_FILE)
        prev_best = accuracy
    else:
        print('Current model CANNOT BEAT the previous best model, previous best is {:.3f}, current is {:.3f}'.format(prev_best, accuracy))
    return prev_best

with open('dialogue_nli_dataset/dialogue_nli_train.jsonl', 'r') as f:
    train_data = json.load(f)
with open('dialogue_nli_dataset/dialogue_nli_dev.jsonl', 'r') as f:
    dev_data = json.load(f)
train_examples = get_input_examples(train_data)
dev_examples = get_input_examples(dev_data)

tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(INPUT_MODEL_PATH)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
if torch.cuda.device_count() > 1:
    device = torch.device('cuda:0')
    model = model.to(device)
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

train_features = convert_examples_to_features(
        train_examples,
        tokenizer,
        label_list=['0', '1', '2'],
        max_length=128,
        output_mode='classification',
    )
dev_features = convert_examples_to_features(
        dev_examples,
        tokenizer,
        label_list=['0', '1', '2'],
        max_length=128,
        output_mode='classification',
    )
train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).to(device)
train_attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long).to(device)
train_labels = torch.tensor([f.label for f in train_features], dtype=torch.long).to(device)
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
dev_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long).to(device)
dev_attention_mask = torch.tensor([f.attention_mask for f in dev_features], dtype=torch.long).to(device)
dev_labels = torch.tensor([f.label for f in dev_features], dtype=torch.long).to(device)
dev_dataset = TensorDataset(dev_input_ids, dev_attention_mask, dev_labels)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

#eval_model(model, dev_dataloader, 0, 0)

t_total = len(train_dataloader) * EPOCHS
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_RATIO, num_training_steps=t_total)

prev_best = 0
for epoch in range(EPOCHS):
    total_loss = 0.0
    tqdm_data = tqdm(train_dataloader, desc='Train (epoch #{})'.format(epoch + 1))
    step = 0
    for batch in tqdm_data:
        model.train()
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs[0]
        loss = loss.mean()
        loss.backward()
        total_loss += loss.item()
        step += 1
        tqdm_data.set_postfix({'loss': total_loss / step})
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step % EVAL_INTERVAL == 0:
            prev_best = eval_model(model, dev_dataloader, prev_best, step)
        if step >= 32:
            break
    prev_best = eval_model(model, dev_dataloader, prev_best, step)

