import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from tqdm import tqdm
from typing import Union

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelWithLMHead
from transformers.data.processors.utils import InputExample, InputFeatures

class EntailmentScorer:
    def __init__(self, pred_file, entail_idx_file, model_path, device):
        with open(pred_file, 'r') as f:
            lines = f.readlines()
        self.all_preds = [line.strip() for line in lines]
        with open(entail_idx_file, 'r') as f:
            refs = json.load(f)
        self.all_data = []
        for ref in refs:
            persona = ref['persona']
            idx = ref['index']
            for i in idx:
                self.all_data.append([[p, self.all_preds[i]] for p in persona])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.device = device

    def _convert_examples_to_features(self, examples, tokenizer, label_list=None, max_length=128, output_mode=None,):
        if max_length is None:
            max_length = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample) -> Union[int, float, None]:
            if example.label is None:
                return None
            if output_mode == "classification":
                return label_map[example.label]
            elif output_mode == "regression":
                return float(example.label)
            raise KeyError(output_mode)

        labels = [label_from_example(example) for example in examples]

        batch_encoding = tokenizer(
            [(example.text_a, example.text_b) for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        return features

    def calculate_entailment_score(self):
        self.model.eval()
        entailed_results = []
        with torch.no_grad():
            for i in tqdm(range(len(self.all_data))):
                cur_data = self.all_data[i]
                cnt = 0
                input_examples = []
                for sample in cur_data:
                    input_examples.append(InputExample(str(cnt), sample[0], sample[1], '0'))
                    cnt += 1
                features = self._convert_examples_to_features(
                    input_examples,
                    self.tokenizer,
                    label_list=['0', '1'],
                    max_length=128,
                    output_mode='classification',
                )
                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
                all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(self.device)
                dataset = TensorDataset(all_input_ids, all_attention_mask)
                train_dataloader = DataLoader(dataset, batch_size=8)
                all_logits = None
                for batch in train_dataloader:
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                    outputs = self.model(**inputs)
                    if all_logits is None:
                        all_logits = outputs[0].detach()
                    else:
                        all_logits = torch.cat((all_logits, outputs[0]), dim=0)
                results = torch.argmax(all_logits, dim=1)
                entailed_results.append(torch.sum(results - 1).item())
        return sum(entailed_results)/len(entailed_results)
