import json
import pickle
import spacy
import torch
from tqdm import tqdm
import numpy as np
import argparse
import math
from itertools import chain
from bert_score.score import get_bert_score

from transformers import pipeline
from transformers.modeling_bert import BertForMaskedLM
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.data.processors.utils import InputExample

REPLACE_POS = ['VERB', 'NOUN', 'PROPN', 'NUM', 'ADV', 'ADP', 'ADJ']
SPECIAL_TOKENS = ['.', 'i', 'to', 'is', 'am', 'are', 'my', 'the', 'at', 'on', 'in']
PUNCS = [',', '.', ';', '!', '?', ':']
ERROR_MAP = {'fiancé': 'fiance'}
MASK_PROB = 1
CANDIDATE_NUM = 20
MAX_SAMPLE_NUM = 20
MAX_CANDIDATE_TRY = 100
BATCH_SIZE = 8
PPL_WEIGHT = 0.7
BERT_SCORE_WEIGHT = 1 - PPL_WEIGHT
PPL_WEIGHT_FOR_GPT2 = 0.3
BERT_SCORE_WEIGHT_FOR_GPT2 = 1 - PPL_WEIGHT_FOR_GPT2
PPL_NORMALIZER = 100
PPL_NORMALIZER_FOR_GPT2 = 25
BERT_MASK_PERSONA = 0
GPT2_PRED_PERSONA = 1
BT_HISTORY = 0
ORIGINAL_HISTORY = 1
GENERATE_RESPONSE = 0
REPLACE_RESPONSE = 1

class NewPersonaGenerator():
    def __init__(self, args):
        self.input_file = args.input_file
        self.output_file = args.output_file
        self.bert_mask_candidates_num = args.bert_mask_candidates_num
        self.bert_mask_ratio = args.bert_mask_ratio
        self.bert_mask_sample_size = args.bert_mask_sample_size
        self.raw_bert_mask_generation_file = args.raw_bert_mask_generation_file
        self.token_replace_num = args.token_replace_num
        self.gpt2_pred_candidates_num = args.gpt2_pred_candidates_num
        self.gpt2_pred_mask_ratio = args.gpt2_pred_mask_ratio
        self.gpt2_pred_max_length_fixed = args.gpt2_pred_max_length_fixed
        self.gpt2_pred_length_extend_ratio = args.gpt2_pred_length_extend_ratio
        self.raw_gpt2_pred_generation_file = args.raw_gpt2_pred_generation_file
        self.bert_mask_filter_th = args.bert_mask_filter_th
        self.gpt2_pred_filter_th = args.gpt2_pred_filter_th

        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.input_data = data
        personas = [p[0] for p in data]
        personas = self.preprocess_persona(personas)
        self.personas_map = {}
        unique_personas = []
        self.persona_ids = []
        self.persona_id_and_data_id = []
        cnt = 0
        for i, p in enumerate(personas):
            if not self.personas_map.__contains__(p):
                unique_personas.append(p)
                self.personas_map[p] = cnt
                self.persona_id_and_data_id.append([])
                cnt += 1
            self.persona_ids.append(self.personas_map[p])
            self.persona_id_and_data_id[self.personas_map[p]].append(i)
        self.personas = unique_personas
        '''Build the correspondence between words and tokens after bpe'''
        self.spacy_model = spacy.load('en_core_web_md')
        self.processed_personas = [s for s in tqdm(self.spacy_model.pipe(self.personas, batch_size=1000), desc='Spacy process personas')]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.bert_tokenizer = BertTokenizer.from_pretrained('../bert_model')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('../gpt2-small')
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        # self.gpt2_tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>'})

    def generate(self):
        if self.token_replace_num > 0:
            print('Generate New Personas using Token replacement')
        if self.bert_mask_candidates_num > 0:
            print('Generate New Personas using BERT masking')
            bert_mask_new_personas = self.generate_bert_mask_personas(self.processed_personas, self.personas)
            bert_mask_new_personas_ppls = self.get_generation_ppls(bert_mask_new_personas)
            bert_scores_p, bert_scores_r, bert_scores_f = self.get_bert_scores(bert_mask_new_personas, self.personas)
            with open(self.raw_bert_mask_generation_file, 'w', encoding='utf-8') as f:
                json.dump({'generation': bert_mask_new_personas, 'ppls': bert_mask_new_personas_ppls,
                           'bert_scores_p': bert_scores_p, 'bert_scores_r': bert_scores_r, 'bert_scores_f' : bert_scores_f}, f)
        if self.gpt2_pred_candidates_num > 0:
            print('Generate New Personas using GPT2 predicting')
            gpt2_pred_new_personas = self.generate_gpt2_pred_personas(self.personas)
            gpt2_pred_new_personas_ppls = self.get_generation_ppls(gpt2_pred_new_personas)
            bert_scores_p, bert_scores_r, bert_scores_f = self.get_bert_scores(gpt2_pred_new_personas, self.personas)
            with open(self.raw_gpt2_pred_generation_file, 'w', encoding='utf-8') as f:
                json.dump({'generation': gpt2_pred_new_personas, 'ppls': gpt2_pred_new_personas_ppls,
                           'bert_scores_p': bert_scores_p, 'bert_scores_r': bert_scores_r, 'bert_scores_f' : bert_scores_f}, f)
        print('111')

    def get_bert_scores(self, new_personas, original_personas):
        print('Obtain BERT Scores of new generated personas')
        new_persona_list, original_persona_list = [], []
        for i, personas in enumerate(new_personas):
            new_persona_list.extend(personas)
            original_persona_list.extend([original_personas[i]] * len(personas))
        all_preds = get_bert_score(
            new_persona_list,
            original_persona_list,
            model_type='../../roberta_large',
            num_layers=16,
            batch_size=16,
        )
        bert_scores_p, bert_scores_r, bert_scores_f = all_preds[0].tolist(), all_preds[1].tolist(), all_preds[2].tolist()
        bert_scores_p_split, bert_scores_r_split, bert_scores_f_split = [], [], []
        index = 0
        for i, personas in enumerate(new_personas):
            bert_scores_p_split.append(bert_scores_p[index: index + len(personas)])
            bert_scores_r_split.append(bert_scores_r[index: index + len(personas)])
            bert_scores_f_split.append(bert_scores_f[index: index + len(personas)])
            index += len(personas)
        return bert_scores_p_split, bert_scores_r_split, bert_scores_f_split

    def generate_bert_mask_personas(self, processed_personas, personas):
        mask_token_indices = []
        for pi, persona in enumerate(tqdm(processed_personas, desc='get mask indices')):
            words = [t.text for t in persona]
            tokens = self.bert_tokenizer.tokenize(persona.text)
            if len(words) > 10 and words[9] == 'gon' and words[10] == 'na':
                words[9] = 'gonna'
                words.pop(10)
                persona = [t for t in persona]
                persona.pop(10)

            word_i, token_i = 0, 0
            tmp_word = ''
            word_map = [[]]
            while word_i < len(words):
                if ERROR_MAP.__contains__(words[word_i]):
                    words[word_i] = ERROR_MAP[words[word_i]]
                if len(tokens[token_i]) > 2 and tokens[token_i].startswith('##'):
                    tmp_word += tokens[token_i][2:]
                else:
                    tmp_word += tokens[token_i]
                word_map[word_i].append(token_i)
                if words[word_i] == tmp_word:
                    word_i += 1
                    word_map.append([])
                    tmp_word = ''
                token_i += 1
            cur_word_indices = []
            for i, t in enumerate(persona):
                if t.pos_ in REPLACE_POS and t.text not in SPECIAL_TOKENS:
                    cur_word_indices.append(i)
            cur_indices = []
            for word_index in cur_word_indices:
                cur_indices.append(word_map[word_index])
            mask_token_indices.append([cur_word_indices, cur_indices])

        self.model = BertForMaskedLM.from_pretrained('../bert_model')
        self.model.to(self.device)
        new_personas = []
        proper_token_ids = self.get_proper_token_ids_list(self.bert_tokenizer)
        with torch.no_grad():
            for i in tqdm(range(0, len(personas), BATCH_SIZE), desc='Generate masked personas'):
                inputs = self.bert_tokenizer(personas[i: i + BATCH_SIZE], return_tensors='pt', padding=True).data
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                logits = self.model(**inputs)[0]
                logits = logits.detach()
                probs = torch.softmax(logits, dim=-1)
                cur_masks = mask_token_indices[i: i + BATCH_SIZE]
                for j in range(probs.size(0)):
                    candidates = set()
                    mask = cur_masks[j]
                    cur_probs = probs[j]
                    candidate_cnt, all_cnt = 0, 0
                    mask_word_num = max(math.ceil(len(mask[0]) * self.bert_mask_ratio), 1)
                    while candidate_cnt < self.bert_mask_candidates_num:
                        random_word_mask = mask[0]
                        np.random.shuffle(random_word_mask)
                        random_word_mask = sorted(random_word_mask[:mask_word_num])
                        random_token_mask = list(chain(*[mask[1][m] for m in range(len(random_word_mask))]))
                        cur_generation = inputs['input_ids'][j].tolist()
                        for index in random_token_mask:
                            sampled_tokens = torch.multinomial(cur_probs[index + 1], self.bert_mask_sample_size)
                            for sampled_token in sampled_tokens:
                                if sampled_token != cur_generation[index + 1] and proper_token_ids.__contains__(
                                        sampled_token.item()):
                                    cur_generation[index + 1] = sampled_token
                                    break
                        new_persona = self.bert_tokenizer.decode(cur_generation, skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False)
                        if not candidates.__contains__(new_persona):

                            candidates.add(new_persona)
                            candidate_cnt += 1
                            all_cnt += 1
                        if all_cnt > MAX_CANDIDATE_TRY:
                            break
                    new_personas.append(list(candidates))
        return new_personas

    def generate_gpt2_pred_personas(self, personas):
        self.model = GPT2LMHeadModel.from_pretrained('./gpt2_persona_model')
        self.model.to(self.device)
        new_personas = []
        for persona in tqdm(personas, desc='Generate GPT2 preds'):
            persona_tokens = persona.split()
            mask_num = min(math.ceil((len(persona_tokens) - 1) * self.gpt2_pred_mask_ratio) + 1, len(persona_tokens) - 2)
            part_personas = ' '.join(persona_tokens[: -mask_num])
            encoded_input = self.gpt2_tokenizer(part_personas, return_tensors='pt')
            for k, v in encoded_input.items():
                encoded_input[k] = v.to(self.device)
            max_length = max(self.gpt2_pred_max_length_fixed,
                    math.ceil(encoded_input['input_ids'][0].size()[0] / (1 - self.gpt2_pred_mask_ratio) * (1 + self.gpt2_pred_length_extend_ratio)))
            generations = self.model.generate(**encoded_input, do_sample=True, num_return_sequences=50,
                                              max_length=max_length, pad_token_id=50256)
            cur_new_personas = set()
            for i in range(generations.size()[0]):
                new_persona = self.gpt2_tokenizer.decode(generations[i].tolist(), skip_special_tokens=True)
                for j, c in enumerate(new_persona):
                    if c in PUNCS:
                        break
                new_persona = new_persona[: j] + ' .'
                if new_persona != persona:
                    cur_new_personas.add(new_persona.strip())
            new_personas.append(list(cur_new_personas))
        return new_personas

    '''preprocess the persona sentences
    1) replace \'m with am, 2) replace don't with do not, 3) replace can't with cannot '''

    def preprocess_persona(self, personas):
        new_personas = []
        for i, p in enumerate(personas):
            p = p.replace('\'m', ' am')
            p = p.replace('isn\'t', 'is not')
            p = p.replace('shouldn\'t', 'should not')
            p = p.replace('wasn\'t', 'was not')
            p = p.replace('don\'t', 'do not')
            p = p.replace('can\'t', 'can not')
            p = p.replace('cannot', 'can not')
            p = p.replace('couldn\'t', 'could not')
            p = p.replace('doesn\'t', 'does not')
            p = p.replace('didn\'t', 'did not')
            p = p.replace('hasn\'t', 'has not')
            p = p.replace('haven\'t', 'have not')
            p = p.replace('hadn\'t', 'had not')
            p = p.replace('aren\'t', 'are not')
            p = p.replace(' m ', ' am ')
            p = p.replace(' s ', ' is ')
            p = p.replace('4am', '4 am')
            new_personas.append(p)
        return new_personas

    def get_generation_ppls(self, personas):
        print('Obtain PPLs of new generated personas')
        self.model = GPT2LMHeadModel.from_pretrained('../gpt2_persona_model')
        self.model.to(self.device)
        ppls = []
        for generated_personas in tqdm(personas, desc='Get GPT2 ppls'):
            cur_ppls = []
            for persona in generated_personas:
                inputs = self.gpt2_tokenizer(persona, return_tensors='pt').data
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                loss = self.model(**inputs, labels=inputs['input_ids'])[0]
                cur_ppls.append(loss.item())
            ppls.append(cur_ppls)
        return ppls

    def get_proper_token_ids_list(self, tokenizer):
        ids = []
        for c in tokenizer.vocab.keys():
            if c.startswith('##'):
                if ((ord(c[2]) - ord('a')) >= 0 and (ord(c[2]) - ord('z')) <= 0) or \
                        ((ord(c[2]) - ord('0')) >= 0 and (ord(c[2]) - ord('9')) <= 0):
                    ids.append(tokenizer.vocab[c])
            else:
                if ((ord(c[0]) - ord('a')) >= 0 and (ord(c[0]) - ord('z')) <= 0) or \
                        ((ord(c[0]) - ord('0')) >= 0 and (ord(c[0]) - ord('9')) <= 0):
                    ids.append(tokenizer.vocab[c])
        return set(ids)

    def filter_new_personas(self, new_personas, ppls, bert_score, th, filter_gpt2_pred=False):
        filtered_res = []
        cnt = 0
        ppl_normalizer = PPL_NORMALIZER
        ppl_weight, bert_score_weight = PPL_WEIGHT, BERT_SCORE_WEIGHT
        if filter_gpt2_pred:
            ppl_normalizer = PPL_NORMALIZER_FOR_GPT2
            ppl_weight, bert_score_weight = PPL_WEIGHT_FOR_GPT2, BERT_SCORE_WEIGHT_FOR_GPT2
        for i, persona in enumerate(new_personas):
            cur_res = []
            for j, p in enumerate(persona):
                score = ppl_weight * (math.exp(ppls[i][j]) / ppl_normalizer) + bert_score_weight * bert_score[i][j]
                if score < th:
                    if filter_gpt2_pred:
                        if not self.personas_map.__contains__(p):
                            cur_res.append((p, score))
                            cnt += 1
                    else:
                        cur_res.append((p, score))
                        cnt += 1
            filtered_res.append(cur_res)
        return filtered_res, cnt

    def get_filtered_data(self):
        with open(self.raw_bert_mask_generation_file, 'r', encoding='utf-8') as f:
            bert_generation = json.load(f)
        with open(self.raw_gpt2_pred_generation_file, 'r', encoding='utf-8') as f:
            gpt2_generation = json.load(f)
        bert_results, bert_ppls, bert_score = bert_generation['generation'], bert_generation['ppls'],\
                bert_generation['bert_scores_f']
        gpt2_results, gpt2_ppls, gpt2_score = gpt2_generation['generation'], gpt2_generation['ppls'],\
                gpt2_generation['bert_scores_f']
        bert_cnt, gpt2_cnt = 0, 0
        type_indicator = []
        augmented_data, augmented_data_from_raw_idx, augmented_data_with_repalced_responses = [], [], []
        filtered_bert_mask_personas = []
        for i in range(len(bert_results)):
            raw_data_idx = self.persona_id_and_data_id[i]
            results = bert_results[i] + gpt2_results[i][:10]
            bert_index = len(bert_results[i])
            ppls = np.array(bert_ppls[i] + gpt2_ppls[i][:10])
            scores = np.array(bert_score[i] + gpt2_score[i][:10])
            rank_scores = 0.4 * ppls / 50 + 0.6 * (scores - 0.9)
            selected_indices = np.argsort(rank_scores)[:5]
            # selected_indices = list(range(len(results)))
            # np.random.shuffle(selected_indices)
            # selected_indices = selected_indices[:5]
            cur_bert_mask_personas = []
            for index in selected_indices:
                persona = results[index]
                if index < bert_index:
                    bert_cnt += 1
                    cur_bert_mask_personas.append(persona)
                else:
                    gpt2_cnt += 1
                for idx in raw_data_idx:
                    augmented_data.append([persona, self.input_data[idx][1], self.input_data[idx][2]])
                    augmented_data_from_raw_idx.append(idx)
                    if index < bert_index:
                        type_indicator.append([BERT_MASK_PERSONA])
                    else:
                        type_indicator.append([GPT2_PRED_PERSONA])
            filtered_bert_mask_personas.append(cur_bert_mask_personas)
        processed_bert_mask_filtered_personas, bert_mask_filtered_personas, replace_indices, cnt = \
                    self.filter_by_pos(filtered_bert_mask_personas)
        return augmented_data, augmented_data_from_raw_idx, type_indicator, processed_bert_mask_filtered_personas, \
                       replace_indices

    # def get_filtered_data(self):
    #     with open(self.raw_bert_mask_generation_file, 'r', encoding='utf-8') as f:
    #         bert_generation = json.load(f)
    #
    #     bert_mask_filtered_personas, cnt = self.filter_new_personas(bert_generation['generation'], bert_generation['ppls'],
    #             bert_generation['bert_scores_f'], self.bert_mask_filter_th)
    #     '''For personas generated by BERT masking, the new words should have the consistent pos tags to the original ones'''
    #     processed_bert_mask_filtered_personas, bert_mask_filtered_personas, replace_indices, cnt = \
    #         self.filter_by_pos(bert_mask_filtered_personas)
    #     print('Thu number of filtered new personas by bert mask is ' + str(cnt))
    #     with open(self.raw_gpt2_pred_generation_file, 'r', encoding='utf-8') as f:
    #         gpt2_generation = json.load(f)
    #     gpt2_pred_filtered_personas, cnt = self.filter_new_personas(gpt2_generation['generation'], gpt2_generation['ppls'],
    #             gpt2_generation['bert_scores_f'], self.gpt2_pred_filter_th, filter_gpt2_pred=True)
    #     print('Thu number of filtered new personas by GPT2 pred is ' + str(cnt))
    #
    #     augmented_data, augmented_data_from_raw_idx, augmented_data_with_repalced_responses = [], [], []
    #     type_indicator = []
    #     for i in range(len(bert_mask_filtered_personas)):
    #         raw_data_idx = self.persona_id_and_data_id[i]
    #         bert_new_personas = bert_mask_filtered_personas[i]
    #         if len(bert_new_personas) > 0:
    #             for p in bert_new_personas:
    #                 for idx in raw_data_idx:
    #                     augmented_data.append([p[0], self.input_data[idx][1], self.input_data[idx][2]])
    #                     augmented_data_from_raw_idx.append(idx)
    #                     type_indicator.append([BERT_MASK_PERSONA])
    #         gpt2_new_personas = gpt2_pred_filtered_personas[i]
    #         if len(gpt2_new_personas) > 0:
    #             for p in gpt2_new_personas:
    #                 for idx in raw_data_idx:
    #                     augmented_data.append([p[0], self.input_data[idx][1], self.input_data[idx][2]])
    #                     augmented_data_from_raw_idx.append(idx)
    #                     type_indicator.append([GPT2_PRED_PERSONA])
    #     return augmented_data, augmented_data_from_raw_idx, type_indicator, processed_bert_mask_filtered_personas, \
    #            replace_indices

    '''Incorporate bt augmented dialogue history into samples with new generated personas'''
    def integrate_bt_history(self, input_data, input_data_from_raw_idx, bt_augmented_files, type_indicator):
        incorporated_data, incorporated_type_indicator = [], []
        if bt_augmented_files is not None:
            for bt_augmented_file in bt_augmented_files:
                with open(bt_augmented_file, 'r', encoding='utf-8') as f:
                    bt_augmented_data = json.load(f)
                for i, raw_idx in enumerate(input_data_from_raw_idx):
                    incorporated_data.append([input_data[i][0], bt_augmented_data[raw_idx][1], input_data[i][2]])
                    incorporated_type_indicator.append(type_indicator[i] + [BT_HISTORY])
                    type_indicator[i].append(ORIGINAL_HISTORY)
        return input_data + incorporated_data, type_indicator + incorporated_type_indicator

    '''Some responses for new personas can be obtained via directly replace some tokens in the original response 
    if the new response is obtained via BERT mask and the masked tokens appeared in both original persona and response'''
    def filter_by_pos(self, bert_mask_personas):
        filtered_personas, filtered_processed_personas, replace_indices = [], [], []
        cnt = 0
        for i in tqdm(range(len(bert_mask_personas))):
            processed_masked_personas = [s for s in self.spacy_model.pipe([x[0] if isinstance(x, list) else x for x in bert_mask_personas[i]], batch_size=1000)]
            processed_original_persona = self.processed_personas[i]
            cur_filtered_processed_personas, cur_replace_indices, cur_filtered_personas  = [], [], []
            for p_i, persona in enumerate(processed_masked_personas):
                if len(processed_original_persona) == len(persona):
                    matched = True
                    different_token_indices = []
                    for j in range(len(persona)):
                        if processed_original_persona[j].text != persona[j].text:
                            different_token_indices.append(j)
                        if processed_original_persona[j].pos_ != persona[j].pos_:
                            matched = False
                            break
                    if matched:
                        cnt += 1
                        cur_filtered_personas.append((persona.text, bert_mask_personas[i][p_i][1]))
                        cur_filtered_processed_personas.append(persona)
                        cur_replace_indices.append(different_token_indices)
            filtered_processed_personas.append(cur_filtered_processed_personas)
            filtered_personas.append(cur_filtered_personas)
            replace_indices.append(cur_replace_indices)
        return filtered_processed_personas, filtered_personas, replace_indices, cnt

    def get_responses_by_replace_tokens(self, processed_new_personas, replace_indices):
        augmented_data_by_token_replace, type_indicator, augmented_data_from_raw_idx = [], [], []
        for i in tqdm(range(len(processed_new_personas))):
            if len(processed_new_personas[i]) > 0:
                original_persona = self.processed_personas[i]
                for j in range(len(processed_new_personas[i])):
                    cur_replace_indices = replace_indices[i][j]
                    new_persona = processed_new_personas[i][j]
                    replace_persona_tokens_and_idx_map = {}
                    for idx in cur_replace_indices:
                        replace_persona_tokens_and_idx_map[original_persona[idx].text] = idx
                    raw_data_idx = self.persona_id_and_data_id[i]
                    responses = [d[2] for d in [self.input_data[idx] for idx in raw_data_idx]]
                    responses = [s for s in self.spacy_model.pipe(responses, batch_size=1000)]
                    for r_i, response in enumerate(responses):
                        response_tokens = [t for t in response]
                        overlap_token_indices = []
                        for response_idx in range(len(response_tokens)):
                            if replace_persona_tokens_and_idx_map.__contains__(response_tokens[response_idx].text):
                                overlap_token_indices.append([response_idx, replace_persona_tokens_and_idx_map[response_tokens[response_idx].text]])
                        if len(overlap_token_indices) > 0:
                            for index in overlap_token_indices:
                                response_tokens[index[0]] = new_persona[index[1]]
                            new_response = ' '.join([t.text for t in response_tokens])
                            augmented_data_by_token_replace.append([new_persona.text, self.input_data[self.persona_id_and_data_id[i][r_i]][1], new_response])
                            type_indicator.append([BERT_MASK_PERSONA])
                            augmented_data_from_raw_idx.append(raw_data_idx[r_i])
        print('Augmented data by token replace number: ' + str(len(augmented_data_by_token_replace)))
        return augmented_data_by_token_replace, augmented_data_from_raw_idx, type_indicator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str)
    parser.add_argument('--output_file', required=True, type=str)
    parser.add_argument('--bert_mask_candidates_num', default=0, type=int)
    parser.add_argument('--bert_mask_ratio', default=0.7, type=float)
    parser.add_argument('--bert_mask_sample_size', default=20, type=float)
    parser.add_argument('--raw_bert_mask_generation_file', default='bert_mask_generation.json', type=str)
    parser.add_argument('--bert_mask_filter_th', default=1.3, type=float)
    parser.add_argument('--token_replace_num', default=0, type=int)
    parser.add_argument('--token_replace_ratio', default=1, type=float)
    parser.add_argument('--token_replace_neighbor_dict', default='persona_train_word_neighbor.bin', type=str)
    parser.add_argument('--gpt2_pred_candidates_num', default=0, type=int)
    parser.add_argument('--gpt2_pred_mask_ratio', default=0.3, type=float)
    parser.add_argument('--gpt2_pred_max_length_fixed', default=15, type=int)
    parser.add_argument('--gpt2_pred_length_extend_ratio', default=0.3, type=float)
    parser.add_argument('--gpt2_pred_filter_th', default=0.775, type=float)
    parser.add_argument('--raw_gpt2_pred_generation_file', default='gpt2_pred_generation.json', type=str)
    parser.add_argument('--generate_new_persona', action='store_true')
    parser.add_argument('--bt_augmented_files', nargs='+', type=str)
    parser.add_argument('--replace_responses_output_file', default=None, type=str)
    args = parser.parse_args()

    generator = NewPersonaGenerator(args)
    if args.generate_new_persona:
        generator.generate()
    else:
        augmented_persona_data, augmented_data_from_raw_idx, type_indicator, processed_bert_mask_personas, \
                bert_mask_personas_replace_indices = generator.get_filtered_data()
        augmented_data_by_token_replace, augmented_data_from_raw_idx_by_token_replace, type_indicator_by_token_replace = \
            generator.get_responses_by_replace_tokens(processed_bert_mask_personas, bert_mask_personas_replace_indices)
        bt_history_augmented_data_without_response, type_indicator = generator.integrate_bt_history(augmented_persona_data,
                augmented_data_from_raw_idx, args.bt_augmented_files, type_indicator)
        bt_history_augmented_data_with_response, type_indicator_with_response = generator.integrate_bt_history(
            augmented_data_by_token_replace, augmented_data_from_raw_idx_by_token_replace, args.bt_augmented_files,
            type_indicator_by_token_replace)
        print('The size of augmented data without response is : ' + str(len(bt_history_augmented_data_without_response)))
        print('The size of augmented data with response by replacement is : ' + str(len(bt_history_augmented_data_with_response)))
        with open(args.output_file + '_without_response.json', 'w', encoding='utf-8') as f:
            json.dump(bt_history_augmented_data_without_response, f)
        with open(args.output_file + '_without_response_type.json', 'w', encoding='utf-8') as f:
            json.dump({'data_type': type_indicator, 'raw_data_idx':
                augmented_data_from_raw_idx + augmented_data_from_raw_idx}, f)
        with open(args.output_file + '_with_replace_response.json', 'w', encoding='utf-8') as f:
            json.dump(bt_history_augmented_data_with_response, f)
        with open(args.output_file + '_with_replace_response_type.json', 'w', encoding='utf-8') as f:
            json.dump({'data_type': type_indicator_with_response, 'raw_data_idx':
                augmented_data_from_raw_idx_by_token_replace + augmented_data_from_raw_idx_by_token_replace}, f)
        with open(args.output_file + '_config.json', 'w', encoding='utf-8') as f:
            json.dump({
                'ppl_weight': PPL_WEIGHT,
                'bert_score_weight': BERT_SCORE_WEIGHT,
                'ppl_weight_for_gpt2': PPL_WEIGHT_FOR_GPT2,
                'bert_score_weight_for_gpt2': BERT_SCORE_WEIGHT_FOR_GPT2,
                'ppl_normalizer': PPL_NORMALIZER,
                'ppl_normalizer_for_gpt2': PPL_NORMALIZER_FOR_GPT2,
                'bert_mask_filter_th': args.bert_mask_filter_th,
                'gpt2_pred_filter_th': args.gpt2_pred_filter_th,
            }, f)
