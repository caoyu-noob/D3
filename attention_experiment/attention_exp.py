import os
from tqdm import tqdm

import json
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from config import get_trainer_config, InputConfig
from model.dataset import FacebookDataset
from model.trainer import Trainer
from model.gpt2_model import GPT2DoubleHeadsModel
from transformers.tokenization_gpt2 import GPT2Tokenizer
from model.utils import open, set_seed, config_logger
from model.seq2seq import TransformerSeq2Seq
from model.seq2seq_vocab import Seq2seqVocab

PADDING_IDX = 0

def modify_tokenizer(tokenizer, data_type):
    additional_special_tokens = ['<info_bos>', '<info_eos>', '<talker1_bos>', '<talker1_eos>', '<talker2_bos>',
                                 '<talker2_eos>']
    if data_type == 'emoji':
        with open('datasets/emoji_talk/emojis.json', 'r') as f:
            emojis = json.load(f)['emojis']
        additional_special_tokens.extend(emojis)
    tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>',
                                  'additional_special_tokens': additional_special_tokens})
    tokenizer.eos_id, tokenizer.bos_id, tokenizer.pad_id = tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.pad_token_id
    tokenizer.sent_dialog_id = tokenizer.bos_token_id
    tokenizer.info_dialog_id, tokenizer.info_bos_id = tokenizer.added_tokens_encoder['<info_bos>'], \
                                                      tokenizer.added_tokens_encoder[
                                                          '<info_bos>']
    tokenizer.info_eos_id = tokenizer.added_tokens_encoder['<info_eos>']
    tokenizer.talker1_dialog_id, tokenizer.talker1_bos_id = tokenizer.added_tokens_encoder['<talker1_bos>'], \
                                                            tokenizer.added_tokens_encoder['<talker1_bos>']
    tokenizer.talker1_eos_id = tokenizer.added_tokens_encoder['<talker1_eos>']
    tokenizer.talker2_dialog_id, tokenizer.talker2_bos_id = tokenizer.added_tokens_encoder['<talker2_bos>'], \
                                                            tokenizer.added_tokens_encoder['<talker2_bos>']
    tokenizer.talker2_eos_id = tokenizer.added_tokens_encoder['<talker2_eos>']
    return tokenizer, len(additional_special_tokens) + 3

def pad_sequence(sequences, batch_first=False, padding_value=0, left=False):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    if not len(sequences):
        return torch.empty(0)
    trailing_dims = sequences[0].size()[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        s_slice = slice(-length, None) if left else slice(None, length)
        s_slice = (i, s_slice) if batch_first else (s_slice, i)
        out_tensor[s_slice] = tensor

    return out_tensor

def collate_func(data):
    persona_info, h, y, distractors_batch = zip(*data)

    contexts = []

    if max(map(len, persona_info)) > 0:
        persona_info = [torch.tensor(d, dtype=torch.long) for d in persona_info]
        contexts.append(persona_info)

    if max(map(len, h)) > 0:
        h = [torch.tensor(d, dtype=torch.long) for d in h]
        contexts.append(h)

    y_out = [torch.tensor(d, dtype=torch.long) for d in y]

    distractors = [torch.tensor(d, dtype=torch.long) for distractors in distractors_batch for d in distractors]

    # Pad now so we pad correctly when we have only a single input (context concatenated with y)
    y_out = pad_sequence(y_out, batch_first=True, padding_value=PADDING_IDX)
    distractors = pad_sequence(distractors, batch_first=True, padding_value=PADDING_IDX)
    contexts = [pad_sequence(c, batch_first=True, padding_value=PADDING_IDX) for c in contexts]

    return contexts, y_out, distractors

def _s2s_loss(targets, enc_contexts, model):
    hidden_state, padding_mask = None, None

    nexts = targets[:, 1:].contiguous() if targets.dim() == 2 else targets[:, 1:, 0].contiguous()
    outputs = model.decode(targets[:, :-1].contiguous(), enc_contexts)

    outputs = outputs.view(-1, outputs.shape[-1]).float()
    nexts = nexts.view(-1)

    lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=PADDING_IDX)
    loss = lm_criterion(outputs, nexts)
    return loss, hidden_state, padding_mask

def _lm_loss(contexts, enc_contexts, model, ignore_idxs, device):
    batch_lm_loss = torch.tensor(0, dtype=torch.float, device=device)

    for context in contexts:
        enc_context = model.encode(context.clone())
        enc_contexts.append(enc_context)

        context_outputs = model.generate(enc_context[0])
        ignore_mask = torch.stack([context == idx for idx in ignore_idxs], dim=-1).any(dim=-1)
        context.masked_fill_(ignore_mask, PADDING_IDX)
        prevs = context_outputs[:, :-1, :].contiguous()
        nexts = context[:, 1:].contiguous() if context.dim() == 2 else context[:, 1:, 0].contiguous()
        lm_criterion = torch.nn.CrossEntropyLoss(ignore_index=PADDING_IDX)
        batch_lm_loss += lm_criterion(prevs.view(-1, prevs.shape[-1]).float(), nexts.view(-1)) / len(contexts)
    return batch_lm_loss



def main():
    args = InputConfig().args

    trainer_config = get_trainer_config(args)

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)
    save_path = trainer_config.load_last[:trainer_config.load_last.rfind('/')]
    generate_file_name = args.generate_file_name

    logger = config_logger(os.path.join(save_path, 'inference.log'))

    parsed_valid_data, parsed_test_data = None, None
    if args.model_type == 'seq2seq':
        seq2seq_vocab = Seq2seqVocab(trainer_config.train_datasets, trainer_config.valid_datasets,
                                     trainer_config.test_datasets, args.vocab_path, data_type=args.data_type)
        tokenizer = seq2seq_vocab.vocab
        model = TransformerSeq2Seq(args.emb_dim, args.hidden_dim, args.num_layers, args.heads, args.depth_size,
                                   args.filter_size, tokenizer, args.pretrained_emb_file, args.pointer_gen, logger,
                                   multi_input=not args.single_input,
                                   attention_pooling_type=args.attention_pooling_type)
        args.dialog_embeddings = False
    else:
        model = GPT2DoubleHeadsModel.from_pretrained('./gpt2-small')
        tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-small')
        tokenizer, additional_length = modify_tokenizer(tokenizer, args.data_type)
        model.embeddings_size = 768
        model.n_embeddings = len(tokenizer)
        model.shared_attention = (args.shared_attention == 1)
        model.shared_module = (args.shared_module == 1)
        model.attention_pooling_type = args.attention_pooling_type
        model.single_input = args.single_input
        model_embedding_weight = model.transformer.wte.weight
        model.transformer.wte = nn.Embedding(model.n_embeddings, 768)
        model.lm_head = nn.Linear(768, model.n_embeddings, bias=False)
        model.transformer.wte.weight.data[:-additional_length, :] = model_embedding_weight.data
        model.transformer.wte.weight.data[-additional_length:, :] = 0
        model.lm_head.weight = model.transformer.wte.weight

    model.padding_idx = tokenizer.pad_id
    model.n_pos_embeddings = 512

    model.talker1_id = tokenizer.talker1_bos_id
    model.talker2_id = tokenizer.talker2_bos_id
    model.bos_id = tokenizer.bos_id
    model.eos_id = tokenizer.eos_id
    model.beam_size = args.beam_size
    model.diversity_groups = 1
    model.max_seq_len = 32
    model.dialog_embeddings = args.dialog_embeddings
    model.bs_temperature = args.bs_temperature
    model.bs_nucleus_p = args.bs_nucleus_p
    model.annealing_topk = args.annealing_topk
    model.length_penalty_coef = args.length_penalty
    model.vocab = None
    model.annealing = args.annealing
    model.diversity_coef = args.diversity_coef
    model.sample = False
    model.inference_mode = args.inference_mode
    model.response_k = args.response_k

    logger.info('loading datasets')
    valid_dataset = None
    test_dataset = FacebookDataset(trainer_config.test_datasets, tokenizer,
                                   max_lengths=(model.n_pos_embeddings - 1) // (3 if args.single_input else 1),  # A bit restrictive here
                                   dialog_embeddings=args.dialog_embeddings,
                                   cache=trainer_config.test_datasets_cache,
                                   use_start_end=args.use_start_end,
                                   negative_samples=0,  # Keep all negative samples
                                   augment=False,
                                   aug_syn_proba=0.0,
                                   limit_size=trainer_config.limit_eval_size,
                                   max_history_size=args.max_history_size,
                                   single_input=args.single_input,
                                   data_type=args.data_type,
                                   parsed_data=parsed_test_data)
    # logger.info(f'valid dataset {len(valid_dataset)} test dataset {(len(test_dataset))}')
    logger.info(f'test dataset {(len(test_dataset))}')

    state_dict = torch.load(trainer_config.load_last, map_location=device)
    if state_dict.__contains__('model'):
        model.load_state_dict(state_dict['model'], strict=False)
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    logger.info('Weights loaded from {}'.format(trainer_config.load_last))

    trainer = Trainer(model,
                      test_dataset,
                      trainer_config,
                      None,
                      logger=logger,
                      test_dataset=test_dataset,
                      valid_dataset=valid_dataset,
                      n_jobs=trainer_config.n_jobs,
                      device=device,
                      ignore_idxs=tokenizer.all_special_ids,
                      local_rank=args.local_rank,
                      apex_level=None,
                      apex_loss_scale=trainer_config.apex_loss_scale,
                      full_input=trainer_config.full_input,
                      uncertainty_loss=args.uncertainty_loss)

    _, all_attention, all_attention_inference = trainer.test_attention()
    torch.save(all_attention, 'augmentation/gpt2_th0.99_mix_all_attention.bin')

def analysis_attention():
    all_attention = torch.load('augmentation/gpt2_th0.99_raw_all_attention.bin')
    with open('augmentation/th0.99_gpt2_positions.json', 'r') as f:
        consistent_pos = json.load(f)
    token_pos, target_persona_pos, whole_persona_pos = consistent_pos['token_positions'], \
                                                       consistent_pos['target_persona_positions'], \
                                                       consistent_pos['whole_persona_positions']
    token_level_attentions, sentence_level_attentions = [], []
    persona_sentence_ratio, avg_token_number = [], []
    for i in tqdm(range(len(target_persona_pos))):
        cur_attention = (all_attention[i] / torch.sum(all_attention[i], dim=-1, keepdim=True)).numpy()
        persona_sentence_attention = cur_attention[:, :, target_persona_pos[i][0]: target_persona_pos[i][1]]
        sentence_level_attentions.append(np.mean(np.sum(persona_sentence_attention, axis=-1), axis=-1))
        cur_token_attention = []
        for p in token_pos[i]:
            cur_token_attention.append(cur_attention[:, p[1], p[0]])
        token_level_attentions.append(cur_token_attention)
        persona_sentence_ratio.append((target_persona_pos[i][1] - target_persona_pos[i][0]) / cur_attention.shape[2])
        avg_token_number.append(cur_attention.shape[2])
    token_by_layer, sentence_by_layer = [[] for _ in range(12)], [[] for _ in range(12)]
    for i in range(len(all_attention)):
        for j in range(12):
            sentence_by_layer[j].append(sentence_level_attentions[i][j])
            if len(token_level_attentions[i]) > 0:
                token_by_layer[j].append(np.mean([a[j] for a in token_level_attentions[i]]))
    for i in range(12):
        print(str(i) + ' layer token-level: ' + str(np.mean(token_by_layer[i])))
        print(str(i) + ' layer sentence-level: ' + str(np.mean(sentence_by_layer[i])))
    print('mean token level value: ' + str(1/ np.mean(avg_token_number)))
    print('mean sentence level value: ' + str(np.mean(persona_sentence_ratio)))
        # cur_whole_attention, cur_persona_attention = [], []
        # for p in token_pos[i]:
        #     cur_whole_attention.append(F.softmax(all_attention[i][:, p[1], :], dim=-1).numo'tpy())
        #     cur_persona_attention.append(F.softmax(all_attention[i][:, p[1], :whole_persona_pos[i][1]], dim=-1).numpy())
        # whole_matched_response_attentions.append(cur_whole_attention)
        # within_persona_matched_response_attentions.append(cur_persona_attention)
        # response_persona_attention = F.softmax(all_attention[i], dim=-1).numpy()[:, :, target_persona_pos[i][0]: target_persona_pos[i][1]]
        # sentence_level_attention.append(np.mean(np.sum(response_persona_attention, axis=-1), axis=-1))
    # all_probs, all_probs_within_persona, all_prob_target_persona = [[] for _ in range(6)], [[] for _ in range(6)], \
    #                                                                [[] for _ in range(6)]
    # all_acc, all_acc_within_persona = [0] * 6, [0] * 6
    # lengths, persona_lengths, target_persona_lengths = [], [], []
    # for i in tqdm(range(len(target_persona_pos))):
    #     for j, pos in enumerate(token_pos[i]):
    #         for m in range(6):
    #             all_probs[m].append(whole_matched_response_attentions[i][j][m][pos[0]])
    #             index = np.argsort(-whole_matched_response_attentions[i][j][m])
    #             if index[0] == pos[0]:
    #                 all_acc[m] += 1
    #             all_probs_within_persona[m].append(within_persona_matched_response_attentions[i][j][m][pos[0]])
    #             index = np.argsort(-within_persona_matched_response_attentions[i][j][m])
    #             if index[0] == pos[0]:
    #                 all_acc_within_persona[m] += 1
    #         lengths.append(whole_matched_response_attentions[i][j][0].shape[0])
    #         persona_lengths.append(within_persona_matched_response_attentions[i][j][0].shape[0])
    #     target_persona_lengths.append(target_persona_pos[i][1] - target_persona_pos[i][0])
    #     for m in range(6):
    #         all_prob_target_persona[m].append(sentence_level_attention[i][m])
    # for i in range(6):
    #     print(str(i) + ' layer prob values: ' + str(np.mean(all_probs[i])))
    #     print(str(i) + ' layer acc: ' + str(all_acc[i] / len(all_probs[i])))
    #     print(str(i) + ' layer prob values within persona: ' + str(np.mean(all_probs_within_persona[i])))
    #     print(str(i) + ' layer acc within persona: ' + str(all_acc_within_persona[i] / len(all_probs[i])))
    #     print(str(i) + ' layer sentence-level prob: ' + str(np.mean(all_prob_target_persona[i])))
    # print('mean prob value: ' + str(1 / np.mean(lengths)))
    # print('mean prob value within persona: ' + str(1 / np.mean(persona_lengths)))
    # print('mean prob value sentence-level: ' + str(1 / np.mean(target_persona_lengths)))

def analysis_attention_history():
    all_attention = torch.load('augmentation/gpt2_th0.99_raw_attention.bin')
    with open('augmentation/th0.99_gpt2_positions.json', 'r') as f:
        positions = json.load(f)
    avgs = []
    lengths = []
    attentions = [[] for _ in range(6)]
    for i in range(len(positions)):
        cur_attention = F.softmax(all_attention[i], dim=-1)
        cur_positions = positions[i]
        # if len(cur_positions[1]) >= 3:
        #     start = cur_positions[0][-1]
            # if len(cur_positions[1]) > 3:
            #     start = cur_positions[1][-4]
            # else:
            #     start = cur_positions[0][-1]
        # else:
        #     start = cur_positions[0][-1]
        start = cur_positions[1][-2] if len(cur_positions[1]) > 1 else cur_positions[0][-1]
        end = cur_positions[1][-1]
        target_attention = cur_attention[:, :, start: end]
        avg_attention = 1 / cur_attention.size()[2] * (end - start)
        for j in range(6):
            attentions[j].append(torch.mean(torch.sum(target_attention[j], dim=-1)).item())
        avgs.append(avg_attention)
        lengths.append(cur_attention.size()[2])
    print(np.mean(attentions[0]))
    print(np.mean(attentions[1]))
    print(np.mean(attentions[2]))
    print(np.mean(attentions[3]))
    print(np.mean(attentions[4]))
    print(np.mean(attentions[5]))
    print(np.mean(avgs))
    print(1 / np.mean(lengths))
    print('111')

if __name__ == '__main__':
    # main()
    analysis_attention()
    # analysis_attention_history()
