import os

import json
import torch
import torch.nn as nn

from config import get_trainer_config, InputConfig
from model.dataset import FacebookDataset
from model.trainer import Trainer
from model.gpt2_model import GPT2EncoderDecoderModel, GPT2DoubleHeadsModel
from model.utils import f1_score,  open, set_seed, config_logger
from transformers.tokenization_gpt2 import GPT2Tokenizer
from metrics import nlp_metrics
from model.seq2seq import TransformerSeq2Seq
from model.seq2seq_vocab import Seq2seqVocab
from model.entailment_score import EntailmentScorer
from bert_score.score import get_bert_score

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

def main():
    args = InputConfig().args

    trainer_config = get_trainer_config(args)

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)
    save_path = trainer_config.load_last[:trainer_config.load_last.rfind('/')]
    generate_file_name = args.generate_file_name

    logger = config_logger(os.path.join(save_path, 'inference.log'))

    parsed_valid_data, parsed_test_data = None, None
    if args.model_type == 'gpt2':
        if args.single_input:
            model = GPT2DoubleHeadsModel.from_pretrained('./gpt2-small')
        else:
            model = GPT2EncoderDecoderModel.from_pretrained('./gpt2-small')
        tokenizer = GPT2Tokenizer.from_pretrained('./gpt2-small')
    elif args.model_type == 'seq2seq'  or args.model_type == 'rnn-seq2seq':
        seq2seq_vocab = Seq2seqVocab(trainer_config.train_datasets, trainer_config.valid_datasets,
                                     trainer_config.test_datasets, args.vocab_path, data_type=args.data_type)
        tokenizer = seq2seq_vocab.vocab
        if args.model_type == 'seq2seq':
            model = TransformerSeq2Seq(args.emb_dim, args.hidden_dim, args.num_layers, args.heads, args.depth_size,
                                       args.filter_size, tokenizer, args.pretrained_emb_file, args.pointer_gen, logger,
                                       multi_input=not args.single_input,
                                       attention_pooling_type=args.attention_pooling_type)
        else:
            model = TransformerSeq2Seq(args.emb_dim, args.hidden_dim, args.num_layers, args.heads, args.depth_size,
                                       args.filter_size, tokenizer, args.pretrained_emb_file, args.pointer_gen, logger,
                                       base_model='gru')
        args.dialog_embeddings = False

    model.shared_attention = (args.shared_attention == 1)
    model.shared_module = (args.shared_module == 1)
    model.attention_pooling_type = args.attention_pooling_type
    if args.model_type in ['gpt', 'dialogpt', 'gpt2']:
        tokenizer, additional_length = modify_tokenizer(tokenizer, args.data_type)
        model.embeddings_size = 768
        model.n_embeddings = len(tokenizer)
        model.shared_attention = (args.shared_attention == 1)
        model.shared_module = (args.shared_module == 1)
        model.attention_pooling_type = args.attention_pooling_type
        model.single_input = args.single_input
        if args.model_type == 'gpt':
            model_embedding_weight = model.transformer.tokens_embed.weight
            model.transformer.tokens_embed = nn.Embedding(model.n_embeddings, 768)
            model.lm_head = nn.Linear(768, model.n_embeddings, bias=False)
            model.transformer.tokens_embed.weight.data[:-additional_length, :] = model_embedding_weight.data
            model.transformer.tokens_embed.weight.data[-additional_length:, :] = 0
            model.lm_head.weight = model.transformer.tokens_embed.weight
        else:
            model_embedding_weight = model.transformer.wte.weight
            model.transformer.wte = nn.Embedding(model.n_embeddings, 768)
            model.lm_head = nn.Linear(768, model.n_embeddings, bias=False)
            model.transformer.wte.weight.data[:-additional_length, :] = model_embedding_weight.data
            model.transformer.wte.weight.data[-additional_length:, :] = 0
            model.lm_head.weight = model.transformer.wte.weight

        if not args.single_input:
            model.reload_module_dict()
        model.sent_dialog_id = tokenizer.sent_dialog_id

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
                                   single_input=args.single_input,
                                   data_type=args.data_type,
                                   parsed_data=parsed_test_data)
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
                      evaluate_full_sequences=trainer_config.evaluate_full_sequences,
                      full_input=trainer_config.full_input,
                      uncertainty_loss=args.uncertainty_loss)

    def external_metrics_func(full_references, full_predictions, epoch, metric=None, generate_entail=False):
        if epoch == -1:
            references_file_path = os.path.join(save_path, 'test_references_file.txt')
            predictions_file_path = os.path.join(save_path, 'test_predictions_file.txt')
        else:
            references_file_path = os.path.join(save_path, 'eval_references_file.txt')
            predictions_file_path = os.path.join(save_path, 'eval_predictions_file.txt')
        with open(references_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_references))
        with open(predictions_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_predictions))

        bleu, bleu_list, nist, nist_list, nist_bleu, nist_bleu_list, s_dist, c_dist, entropy, meteor, \
        rouge_l, f1_score, avg_length = nlp_metrics(references_file_path, predictions_file_path)

        metrics = {'meteor': meteor * 100, 'avg_len': avg_length, 'rouge-l': rouge_l * 100, 'bleu': bleu, 'nist': nist,
                   'nist-bleu': nist_bleu, 'f1': f1_score * 100}
        for name, metric in (
        ('bleu', bleu_list), ('nist', nist_list), ('nist_bleu', nist_bleu_list), ('entropy', entropy),
        ('sentence_div', s_dist), ('corpus_div', c_dist)):
            for i, m in enumerate(metric, 1):
                if name == 'sentence_div' or name == 'corpus_div':
                    metrics['{}_{}'.format(name, i)] = m * 100
                else:
                    metrics['{}_{}'.format(name, i)] = m
        if args.entail_score_refs_file and epoch == -1:
            entailment_scorer = EntailmentScorer(predictions_file_path, args.entail_score_refs_file,
                                                 args.entail_model_path, device)
            metrics['entail_score'] = entailment_scorer.calculate_entailment_score()
        if args.bert_score_model_path is not None and epoch == -1:
            all_preds = get_bert_score(
                full_predictions,
                full_references,
                model_type=args.bert_score_model_path,
                num_layers=16,
                batch_size=16,
            )
            metrics['bert_score_p'] = torch.mean(all_preds[0]).item()
            metrics['bert_score_r'] = torch.mean(all_preds[1]).item()
            metrics['bert_score_f'] = torch.mean(all_preds[2]).item()
        for k, v in metrics.items():
            metrics[k] = round(v, 6)
        return metrics

    def external_metrics_func_entail_data(full_predictions, raw_entail_data):
        for i, prediction in enumerate(full_predictions):
            raw_entail_data[i][2] = prediction
        with open(generate_file_name, 'w') as f:
            json.dump(raw_entail_data, f)

    metric_funcs = {'f1_score': f1_score}
    # trainer.test(metric_funcs, external_metrics_func, epoch=0, inference=True)
    if args.data_type == 'entailment':
        with open(args.test_datasets, 'r') as f:
            raw_entail_data = json.load(f)
            trainer.test(metric_funcs, external_metrics_func_entail_data, epoch=-1, inference=True,
                         raw_entail_data=raw_entail_data)
    else:
        trainer.test(metric_funcs, external_metrics_func, epoch=-1, inference=True)

if __name__ == '__main__':
    main()
