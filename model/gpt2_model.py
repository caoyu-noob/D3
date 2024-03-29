# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

import datetime
import logging
import math
import os
import random

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from copy import deepcopy
from transformers.activations import gelu_new
from transformers.configuration_gpt2 import GPT2Config
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, prune_conv1d_layer
from .utils import repeat_along_dim1
from .loss import SoftCrossEntropyLoss


logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
    "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin",
    "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-pytorch_model.bin",
    "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-pytorch_model.bin",
}


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, fuse_attention=False):
        super().__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        if not fuse_attention:
            self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        # if fuse_attention:
        #     self.c_proj.weight = nn.Parameter(torch.eye(n_state))
        #     self.c_proj.bias = nn.Parameter(torch.zeros(n_state))
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, bias_mask=True):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        if bias_mask:
            nd, ns = w.size(-2), w.size(-1)
            b = self.bias[:, :, ns - nd : ns, :ns]
            w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask.unsqueeze(1).unsqueeze(2)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    '''When k is None, it is still self-attention for input x, while k is not None, it becomes the mutual attention
    between x and k in which k is used to generate key and value'''
    def forward(self, x, k=None, layer_past=None, attention_mask=None, head_mask=None, previous_query=None,
                bias_mask=True):
        if k is None:
            x = self.c_attn(x)
            query, key, value = x.split(self.split_size, dim=2)
            query = self.split_heads(query)
            key = self.split_heads(key, k=True)
            value = self.split_heads(value)
            if layer_past is not None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
        else:
            proj_weight, proj_bias = self.c_attn.weight, self.c_attn.bias
            if previous_query is not None:
                query = previous_query
            else:
                size_out = x.size()[:-1] + (self.split_size,)
                query = torch.addmm(proj_bias[: self.split_size], x.view(-1, x.size(-1)), proj_weight[:, :self.split_size])
                query = query.view(*size_out)
                query = self.split_heads(query)
            if layer_past is None:
                enc_context = k[0]
                size_out = enc_context.size()[:-1] + (self.split_size * 2,)
                key = torch.addmm(proj_bias[self.split_size:], enc_context.view(-1, enc_context.size(-1)), proj_weight[:, self.split_size:])
                key = key.view(*size_out)
                key, value = key.split(self.split_size, dim=2)
                key = self.split_heads(key, k=True)
                value = self.split_heads(value)
            else:
                key, value = layer_past[0].transpose(-2, -1), layer_past[1]
            padding_mask = k[1]
            attention_mask = padding_mask.float() * float('-1e5')
        saved_query = query
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, bias_mask=(k is None))
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, saved_query, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)

    def fuse_qkv(self, query, key, value, layer_past=None):
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_query, past_key, past_value = layer_past[0], layer_past[1].transpose(-2, -1), layer_past[2]
            query = torch.cat((past_query, query), dim=-2)
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((query, key.transpose(-2, -1), value))
        w = torch.sqrt(torch.matmul(torch.abs(query), torch.abs(key)))
        w = torch.sign(torch.matmul(torch.abs(query), torch.abs(key))) * w / math.sqrt(value.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd: ns, :ns]
        w = w * b - 1e4 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        a = torch.matmul(w, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return [a, present]

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu_new
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False, single_input=False):
        super().__init__()
        nx = config.n_embd
        self.nx = nx
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.output_attentions = config.output_attentions
        if hasattr(config, 'shared_attention'):
            self.shared_attention = config.shared_attention
        else:
            self.shared_attention = True
        if not single_input:
            self.context_size = config.context_size
            if self.context_size > 0:
                self.context_attns = nn.ModuleList([Attention(nx, n_ctx, config, scale) for _ in range(self.context_size)])
            self.dropout = nn.Dropout(config.attn_pdrop)
            self.attention_module = Attention(nx, n_ctx, config, scale, fuse_attention=True)
            self.attention_pooling_type = 'mean'

    def attention_pooling(self, attention_list, layer_past=None):
        if self.attention_pooling_type == "mean":
            return torch.mean(torch.stack(attention_list), dim=0)
        elif self.attention_pooling_type == "max":
            return torch.max(torch.stack(attention_list), dim=0)[0]
        elif self.attention_pooling_type == "min":
            return torch.min(torch.stack(attention_list), dim=0)[0]
        elif self.attention_pooling_type == "sw":
            return torch.mean(torch.stack(attention_list) * self.attention_module.unsqueeze(-1).unsqueeze(-1), dim=0)
        elif self.attention_pooling_type == 'dw':
            return torch.mean(torch.stack(attention_list) * self.attention_module.unsqueeze(1).unsqueeze(1), dim=0)
        elif self.attention_pooling_type == 'linear':
            return self.attention_module(torch.cat(attention_list, dim=-1))
        elif self.attention_pooling_type == 'dys':
            weight = torch.cat([f(torch.mean(attention_list[i], dim=1)) for i, f in enumerate(self.attention_module)], dim=-1)
            weight = nn.Softmax(dim=-1)(weight)
            return torch.mean(torch.stack(attention_list) * weight.transpose(1, 0).unsqueeze(-1).unsqueeze(-1), dim=0)
        elif self.attention_pooling_type == 'dyd':
            weight = torch.cat([f(torch.mean(attention_list[i], dim=1)).unsqueeze(0) for i, f in enumerate(self.attention_module)], dim=0)
            weight = nn.Softmax(dim=0)(weight)
            return torch.mean(torch.stack(attention_list) * weight.unsqueeze(-2), dim=0)
        elif self.attention_pooling_type == 'mdys':
            weight = self.attention_module(torch.mean(torch.cat(attention_list, dim=-1), dim=1))
            weight = nn.Softmax(dim=-1)(weight)
            return torch.mean(torch.stack(attention_list) * weight.transpose(1, 0).unsqueeze(-1).unsqueeze(-1), dim=0)
        elif self.attention_pooling_type == 'mdyd':
            weight = self.attention_module(torch.mean(torch.cat(attention_list, dim=-1), dim=1))
            weight = nn.Softmax(dim=1)(weight.view(-1, 3, self.nx))
            return torch.mean(torch.stack(attention_list) * weight.transpose(1, 0).unsqueeze(-2), dim=0)
        elif self.attention_pooling_type == 'att':
            return self.attention_module.fuse_qkv(attention_list[1], attention_list[2], attention_list[0], layer_past)

    def get_attention_pooling_module(self):
        if self.attention_pooling_type == 'sw':
            self.attention_module = torch.nn.Parameter(torch.ones(3, 1) / 3)
        elif self.attention_pooling_type == 'dw':
            self.attention_module = torch.nn.Parameter(torch.ones(3, self.nx) / 3)
        elif self.attention_pooling_type == 'linear':
            self.attention_module = nn.Linear(self.nx * 3, self.nx)
            # weight = torch.cat([torch.eye(self.nx) for i in range(3)], dim=0)
            # self.attention_module.weight = nn.Parameter(weight.transpose(1, 0))
            # self.attention_module.bias = nn.Parameter(torch.zeros(self.nx))
        elif self.attention_pooling_type == 'dys':
            self.attention_module = nn.ModuleList([nn.Linear(self.nx, 1) for _ in range(3)])
        elif self.attention_pooling_type == 'dyd':
            self.attention_module = nn.ModuleList([nn.Linear(self.nx, self.nx) for _ in range(3)])
        elif self.attention_pooling_type == 'mdys':
            self.attention_module = nn.Linear(self.nx * 3, 3)
        elif self.attention_pooling_type == 'mdyd':
            self.attention_module = nn.Linear(self.nx * 3, self.nx * 3)
        elif self.attention_pooling_type == 'att':
            self.attention_module.c_proj.weight = nn.Parameter(torch.ones(self.nx, self.nx) / self.nx)
            self.attention_module.c_proj.bias = nn.Parameter(torch.zeros(self.nx))
        else:
            del self.attention_module

    def forward(self, x, encoded_context=[], layer_past=None, attention_mask=None, head_mask=None):
        ln_x = self.ln_1(x)
        x_layer_past = layer_past
        if isinstance(layer_past, list):
            x_layer_past = layer_past[0]
        output_attn = self.attn(
            ln_x, layer_past=x_layer_past, attention_mask=attention_mask, head_mask=head_mask
        )
        a = output_attn[0]  # output_attn: a, present, (attentions)
        attentions = []
        if self.output_attentions:
            presents = [output_attn[2]]
            attentions = [output_attn[3]]
        else:
            presents = output_attn[2:]
        if len(encoded_context) != 0:
            '''if encoded persona and history is used as input'''
            context_attention = []
            saved_query = output_attn[1]
            for i, enc in enumerate(encoded_context):
                cur_layer_past = None
                if layer_past is not None:
                    cur_layer_past = layer_past[i + 1]
                if self.shared_attention:
                    enc_output_attn = self.attn(ln_x, k=encoded_context[i], layer_past=cur_layer_past, bias_mask=False,
                                                previous_query=saved_query)
                else:
                    enc_output_attn = self.context_attns[i](ln_x, k=encoded_context[i], layer_past=cur_layer_past,
                                                            bias_mask=False)
                context_attention.append(enc_output_attn[0])
                presents.append(enc_output_attn[2])
                if self.output_attentions:
                    attentions.append(enc_output_attn[3])
            # if hasattr(self, 'attention_weight'):
            #     a = torch.mean(torch.stack([a] + context_attention) * self.attention_weight.unsqueeze(-1).unsqueeze(-1), dim=0)
            # else:
            #     a = torch.mean(torch.stack([a] + context_attention), dim=0)
            pooling_layer_past = None
            if layer_past is not None and len(layer_past) == 4:
                pooling_layer_past = layer_past[3]
            a = self.attention_pooling([a] + context_attention, layer_past=pooling_layer_past)
            if isinstance(a, list):
                presents.append(a[1])
                if layer_past is None:
                    a = a[0]
                else:
                    a = a[0][:, -1:, :]
            a = self.dropout(a)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + [presents] + [attentions]
        return outputs  # x, present, (attentions)


class GPT2PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = GPT2Config
    pretrained_model_archive_map = GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


GPT2_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

GPT2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.GPT2Tokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
            (see `past` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        input_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING,
)
class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config, sinlge_input=False):
        super().__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        # self.output_past = config.output_past

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True, single_input=sinlge_input) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def _mix_up_embedding(self, inputs_embeds, mix_replace):
        for i in range(inputs_embeds.size(0)):
            mask = torch.ones_like(inputs_embeds[i, :, 0], dtype=torch.float)
            new_embed = torch.zeros_like(inputs_embeds[i, :, :], dtype=torch.float)
            for j in range(len(mix_replace[i])):
                mix_neighbors_probs = mix_replace[i][j][1]
                pos = mix_replace[i][j][2]
                neighbor_embed = torch.mean(self.wte(mix_replace[i][j][0]), dim=1)
                weight_sum = torch.sum(mix_neighbors_probs) + 1
                neighbor_embed = torch.sum(neighbor_embed * mix_neighbors_probs.unsqueeze(-1), dim=0) / weight_sum
                mask.index_fill_(0, pos, 1 / weight_sum)
                new_embed[pos] = neighbor_embed.unsqueeze(0)
            inputs_embeds[i, :, :] = inputs_embeds[i, :, :] * mask.unsqueeze(-1) + new_embed
        return inputs_embeds

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        enc_contexts=[],
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mix_replace=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import GPT2Tokenizer, GPT2Model
        import torch

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # new_enc_contexts = []
        # if len(enc_contexts) != 0:
        #     for i, enc in enumerate(enc_contexts):
        #         if isinstance(enc, tuple):
        #             enc = enc[0]
        #         new_enc_contexts.append(enc.view(-1, enc.size()[-2], enc.size()[-1]))

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        # if attention_mask is not None:
        #     attention_mask = attention_mask.view(-1, input_shape[-1])
        #     # We create a 3D attention mask from a 2D tensor mask.
        #     # Sizes are [batch_size, 1, 1, to_seq_length]
        #     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        #     # this attention mask is more simple than the triangular masking of causal attention
        #     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #
        #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        #     # masked positions, this operation will create a tensor which is 0.0 for
        #     # positions we want to attend and -10000.0 for masked positions.
        #     # Since we are adding it to the raw scores before the softmax, this is
        #     # effectively the same as removing these entirely.
        #     attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        if mix_replace is not None:
            start = datetime.datetime.now()
            inputs_embeds = self._mix_up_embedding(inputs_embeds, mix_replace)
            end = datetime.datetime.now()
            # print('mixup: ' + str(end - start))
            inputs_embeds = inputs_embeds.detach()
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states, encoded_context=enc_contexts, layer_past=layer_past, attention_mask=attention_mask,
                head_mask=head_mask[i]
            )

            hidden_states, present = outputs[:2]
            # if self.output_past:
            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        # if self.output_past:
        outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            all_attentions = [torch.mean(a[0], dim=1, keepdim=True) for a in all_attentions]
            all_attentions = torch.cat(all_attentions, dim=1)
            # let the number of heads free (-1) so we can extract attention even after head pruning
            # attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0][0].shape[-2:]
            # all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    GPT2_START_DOCSTRING,
)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        return inputs

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        """
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


@add_start_docstrings(
    """The GPT2 Model transformer with a language modeling and a multiple-choice classification
    head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.
    The language modeling head has its weights tied to the input embeddings,
    the classification head takes as input the input of a specified classification token index in the input sequence).
""",
    GPT2_START_DOCSTRING,
)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config, sinlge_input=True)
        self.transformer.output_attentions = True
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        distractor=False,
        mix_replace=None,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`)
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`, defaults to :obj:`None`)
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``lm_labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`multiple_choice_labels` is provided):
            Multiple choice classification loss.
        lm_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

        """
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            mix_replace=mix_replace
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if mix_replace is not None and self.mixup_soft_loss_weight > 0:
                soft_loss_fct = SoftCrossEntropyLoss()
                soft_labels = torch.zeros_like(shift_logits)
                lengths = []
                for i in range(len(mix_replace)):
                    for replace_item in mix_replace[i]:
                        replace_ids = replace_item[0]
                        replace_probs = replace_item[1] / (torch.sum(replace_item[1]) + 1)
                        for replace_i in range(replace_ids.size(0)):
                            soft_labels[i, replace_item[2][0] - 1:replace_item[2][-1] , replace_ids[replace_i]] = \
                                replace_probs[replace_i]
                    lengths.append(len(mix_replace[i]))
                mixup_soft_loss = soft_loss_fct(shift_logits, soft_labels,
                                                torch.tensor(lengths, dtype=torch.float, device=soft_labels.device))
                loss = loss + self.mixup_soft_loss_weight * mixup_soft_loss
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)

    def inference(self, input_ids, token_type_ids, return_beams=False):
        if self.inference_mode == 'beam':
            return self.beam_search(input_ids, token_type_ids, return_beams)
        elif self.inference_mode == 'sampling':
            return self.sampling_inference(input_ids, token_type_ids)

    def sampling_inference(self, input_ids, token_type_ids):
        with torch.no_grad():
            batch_size = 1
            device = next(self.parameters()).device
            scores = torch.zeros(self.response_k, device=device)
            predicts = []
            for k in range(self.response_k):
                prevs = torch.full((batch_size, 1), fill_value=self.talker1_id, dtype=torch.long, device=device)
                sample_scores, lens = 0, 1
                for i in range(self.max_seq_len):
                    if i == 0:
                        cur_input_ids = input_ids.unsqueeze(0)
                        cur_token_type_ids = token_type_ids.unsqueeze(0)
                        lm_logits, _, past = self.forward(cur_input_ids, token_type_ids=cur_token_type_ids)
                        lm_logits = lm_logits[:, -1:, :]
                    else:
                        inputs = prevs[:, -1:, ...]  # only use the last token (rest is in past)
                        cur_token_type_ids = torch.full_like(inputs, self.talker1_id)
                        lm_logits, _, past = self.forward(inputs, token_type_ids=cur_token_type_ids, past=past)
                    probs = self._get_proba_with_temperature(lm_logits.float()).squeeze(1)
                    cur_idxs = torch.multinomial(probs, 1)
                    prevs = torch.cat([prevs, cur_idxs], 1)
                    lens += 1
                    cur_scores = torch.gather(probs, 1, cur_idxs)
                    sample_scores += torch.log(cur_scores)
                    if cur_idxs == self.eos_id:
                        lens -= 1
                        break
                sample_scores /= self._length_penalty(float(lens))
                scores[k] = sample_scores.squeeze(1)
                predicts.append(prevs[0, 1: lens].tolist())
            best_idx = scores.argmax(dim=0)
            return predicts[best_idx]

    def beam_search(self, input_ids, token_type_ids, return_beams=False):
        with torch.no_grad():
            batch_size = 1
            device = next(self.parameters()).device

            prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.talker1_id, dtype=torch.long,
                               device=device)

            beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.bool, device=device)

            current_sample_prob = 1
            group_size = self.beam_size // self.diversity_groups
            diversity_penalty = torch.zeros((batch_size, self.n_embeddings), device=device)
            past = None

            for i in range(self.max_seq_len):
                if i == 0:
                    cur_input_ids = input_ids.unsqueeze(0)
                    cur_token_type_ids = token_type_ids.unsqueeze(0)
                    cur_input_ids = torch.cat([cur_input_ids] * self.beam_size, dim=0)
                    cur_token_type_ids = torch.cat([cur_token_type_ids] * self.beam_size, dim=0)
                    lm_logits, _, past = self.forward(cur_input_ids, token_type_ids=cur_token_type_ids)
                    lm_logits = lm_logits[:, -1:, :]
                else:
                    inputs = prevs[:, -1:, ...]  # only use the last token (rest is in past)
                    token_type_ids = torch.full_like(inputs, self.talker1_id)
                    lm_logits, _, past = self.forward(inputs, token_type_ids=token_type_ids, past=past)
                probs = self._get_proba_with_temperature(lm_logits.float())
                probs = probs.view(batch_size, self.beam_size, -1)

                beam_scores = self._get_beam_scores(probs, beam_scores, is_end)
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float()).unsqueeze(-1)
                beam_scores = beam_scores / penalty

                if i == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, self.beam_size), dtype=torch.long, device=device)
                else:
                    penalty = penalty.view(batch_size, self.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        g_scores, g_idxs = self._sample(g_beam_scores, group_size, sample_prob=current_sample_prob)
                        g_idxs += g * group_size * self.n_embeddings

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1,
                                                       torch.fmod(g_idxs, self.n_embeddings),
                                                       torch.ones((batch_size, group_size), device=device))

                    diversity_penalty.fill_(0)
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1)

                    beam_idxs = (idxs.float() / self.n_embeddings).long()

                sym_idxs = torch.fmod(idxs, probs.shape[-1])
                is_end = torch.gather(is_end, 1, beam_idxs)
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                if self.vocab is not None:
                    logger.info(
                        '\nbeams:\n' + '\n'.join(self.vocab.ids2string(t.detach().cpu().tolist()) for t in prevs))
                    logger.info('\ntop-options:\n' + '\n'.join(self.vocab.ids2string(t.detach().cpu().tolist())
                                                               + str(bi.detach().cpu().tolist()) for t, bi in
                                                               zip(sym_idxs, beam_idxs)))
                # torch.save([sym_idxs, is_end], 'error.bin')
                # print('error saved!')
                # print(sym_idxs)
                # print(is_end)
                try:
                    sym_idxs[is_end] = self.padding_idx
                except:
                    print(is_end)
                    print(sym_idxs)
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.eos_id] = 1

                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
                prevs = prevs.view(batch_size, self.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                past = self._fix_past(past, beam_idxs)

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.annealing

            result = prevs.view(batch_size, self.beam_size, -1)

            if return_beams:
                return result, beam_lens

            if self.sample:
                probs = torch.nn.functional.softmax(beam_scores, dim=-1)
                bests = torch.multinomial(probs, 1).view(-1)
            else:
                bests = beam_scores.argmax(dim=-1)

            best_len = beam_lens[0, bests[0]]
            best_seq = result[0, bests[0], :best_len - 1]
            predicts = best_seq.tolist()

        return predicts

    def _get_proba_with_temperature(self, logits):
        if self.bs_temperature != 1:
            logits /= self.bs_temperature

        return torch.nn.functional.softmax(logits, dim=-1)

    def _get_beam_scores(self, probas, beam_scores, is_end):
        skip_mask = None

        if self.bs_nucleus_p > 0:
            assert self.annealing_topk is None

            sorted_probas, idxs = torch.sort(probas, descending=True, dim=-1)
            skip_mask = torch.cumsum(sorted_probas.cumsum(dim=-1) > self.bs_nucleus_p, dim=-1) > 1
            sorted_probas.masked_fill_(skip_mask, 0.0)
            _, idxs = torch.sort(idxs, dim=-1)
            probas = torch.gather(sorted_probas, -1, idxs)
            skip_mask = torch.gather(skip_mask, -1, idxs)
        beam_scores = beam_scores.unsqueeze(-1) + torch.log(probas + 1e-20) * (1 - is_end.float().unsqueeze(-1))

        if skip_mask is not None:
            beam_scores.masked_fill_(skip_mask, float('-inf'))

        return beam_scores

    def _sample(self, beam_scores, num_samples, sample_prob=1.):
        if random.random() < sample_prob:
            beam_probas = torch.nn.functional.softmax(beam_scores, dim=-1)
            if self.annealing_topk is not None:
                beam_probas, sample_idxs = beam_probas.topk(self.annealing_topk, dim=-1)
                idxs = torch.multinomial(beam_probas, num_samples)
                idxs = torch.gather(sample_idxs, 1, idxs)
            else:
                idxs = torch.multinomial(beam_probas, num_samples)
            scores = torch.gather(beam_scores, 1, idxs)
        else:
            scores, idxs = beam_scores.topk(num_samples, dim=-1)

        return scores, idxs

    def _fix_past(self, past, beam_idxs):
        for layer_output in past:
            for context in layer_output:
                for v in context:
                    size_ = v.size()
                    tile_size = size_[-2] * size_[-1] * size_[-3]
                    new_v = v.contiguous().view(-1, self.beam_size, tile_size)
                    new_v = new_v.gather(1, beam_idxs.unsqueeze(-1).repeat([1, 1, tile_size]))
                    v[...] = new_v.view(*size_)
        return past

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.length_penalty_coef / (5 + 1) ** self.length_penalty_coef

class GPT2EncoderDecoderModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.encoder = self.transformer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.shared_module = config.shared_module
        self.shared_attention = config.shared_attention
        self.context_size = config.context_size

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def reload_module_dict(self):
        if not self.shared_module:
            self.encoder = deepcopy(self.transformer)
            for block in self.encoder.h:
                if hasattr(block, 'context_attns'):
                    del block.context_attns
                if hasattr(block, 'attention_module'):
                    del block.attention_module
        else:
            if hasattr(self, 'encoder'):
                del self.encoder
        for block in self.transformer.h:
            if not self.shared_attention:
                for context_attn in block.context_attns:
                    context_attn.load_state_dict(block.attn.state_dict())
            else:
                if hasattr(block, 'context_attns'):
                    del block.context_attns
                block.shared_attention = True
            block.attention_pooling_type = self.attention_pooling_type
            block.get_attention_pooling_module()

    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        persona_ids=None,
        history_ids=None,
        past=None,
        persona_past=None,
        history_past=None,
        attention_mask=None,
        token_type_ids=None,
        persona_token_type_ids=None,
        history_token_type_ids=None,
        enc_persona_state=None,
        enc_history_state=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`)
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`, defaults to :obj:`None`)
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``lm_labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`multiple_choice_labels` is provided):
            Multiple choice classification loss.
        lm_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

        # Add a [CLS] to the vocabulary (we should train it also!)
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size
        print(tokenizer.cls_token_id, len(tokenizer))  # The newly token the last token of the vocabulary

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        encoded_choices = [tokenizer.encode(s) for s in choices]
        cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

        input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
        mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

        """
        lm_persona_loss, lm_history_loss = None, None
        contexts = []
        if persona_ids is not None and enc_persona_state is None:
            encoded_persona_outputs = self.encoder(persona_ids, past=persona_past, attention_mask=attention_mask,
                    token_type_ids=persona_token_type_ids, position_ids=position_ids, head_mask=head_mask,
                    inputs_embeds=inputs_embeds,)
            contexts.append(encoded_persona_outputs[0])
            enc_persona_state = encoded_persona_outputs[0]
            if lm_labels is not None:
                loss_fct = CrossEntropyLoss()
                lm_persona_logits = self.lm_head(encoded_persona_outputs[0])
                shift_persona_logits = lm_persona_logits[..., :-1, :].contiguous()
                shift_persona_labels = persona_ids[..., 1:].contiguous()
                lm_persona_loss = loss_fct(shift_persona_logits.view(-1, shift_persona_logits.size(-1)),
                                           shift_persona_labels.view(-1))
        elif enc_persona_state is not None:
            contexts.append(enc_persona_state)

        if history_ids is not None and enc_history_state is None:
            encoded_history_outputs = self.encoder(history_ids, past=history_past, attention_mask=attention_mask,
                    token_type_ids=history_token_type_ids, position_ids=position_ids, head_mask=head_mask,
                    inputs_embeds=inputs_embeds,)
            contexts.append(encoded_history_outputs[0])
            enc_history_state = encoded_history_outputs[0]
            if lm_labels is not None:
                loss_fct = CrossEntropyLoss()
                lm_history_logits = self.lm_head(encoded_history_outputs[0])
                shift_history_logits = lm_history_logits[..., :-1, :].contiguous()
                shift_history_labels = history_ids[..., 1:].contiguous()
                lm_history_loss = loss_fct(shift_history_logits.view(-1, shift_history_logits.size(-1)),
                                           shift_history_labels.view(-1))
        elif enc_history_state is not None:
            contexts.append(enc_history_state)

        transformer_outputs = self.transformer(
            input_ids,
            encoded_context=contexts,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:] + (enc_persona_state, enc_history_state)
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss, lm_persona_loss, lm_history_loss) + outputs

        return outputs  # (lm loss),

    def encode(self, x):
        input_ids = x[:, :, 0].squeeze(-1)
        token_type_ids = x[:, :, 1].squeeze(-1)
        attention_mask = torch.zeros_like(input_ids).float().masked_fill_(input_ids.eq(self.padding_idx), float('-1e5'))
        x, _ = self.transformer(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask) if self.shared_module else \
                self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        padding_mask = input_ids.eq(self.padding_idx)
        return x, padding_mask

    def generate(self, enc_x):
        return self.lm_head(enc_x)

    def decode(self, x, enc_contexts=[]):
        input_ids = x[:, :, 0].squeeze(-1)
        token_type_ids = x[:, :, 1].squeeze(-1)
        x, *_ = self.transformer(input_ids, token_type_ids=token_type_ids, enc_contexts=enc_contexts)
        padding_mask = input_ids.eq(self.padding_idx)
        return self.generate(x), x, padding_mask

    def classify(self, x, padding_mask):
        cls_index = padding_mask.size(-1) - 1 - torch.sum(padding_mask, -1)
        return self.multiple_choice_head(x, cls_index)

    def decode_classify(self, x, enc_contexts=[]):
        input_ids = x[:, :, 0].squeeze(-1)
        token_type_ids = x[:, :, 1].squeeze(-1)
        x, _ = self.transformer(input_ids, token_type_ids=token_type_ids, enc_contexts=enc_contexts)
        cls_index = input_ids.size(-1) - 1 - torch.sum(input_ids.eq(self.padding_idx), -1)
        return self.multiple_choice_head(x, cls_index)

    def _get_proba_with_temperature(self, logits):
        if self.bs_temperature != 1:
            logits /= self.bs_temperature

        return torch.nn.functional.softmax(logits, dim=-1)

    def _get_beam_scores(self, probas, beam_scores, is_end):
        skip_mask = None

        if self.bs_nucleus_p > 0:
            assert self.annealing_topk is None

            sorted_probas, idxs = torch.sort(probas, descending=True, dim=-1)
            skip_mask = torch.cumsum(sorted_probas.cumsum(dim=-1) > self.bs_nucleus_p, dim=-1) > 1
            sorted_probas.masked_fill_(skip_mask, 0.0)
            _, idxs = torch.sort(idxs, dim=-1)
            probas = torch.gather(sorted_probas, -1, idxs)
            skip_mask = torch.gather(skip_mask, -1, idxs)

        beam_scores = beam_scores.unsqueeze(-1) + torch.log(probas) * (1 - is_end.float().unsqueeze(-1))

        if skip_mask is not None:
            beam_scores.masked_fill_(skip_mask, float('-inf'))

        return beam_scores

    def _sample(self, beam_scores, num_samples, sample_prob=1.):
        if random.random() < sample_prob:
            beam_probas = torch.nn.functional.softmax(beam_scores, dim=-1)
            if self.annealing_topk is not None:
                beam_probas, sample_idxs = beam_probas.topk(self.annealing_topk, dim=-1)
                idxs = torch.multinomial(beam_probas, num_samples)
                idxs = torch.gather(sample_idxs, 1, idxs)
            else:
                idxs = torch.multinomial(beam_probas, num_samples)
            scores = torch.gather(beam_scores, 1, idxs)
        else:
            scores, idxs = beam_scores.topk(num_samples, dim=-1)

        return scores, idxs

    def _fix_past(self, past, beam_idxs):
        for layer_output in past:
            for context in layer_output:
                for v in context:
                    size_ = v.size()
                    tile_size = size_[-2] * size_[-1] * size_[-3]
                    new_v = v.contiguous().view(-1, self.beam_size, tile_size)
                    new_v = new_v.gather(1, beam_idxs.unsqueeze(-1).repeat([1, 1, tile_size]))
                    v[...] = new_v.view(*size_)
        return past

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.length_penalty_coef / (5 + 1) ** self.length_penalty_coef

    def inference(self, enc_contexts=[], return_beams=False, beam_starts=None):
        if self.inference_mode == 'beam':
            return self.beam_search(enc_contexts, return_beams, beam_starts)
        elif self.inference_mode == 'sampling':
            return self.sampling_inference(enc_contexts, beam_starts)

    def sampling_inference(self, enc_contexts, beam_starts):
        with torch.no_grad():
            if len(enc_contexts) == 0 and beam_starts is None:
                return []
            predicts = []
            batch_size = enc_contexts[0][0].shape[0] if beam_starts is None else beam_starts.shape[0]
            device = next(self.parameters()).device
            scores = torch.zeros(batch_size, self.response_k, device=device)
            for k in range(self.response_k):
                prevs = torch.full((batch_size, 1), fill_value=self.bos_id, dtype=torch.long, device=device)
                sample_scores = torch.zeros(batch_size, 1, device=device)
                lens = torch.ones(batch_size, 1, dtype=torch.long, device=device)
                is_end = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
                max_seq_len = min(
                    self.n_pos_embeddings - prevs.shape[1] - (beam_starts.shape[1] if beam_starts is not None else 0),
                    self.max_seq_len)
                past = None
                for i in range(max_seq_len):
                    inputs = prevs[:, -1:, ...]  # only use the last token (rest is in past)
                    token_type_ids = torch.full_like(inputs, self.sent_dialog_id)
                    outputs, past = self.transformer(inputs, token_type_ids=token_type_ids,
                                                     enc_contexts=enc_contexts, past=past)
                    logits = self.generate(outputs[:, -1, :])
                    probs = self._get_proba_with_temperature(logits.float())
                    cur_idxs = torch.multinomial(probs, 1)
                    prevs = torch.cat([prevs, cur_idxs], 1)
                    is_end[cur_idxs == self.eos_id] = 1
                    lens[~is_end] += 1
                    cur_scores = torch.gather(probs, 1, cur_idxs)
                    sample_scores += torch.log(cur_scores)
                sample_scores /= self._length_penalty(lens.float())
                scores[:, k] = sample_scores.squeeze(1)
                cur_predict = []
                for i in range(batch_size):
                    length = lens[i]
                    cur_predict.append(prevs[i, 1: length].tolist())
                predicts.append(cur_predict)
            best_idx = scores.argmax(dim=1)
            final_predicts = []
            for i in range(batch_size):
                final_predicts.append(predicts[best_idx[i]][i])
            return final_predicts

    def beam_search(self, enc_contexts=[], return_beams=False, beam_starts=None):
        with torch.no_grad():
            if len(enc_contexts) == 0 and beam_starts is None:
                return []

            batch_size = enc_contexts[0][0].shape[0] if beam_starts is None else beam_starts.shape[0]
            device = next(self.parameters()).device

            prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.bos_id, dtype=torch.long,
                               device=device)

            beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.bool, device=device)

            if beam_starts is not None:
                beam_starts = repeat_along_dim1(beam_starts, self.beam_size)
            beam_enc_contexts = repeat_along_dim1(enc_contexts, self.beam_size)

            current_sample_prob = 1
            group_size = self.beam_size // self.diversity_groups
            diversity_penalty = torch.zeros((batch_size, self.n_embeddings), device=device)
            past = None

            max_seq_len = min(
                self.n_pos_embeddings - prevs.shape[1] - (beam_starts.shape[1] if beam_starts is not None else 0),
                self.max_seq_len)

            for i in range(max_seq_len):
                inputs = prevs[:, -1:, ...]  # only use the last token (rest is in past)
                token_type_ids = torch.full_like(inputs, self.sent_dialog_id)
                # if self.dialog_embeddings and inputs.dim() < 3:
                #     inputs = torch.stack((inputs, torch.full_like(inputs, self.sent_dialog_id)), dim=inputs.dim())
                # if i == 0 and beam_starts is not None:
                #     inputs = torch.cat((beam_starts, inputs), dim=1)

                outputs, past = self.transformer(inputs, token_type_ids=token_type_ids, enc_contexts=beam_enc_contexts, past=past)

                logits = self.generate(outputs[:, -1, :])

                probs = self._get_proba_with_temperature(logits.float())
                probs = probs.view(batch_size, self.beam_size, -1)

                beam_scores = self._get_beam_scores(probs, beam_scores, is_end)
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float()).unsqueeze(-1)
                beam_scores = beam_scores / penalty

                if i == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, self.beam_size), dtype=torch.long, device=device)
                else:
                    penalty = penalty.view(batch_size, self.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        g_scores, g_idxs = self._sample(g_beam_scores, group_size, sample_prob=current_sample_prob)
                        g_idxs += g * group_size * self.n_embeddings

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1,
                                                       torch.fmod(g_idxs, self.n_embeddings),
                                                       torch.ones((batch_size, group_size), device=device))

                    diversity_penalty.fill_(0)
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1)

                    beam_idxs = (idxs.float() / self.n_embeddings).long()

                sym_idxs = torch.fmod(idxs, probs.shape[-1])
                is_end = torch.gather(is_end, 1, beam_idxs)
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                if self.vocab is not None:
                    logger.info(
                        '\nbeams:\n' + '\n'.join(self.vocab.ids2string(t.detach().cpu().tolist()) for t in prevs))
                    logger.info('\ntop-options:\n' + '\n'.join(self.vocab.ids2string(t.detach().cpu().tolist())
                                                               + str(bi.detach().cpu().tolist()) for t, bi in
                                                               zip(sym_idxs, beam_idxs)))

                sym_idxs[is_end] = self.padding_idx
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.eos_id] = 1

                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
                prevs = prevs.view(batch_size, self.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                past = self._fix_past(past, beam_idxs)

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.annealing

            predicts = []
            result = prevs.view(batch_size, self.beam_size, -1)

            if return_beams:
                return result, beam_lens

            if self.sample:
                probs = torch.nn.functional.softmax(beam_scores, dim=-1)
                bests = torch.multinomial(probs, 1).view(-1)
            else:
                bests = beam_scores.argmax(dim=-1)

            for i in range(batch_size):
                best_len = beam_lens[i, bests[i]]
                best_seq = result[i, bests[i], 1:best_len - 1]
                predicts.append(best_seq.tolist())

        return predicts

class GPT2PrototypeModel(GPT2EncoderDecoderModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, label_ids, insert_ids, delete_ids):
        return
