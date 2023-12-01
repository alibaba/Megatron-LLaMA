# Copyright (c) 2023, ALIBABA CORPORATION.  All rights reserved.


"""Baichuan model."""

import torch

from megatron import get_args
from megatron.core import tensor_parallel
from .module import MegatronModule

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
        
        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


class BaichuanModel(MegatronModule):
    """Baichuan Language model."""

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True):
        args = get_args()
        super(BaichuanModel, self).__init__(share_word_embeddings=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.sequence_parallel = args.sequence_parallel
        self.padded_vocab_size = args.padded_vocab_size

        self.language_model, self._language_model_key = get_language_model(
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers),
            pre_process=self.pre_process,
            post_process=self.post_process)
        
        self.causal_lm = args.causal_lm

        if not args.untie_embeddings_and_output_weights and not self.causal_lm:
            self.initialize_word_embeddings(init_method_normal)
        
        if self.causal_lm and self.post_process:
            self.lm_head = torch.nn.Linear(args.hidden_size, args.padded_vocab_size, bias=False)

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def _causal_lm_process(self, lm_output, labels):
        if self.sequence_parallel:
            lm_output = tensor_parallel.gather_from_sequence_parallel_region(lm_output, False)
        lm_output = lm_output.transpose(0, 1)
        logits = self.lm_head(lm_output)

        if labels is None:
            return logits
        else:
            loss = None
            # [invalid] Shift so that tokens < n predict n
            # Do not need to shift here
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., :-1].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            shift_logits = shift_logits.view(-1, self.padded_vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            return loss

    def forward(self, input_ids, position_ids, attention_mask,
                ret_input_ids=None, ret_position_ids=None, ret_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None):
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            ret_input_ids=ret_input_ids,
            ret_position_ids=ret_position_ids,
            ret_attn_mask=ret_attn_mask,
            inference_params=inference_params)

        if self.post_process:
            if self.causal_lm:
                return self._causal_lm_process(lm_output=lm_output, labels=labels)
            else:
                return post_language_model_processing(
                    lm_output, labels,
                    self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.word_embeddings_weight(),
                    self.parallel_output,
                    self.fp16_lm_cross_entropy)
        else:
            return lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if (self.post_process
            and not self.pre_process
            and not self.untie_embeddings_and_output_weights
            and not self.causal_lm):
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        if self.post_process and self.causal_lm:
            state_dict_['lm_head'] = self.lm_head.state_dict()

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        if self.causal_lm and self.post_process:
            self.lm_head.load_state_dict(state_dict['lm_head'], strict=strict)

        # Load word_embeddings.
        if self.post_process and \
            not self.pre_process \
            and not self.untie_embeddings_and_output_weights \
            and not self.causal_lm:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
