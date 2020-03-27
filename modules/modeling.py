import torch
import torch.nn as nn
import copy

from reformer_pytorch import ReformerLM
from collections import OrderedDict


class ReformerQA(nn.Module):
    def __init__(self, config, pretrained_model_path=None):
        super(ReformerQA, self).__init__()
        self.reformer_lm = ReformerLM(
            num_tokens=config['num_tokens'],
            dim=config['dim'],
            depth=config['depth'],
            max_seq_len=config['max_seq_len'],
            heads=config['heads'],
            causal=config['casual'],
            return_embeddings=config['return_embeddings']
        )
        self.qa_outputs = nn.Linear(config['dim'], config['num_label'])
        if pretrained_model_path:
            self._load_weights(pretrained_model_path)

    def _load_weights(self, pretrained_model_path):
        state_dict = copy.deepcopy(torch.load(pretrained_model_path))
        state_dict = OrderedDict(
            (k, v) for k, v in state_dict.items() if 'to_logits' not in k
        )
        self.reformer_lm.load_state_dict(state_dict)

    def forward(self, input_ids=None,
                start_positions=None, end_positions=None):
        sequence_output = self.reformer_lm(input_ids)
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside
            # our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs
