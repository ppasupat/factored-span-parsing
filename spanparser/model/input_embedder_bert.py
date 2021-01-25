from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
)
from transformers import AutoModel

from spanparser.utils import try_gpu


class InputEmbedderBert(nn.Module):
    """
    Embeds spans by concatenating the specified span features.
    """

    def __init__(self, config, meta):
        super().__init__()
        c_emb = config.model.input_embedder

        self.CLS_index = meta.vocab_x['[CLS]']
        self.SEP_index = meta.vocab_x['[SEP]']

        self.bert = AutoModel.from_pretrained(c_emb.model)
        self.seq_in_size = seq_in_size = self.bert.config.hidden_size
        self.span_embedding_dim = seq_in_size * 3

        print(
            "Dimensions:",
            f"seq_in_size = {seq_in_size};",
            f"span_embed_dim = {self.span_embedding_dim}",
        )

    def forward(self, batch):
        seq_lengths = try_gpu(torch.tensor([x.length for x in batch]))
        max_seq_len = max(x.length for x in batch)

        # bert_out: (batch_size, max_seq_len + 2, seq_in_size)
        bert_out = self.embed_stuff(batch, seq_lengths)

        # Construct a list of all possible spans
        spans = self.all_spans(max_seq_len)

        # Batch head-word reprsentation
        batch_representations = self.get_batch_representation(
            bert_out, spans, seq_lengths,
        )

        return batch_representations, spans, seq_lengths

    def embed_stuff(self, batch, seq_lengths):
        """
        Build tensors from the token indices

        Args:
            batch: List[Example]
            seq_lengths: (batch_size,)
        Returns:
            bert_out: (batch_size, max_seq_len, seq_in_size)
        """
        batch_token_indices = []
        for example in batch:
            token_indices = example.token_indices
            # Pad with [CLS] and [SEP]
            token_indices = [self.CLS_index] + token_indices + [self.SEP_index]
            # Add to list
            token_indices = torch.tensor(token_indices)
            batch_token_indices.append(token_indices)
        # input_ids: (batch_size, max_seq_len + 2)
        input_ids = pad_sequence(batch_token_indices, batch_first=True)
        input_ids = try_gpu(input_ids)
        # attention_mask and token_type_ids
        positions = try_gpu(torch.arange(input_ids.size()[1])[None, :])
        attention_mask = 1 * (positions < (seq_lengths + 2)[:, None])
        token_type_ids = try_gpu(torch.zeros(input_ids.size(), dtype=int))
        # Encode!
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return bert_output.last_hidden_state

    def all_spans(self, seq_len: int) -> List[List[int]]:
        indices = []
        for span_size in range(seq_len):
            for span in range(seq_len - span_size):
                indices.append([span, span + span_size + 1])
        return indices

    def get_batch_representation(
        self,
        bert_out: torch.Tensor,
        spans: List[List[int]],
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            bert_out: (batch_size, max_seq_len + 2, seq_in_size)
            spans: (num_spans, 2)
            seq_lengths: (batch_size,)
        Returns:
            representation: (batch_size, num_spans, ???)
        """
        batch_size, padded_max_seq_len, _ = bert_out.size()
        max_seq_len = padded_max_seq_len - 2
        num_spans = len(spans)

        # Create python lists for span-features
        index_matrix = [None] * num_spans
        for i, span in enumerate(spans):
            index_matrix[i] = torch.zeros(padded_max_seq_len).index_put(
                [torch.arange(span[0] + 1, span[1] + 1)], torch.tensor([1.0])
            )

        start_indices = try_gpu(torch.tensor([x[0] for x in spans]))
        end_indices = try_gpu(torch.tensor([x[1] for x in spans]))
        # (batch_size, num_spans, max_seq_len + 2)
        batch_index_matrix = try_gpu(
            torch.stack(index_matrix).expand(batch_size, -1, -1)
        )

        # List[(batch_size, num_span, ???)]
        things_to_concat: List[torch.Tensor] = []

        # Index select the start and end lstm outputs.
        # (batch_size, num_spans, embed_dim)
        start_hidden = bert_out[:, start_indices + 1, :]
        # (batch_size, num_spans, seq_in_size)
        end_hidden = bert_out[:, end_indices, :]
        things_to_concat += [start_hidden, end_hidden]

        average_weights = batch_index_matrix / torch.sum(
            batch_index_matrix, dim=2, keepdim=True
        )
        # (batch_size, num_spans, seq_in_size)
        average_hidden = torch.bmm(average_weights, bert_out)
        things_to_concat.append(average_hidden)

        # Concatenate all portions of the representations.
        representation = torch.cat(things_to_concat, dim=2)
        return representation

