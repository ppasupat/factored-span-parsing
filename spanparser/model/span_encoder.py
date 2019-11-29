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

from spanparser.utils import try_gpu, PAD, UNK, SOS, EOS


class SpanFeature(object):
    INSIDE_TOKEN = "inside_token"
    INSIDE_HIDDEN = "inside_hidden"
    STERN_HIDDEN = "stern_hidden"
    AVERAGE_TOKEN = "average_token"
    AVERAGE_HIDDEN = "average_hidden"
    ATTENTION_TOKEN = "attention_token"
    ATTENTION_HIDDEN = "attention_hidden"
    LENGTH = "length"


class SpanEncoder(nn.Module):
    """
    Embeds spans by concatenating the specified span features.
    """

    def __init__(self, config, meta):
        super().__init__()
        c_enc = config.model.span_encoder
        self.word_dropout = c_enc.word_dropout
        if self.word_dropout:
            self._build_word_dropout_probs(meta)

        self.UNK_index = meta.vocab_x[UNK]
        self.PAD_index = meta.vocab_x[PAD]
        self.SOS_index = meta.vocab_x[SOS]
        self.EOS_index = meta.vocab_x[EOS]

        self.token_embedder = nn.Embedding(
            len(meta.vocab),
            c_enc.embedding.dim,
        )

        self.lstm = nn.LSTM(
            c_enc.embedding.dim,
            c_enc.lstm.dim,
            batch_first=True,
            num_layers=c_enc.lstm.layers,
            bidirectional=True,
            dropout=config.model.dropout,
        )

        self.embed_dim = embed_dim = c_enc.embedding.dim
        self.lstm_dim = c_enc.lstm.dim
        self.seq_in_size = seq_in_size = c_enc.lstm.dim * 2
        self._build_length_buckets()

        self.span_features = set(c_enc.span_features)
        print("Span features:", self.span_features)
        self.attention_ffnn = nn.Linear(seq_in_size, 1)

        # We bin the span length, and then have an embedding for each bin.
        if SpanFeature.LENGTH in self.span_features:
            self.length_embeddings = nn.Embedding(
                self.num_buckets, c_enc.length_embed_dim
            )

        # Span embedding dim
        self.span_embedding_dim = 0
        if SpanFeature.INSIDE_TOKEN in self.span_features:
            self.span_embedding_dim += embed_dim * 2
        if SpanFeature.INSIDE_HIDDEN in self.span_features:
            self.span_embedding_dim += seq_in_size * 2
        if SpanFeature.STERN_HIDDEN in self.span_features:
            self.span_embedding_dim += seq_in_size
        if SpanFeature.AVERAGE_TOKEN in self.span_features:
            self.span_embedding_dim += embed_dim
        if SpanFeature.AVERAGE_HIDDEN in self.span_features:
            self.span_embedding_dim += seq_in_size
        if SpanFeature.ATTENTION_TOKEN in self.span_features:
            self.span_embedding_dim += embed_dim
        if SpanFeature.ATTENTION_HIDDEN in self.span_features:
            self.span_embedding_dim += seq_in_size
        if SpanFeature.LENGTH in self.span_features:
            self.span_embedding_dim += c_enc.length_embed_dim
        assert self.span_embedding_dim > 0
        print(
            "Dimensions:",
            f"embed_dim = {embed_dim};",
            f"seq_in_size = {seq_in_size};",
            f"span_embed_dim = {self.span_embedding_dim}",
        )

    def _build_length_buckets(self):
        self.num_buckets = 9
        self.length_buckets = {0: 999}  # Accessing 0 should throw an error
        for i in range(1, 5):
            self.length_buckets[i] = i - 1
        for i in range(5, 8):
            self.length_buckets[i] = 4
        for i in range(8, 16):
            self.length_buckets[i] = 5
        for i in range(16, 32):
            self.length_buckets[i] = 6
        for i in range(32, 64):
            self.length_buckets[i] = 7

    def _build_word_dropout_probs(self, meta):
        self._dropout_probs = {}
        for i, token in enumerate(meta.vocab):
            if i <= 3:      # Special tokens
                continue
            self._dropout_probs[i] = 1. / (1 + meta.vocab_cnt[token])

    def forward(self, batch):
        seq_lengths = try_gpu(torch.tensor([x.length for x in batch]))
        max_seq_len = max(x.length for x in batch)

        # embedded_tokens: (batch_size, max_seq_len + 2, embed_dim)
        # lstm_out: (batch_size, max_seq_len + 2, seq_in_size)
        embedded_tokens, lstm_out = self.embed_stuff(batch, seq_lengths)

        # Step 0: construct a list of all possible spans
        spans = self.all_spans(max_seq_len)

        # Step 2: pass lstm output to the FFNN
        attention_weights = self.attention_ffnn(lstm_out)

        # Batch head-word reprsentation
        batch_representations = self.get_batch_representation(
            lstm_out, attention_weights, embedded_tokens, spans, seq_lengths
        )

        return batch_representations, spans, seq_lengths

    def embed_stuff(self, batch, seq_lengths):
        """
        Build tensors from the token indices

        Args:
            batch: List[Example]
        Returns:
            embedded_tokens: (batch_size, max_seq_len, embed_dim)
            lstm_out: (batch_size, max_seq_len, seq_in_size)
        """
        batch_token_indices = []
        for example in batch:
            token_indices = example.token_indices
            if self.word_dropout and self.training:
                # Do word-level dropout
                token_indices = [
                    self.UNK_index
                    if np.random.rand() < self._dropout_probs.get(x, 0)
                    else x
                    for x in token_indices
                ]
            # Pad with SOS and EOS
            token_indices = [self.SOS_index] + token_indices + [self.EOS_index]
            # Add to list
            token_indices = torch.tensor(token_indices)
            batch_token_indices.append(token_indices)
        padded_token_indices = pad_sequence(batch_token_indices, batch_first=True)
        padded_token_indices = try_gpu(padded_token_indices)
        # embedded_tokens: (batch_size, max_seq_len + 2, embed_dim)
        embedded_tokens = self.token_embedder(padded_token_indices)
        packed_embedded_tokens = pack_padded_sequence(
            embedded_tokens, seq_lengths + 2, batch_first=True,
        )
        packed_lstm_out, _ = self.lstm(packed_embedded_tokens)
        # lstm_out: (batch_size, max_seq_len + 2, seq_in_size)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
        return embedded_tokens, lstm_out

    def all_spans(self, seq_len: int) -> List[List[int]]:
        indices = []
        for span_size in range(seq_len):
            for span in range(seq_len - span_size):
                indices.append([span, span + span_size + 1])
        return indices

    def get_batch_representation(
        self,
        lstm_out: torch.Tensor,
        attention_weights: torch.Tensor,
        embedded_tokens: torch.Tensor,
        spans: List[List[int]],
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            lstm_out: (batch_size, max_seq_len + 2, seq_in_size)
                (seq_in_size is the total of LSTM hidden dims)
            attention_weights: (batch_size, max_seq_len + 2, 1)
            embedded_tokens: (batch_size, max_seq_len + 2, embed_dim)
                (embed_dim is the word embedding dim)
            spans: (num_spans, 2)
            seq_lengths: (batch_size,)
        Returns:
            representation: (batch_size, num_spans, ???)
        """
        batch_size, padded_max_seq_len, _ = attention_weights.size()
        max_seq_len = padded_max_seq_len - 2
        num_spans = len(spans)

        # Create python lists for span-features
        index_matrix = [None] * num_spans
        length_buckets = [0] * num_spans
        for i, span in enumerate(spans):
            index_matrix[i] = torch.zeros(padded_max_seq_len).index_put(
                [torch.arange(span[0] + 1, span[1] + 1)], torch.tensor([1.0])
            )
            length_buckets[i] = self.length_buckets.get(
                span[1] - span[0], self.num_buckets - 1
            )

        start_indices = try_gpu(torch.tensor([x[0] for x in spans]))
        end_indices = try_gpu(torch.tensor([x[1] for x in spans]))
        # (batch_size, num_spans, max_seq_len + 2)
        batch_index_matrix = try_gpu(
            torch.stack(index_matrix).expand(batch_size, -1, -1)
        )

        # List[(batch_size, num_span, ???)]
        things_to_concat: List[torch.Tensor] = []

        if SpanFeature.INSIDE_TOKEN in self.span_features:
            # Index select the start and end word embeddings
            # (batch_size, num_spans, embed_dim)
            start_token = embedded_tokens[:, start_indices + 1, :]
            # (batch_size, num_spans, embed_dim)
            end_token = embedded_tokens[:, end_indices, :]
            things_to_concat += [start_token, end_token]

        if SpanFeature.INSIDE_HIDDEN in self.span_features:
            # Index select the start and end lstm outputs.
            # (batch_size, num_spans, embed_dim)
            start_hidden = lstm_out[:, start_indices + 1, :]
            # (batch_size, num_spans, seq_in_size)
            end_hidden = lstm_out[:, end_indices, :]
            things_to_concat += [start_hidden, end_hidden]

        if SpanFeature.STERN_HIDDEN in self.span_features:
            forward_before = lstm_out[:, start_indices, :self.lstm_dim]
            forward_after = lstm_out[:, end_indices, :self.lstm_dim]
            backward_before = lstm_out[:, end_indices + 1, self.lstm_dim:]
            backward_after = lstm_out[:, start_indices + 1, self.lstm_dim:]
            things_to_concat += [
                forward_after - forward_before,
                backward_after - backward_before,
            ]

        if (
            SpanFeature.AVERAGE_TOKEN in self.span_features
            or SpanFeature.AVERAGE_HIDDEN in self.span_features
        ):
            average_weights = batch_index_matrix / torch.sum(
                batch_index_matrix, dim=2, keepdim=True
            )
            if SpanFeature.AVERAGE_TOKEN in self.span_features:
                # (batch_size, num_spans, embed_dim)
                average_token = torch.bmm(average_weights, embedded_tokens)
                things_to_concat.append(average_token)
            if SpanFeature.AVERAGE_HIDDEN in self.span_features:
                # (batch_size, num_spans, seq_in_size)
                average_hidden = torch.bmm(average_weights, lstm_out)
                things_to_concat.append(average_hidden)

        if (
            SpanFeature.ATTENTION_TOKEN in self.span_features
            or SpanFeature.ATTENTION_HIDDEN in self.span_features
        ):
            # Batch the attention weights.
            # (batch_size, 1, max_seq_len)
            attention = attention_weights.transpose(1, 2)
            # Element-wise multriplication of attention weights with index matrices
            # as 0-indices will zero-out irrelevant attention indices.
            # (batch_size, num_spans, max_seq_len)
            attention_matrix = batch_index_matrix * attention
            # Replace all 0 indices with -float('inf') in preparation for softmax.
            # Softmax the attentions.
            attention_matrix[batch_index_matrix == 0.0] = float("-inf")
            # (batch_size, num_spans, max_seq_len)
            attention_matrix_normalized = F.softmax(attention_matrix, dim=2)
            # Batch matrix multiplication to obtain representation weighted by
            # normalized attention weights.
            if SpanFeature.ATTENTION_TOKEN in self.span_features:
                # (batch_size, num_spans, embed_dim)
                attention_token = torch.bmm(
                    attention_matrix_normalized, embedded_tokens
                )
                things_to_concat.append(attention_token)
            if SpanFeature.ATTENTION_HIDDEN in self.span_features:
                # (batch_size, num_spans, seq_in_size)
                attention_hidden = torch.bmm(attention_matrix_normalized, lstm_out)
                things_to_concat.append(attention_hidden)

        if SpanFeature.LENGTH in self.span_features:
            # (num_spans,)
            length_buckets = try_gpu(torch.tensor(length_buckets))
            # (num_spans, length_embed_dim)
            length_embeddings = self.length_embeddings(length_buckets)
            # (batch_size, num_spans, length_embed_dim)
            things_to_concat.append(length_embeddings.expand(batch_size, -1, -1))

        # Concatenate all portions of the representations.
        representation = torch.cat(things_to_concat, dim=2)
        return representation

