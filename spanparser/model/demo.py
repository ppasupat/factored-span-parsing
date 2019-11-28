from __future__ import (absolute_import, division, print_function)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
)

from spanparser.model.base import Model
from spanparser.utils import try_gpu, BOS_INDEX, EOS_INDEX


class DemoModel(Model):
    """
    A simple text classification model.
    """

    def __init__(self, config, meta):
        super(DemoModel, self).__init__(config, meta)
        c_model = config.model

        self.token_embedder = nn.Embedding(
            len(meta.vocab),
            c_model.embed_dim,
        )
        self.dropout = nn.Dropout(c_model.dropout)

        # input: (batch, seq_len, embed_dim)
        # output: (hiddens, (h_n, c_n))
        # hiddens: (batch, seq_len, 2 * lstm_dim)
        # h_n and c_n: (lstm_layers * 2, batch, lstm_dim)
        #       [batch is at index 1 even when batch_first=True!]
        self.lstm = nn.LSTM(
            c_model.embed_dim,
            c_model.lstm_dim,
            batch_first=True,
            num_layers=c_model.lstm_layers,
            bidirectional=True,
            dropout=c_model.dropout,
        )
        self.hiddens_dim = 2 * c_model.lstm_dim
        self.lstm_out_dim = 4 * c_model.lstm_layers * c_model.lstm_dim

        # input: (batch, lstm_out_dim)
        # output: (batch, num_labels)
        self.mlp = nn.Sequential(
            self.dropout,
            nn.Linear(self.lstm_out_dim, c_model.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(c_model.mlp_hidden_dim, len(meta.labels)),
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        """
        Return "logit" that can be read by get_loss and get_pred.
        """
        # embedded_tokens: (batch, seq_len, embed_dim)
        # hiddens: (batch, seq_len, hiddens_dim)
        # lstm_out: (batch, lstm_out_dim)
        embedded_tokens, hiddens, lstm_out = self.embed_stuff(batch)

        # logits: (batch, num_labels)
        logits = self.mlp(lstm_out)

        return logits

    def embed_stuff(self, batch):
        """
        Build tensors from the token indices

        Args:
            batch: List[Example]
        Returns:
            embedded_tokens: (batch, padded_seq_len, embed_dim)
            hiddens: (batch, padded_seq_len, hiddens_dim)
            lstm_out: (batch, lstm_out_dim)
                where padded_seq_len = max seq_len + 2 (for BOS and EOS)
        """
        # Construct a batch of token indices
        batch_token_indices = []
        for example in batch:
            token_indices = example.token_indices
            # Pad with BOS and EOS
            token_indices = [BOS_INDEX] + token_indices + [EOS_INDEX]
            # Add to list
            token_indices = torch.tensor(token_indices)
            batch_token_indices.append(token_indices)
        padded_token_indices = pad_sequence(batch_token_indices, batch_first=True)
        padded_token_indices = try_gpu(padded_token_indices)

        # embedded_tokens: (batch, padded_seq_len, embed_dim)
        embedded_tokens = self.token_embedder(padded_token_indices)
        embedded_tokens = self.dropout(embedded_tokens)

        # hiddens: (batch, padded_seq_len, hiddens_dim)
        # h_n and c_n: (lstm_layers * 2, batch, lstm_dim)
        seq_lengths = try_gpu(torch.tensor([x.length for x in batch]))
        packed_embedded_tokens = pack_padded_sequence(
            embedded_tokens, seq_lengths + 2, batch_first=True,
        )
        packed_lstm_out, (h_n, c_n) = self.lstm(packed_embedded_tokens)
        hiddens, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)

        # lstm_out: (batch, lstm_out_dim)
        lstm_out = torch.cat([
            h_n.transpose(0, 1).reshape(len(batch), -1),
            c_n.transpose(0, 1).reshape(len(batch), -1),
        ], dim=1)

        return embedded_tokens, hiddens, lstm_out

    def get_loss(self, logit, batch):
        """
        Args:
            logit: Output from forward(batch)
            batch: list[Example]
        Returns:
            a scalar tensor
        """
        labels = [example.label_index for example in batch]
        labels = try_gpu(torch.tensor(labels))
        loss = self.loss(logit, labels)
        return loss

    def get_pred(self, logit, batch):
        """
        Args:
            logit: Output from forward(batch)
            batch: list[Example]
        Returns:
            predicted_indices: (batch,)
            predicted_scores: (batch, num_labels)
        """
        return logit.argmax(dim=1), logit
