from typing import List, Tuple

import torch
import torch.nn as nn

from spanparser.model.utils import DecodedSpan


class SpanDetectionDecoder(nn.Module):
    """
    A decoder which assigns labels and existence scores to each encoded span.

    The decoder takes the span embeddings and apply a feedforward network.
    The outputs are logits for existence and label scores.
    """

    def __init__(self, config, meta, span_embedding_dim, num_labels):
        """
        Configs:
            dropout (float): Dropout before each linear layer.
            mlp_layers (int): Number of layers, including the final logit layer.
                Must be at least 1.
            mlp_dim (int): Dimension of each hidden layer.
        """
        super().__init__()
        c_dsd = config.model.decoder.span_detector
        assert c_dsd.mlp_layers >= 1
        in_dim = span_embedding_dim

        layers = []
        for _ in range(c_dsd.mlp_layers - 1):
            layers.append(nn.Dropout(config.model.dropout))
            layers.append(nn.Linear(in_dim, c_dsd.mlp_dim))
            layers.append(nn.ReLU())
            in_dim = c_dsd.mlp_dim
        layers.append(nn.Dropout(config.model.dropout))
        self.mlp = nn.Sequential(*layers)

        self.existence_out = nn.Linear(in_dim, 1)
        self.label_out = nn.Linear(in_dim, num_labels)

    def forward(
        self,
        x_d: torch.Tensor,
        span_indices: List[List[int]],
        seq_lengths: torch.Tensor,
    ) -> Tuple[List[List[DecodedSpan]], torch.Tensor]:
        """
        Apply the feedforward network to compute the existence and label scores.

        Args:
            x_d (Tensor): size = (batch_size, num_spans, span_embedding_dim)
                Span embeddings. When batch_size > 1, this will also include
                out-of-bound spans for shorter utterances in the batch.
            span_indices (List): size = (num_spans, 2)
                The start and end indices for the spans.
            seq_length (Tensor): size = (batch_size,)
                The number of tokens for each utterance in the batch.
                Used to exclude out-of-bound spans.

        Returns:
            decoded_spans (List[List[DecodedSpan]]): The decoded spans for each
                example in the batch. Does not include out-of-bound spans.
            seq_length (Tensor): Same as the argument seq_length.
        """
        # Apply MLP and the final layers
        batch_size, num_spans, _ = x_d.size()
        mlp_out = self.mlp(x_d)
        # size = (batch_size, num_spans, 1)
        all_existence_d = self.existence_out(mlp_out)
        # size = (batch_size, num_spans, num_labels)
        all_label_d = self.label_out(mlp_out)

        decoded_spans = []
        for i in range(batch_size):
            spans = []
            seq_length = seq_lengths[i].item()
            # loop through all the span representations
            for j in range(num_spans):
                start, end = span_indices[j]
                if end > seq_length:
                    continue
                label_d = all_label_d[i][j]
                existence_d = all_existence_d[i][j]
                spans.append(DecodedSpan(start, end, label_d, existence_d))
            decoded_spans.append(spans)

        return decoded_spans, seq_lengths
