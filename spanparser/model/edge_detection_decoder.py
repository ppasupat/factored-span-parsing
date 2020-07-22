from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from spanparser.model.utils import DecodedEdges


class ChildLabelScheme(object):
    # Concat the child label to the span representation
    CONCAT = "concat"
    # Put the child label in softmax
    SOFTMAX = "softmax"


class EdgeDetectionDecoder(nn.Module):
    """
    A decoder which assigns a score for each edge:
        edge_score(child span embedding, child label, parent label)

    This measures the compatibility of the parent *label* with respect to the child
    (child span embedding + child label). It does not use the parent span embedding
    since that will not be available at scoring time in the CKY algorithm.

    The computation involves passing the child span embedding through a feedforward
    network, followed by a softmax. Depending on the config, the output is either
    a softmax over the parent labels, or a softmax over combinations
    (child label, parent label).
    """

    def __init__(
        self, config, meta, span_embedding_dim, num_child_labels, num_parent_labels
    ) -> None:
        """
        Configs:
            child_label_scheme: How the label of the child span is used.
                - CONCAT: Concatenate the span embedding and child label embedding
                    and pass it to MLP. The softmax is over the parent labels.
                - SOFTMAX: Pass only the span embedding to MLP. The softmax
                    is over all combinations (child label, parent label).
            child_label_embed_size (int): Embedding size for child label
            projection_size (int): Project the span embedding to this size first.
                If 0 or negative, do not perform projection.
            dropout (float): Dropout before each linear layer.
            mlp_layers (int): Number of layers, including the final logit layer.
                Must be at least 1.
            mlp_dim (int): Dimension of each hidden layer.
        """
        super().__init__()
        c_ded = config.model.decoder.edge_detector
        self.num_child_labels = num_child_labels
        self.num_parent_labels = num_parent_labels
        self.child_label_scheme = c_ded.child_label_scheme

        # Project the span embedding
        if c_ded.projection_size <= 0:
            self.projection = lambda x: x
            in_dim = span_embedding_dim
        else:
            self.projection = nn.Sequential(
                nn.Dropout(config.model.dropout),
                nn.Linear(span_embedding_dim, c_ded.projection_size),
                nn.ReLU(),
            )
            in_dim = c_ded.projection_size

        if self.child_label_scheme == ChildLabelScheme.CONCAT:
            # For CONCAT, build a child label embedding
            self.child_embedding = nn.Embedding(
                num_child_labels, c_ded.child_label_embed_size
            )
            in_dim += c_ded.child_label_embed_size
            out_dim = num_parent_labels
        elif self.child_label_scheme == ChildLabelScheme.SOFTMAX:
            # For SOFTMAX, increase the number of outputs
            self.child_embedding = None
            out_dim = num_child_labels * num_parent_labels
        else:
            raise ValueError(f"Unknown child_label_scheme {self.child_label_scheme}")

        # MLP
        layers = []
        for _ in range(c_ded.mlp_layers - 1):
            layers.append(nn.Dropout(config.model.dropout))
            layers.append(nn.Linear(in_dim, c_ded.mlp_dim))
            layers.append(nn.ReLU())
            in_dim = c_ded.mlp_dim
        layers.append(nn.Dropout(config.model.dropout))
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_d: torch.Tensor,
        span_indices: List[List[int]],
        seq_lengths: torch.Tensor,
    ) -> List[List[DecodedEdges]]:
        """
        Apply the feedforward network to compute the edge scores.

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
            decoded_edges (List[List[DecodedEdges]]): The decoded edges for each
                example in the batch. Does not include out-of-bound spans.
        """
        # size of x_d = (batch_size, num_spans, span_embedding_dim)
        batch_size, num_spans, _ = x_d.size()
        x_d = self.projection(x_d)
        if self.child_label_scheme == ChildLabelScheme.CONCAT:
            x_d = x_d.unsqueeze(2).expand(-1, -1, self.num_child_labels, -1)
            l_d = self.child_embedding.weight.expand(batch_size, num_spans, -1, -1)
            x_d = torch.cat([x_d, l_d], dim=3)
        scores = self.mlp(x_d)
        if self.child_label_scheme == ChildLabelScheme.SOFTMAX:
            scores = scores.reshape(
                batch_size, num_spans, self.num_child_labels, self.num_parent_labels
            )
        # Normalize so that for all child labels, sum_{parent labels} scores = 1
        scores = F.log_softmax(scores, dim=3)

        # Pack into DecodedEdges objects.
        decoded_spans = []
        for i in range(batch_size):
            spans = []
            seq_length = seq_lengths[i].item()
            # loop through all the span representations
            for j in range(num_spans):
                start, end = span_indices[j]
                if end > seq_length:
                    continue
                spans.append(DecodedEdges(start, end, scores[i][j]))
            decoded_spans.append(spans)

        return decoded_spans
