from typing import Any, List, Optional, Tuple

import torch
from torch import nn

from spanparser.model.utils import ProtoNode, TreeParserOutput


class DecoderBase(nn.Module):
    """
    An abstract class for span-based tree parsers.

    Takes the span embeddings and builds a tree for each example in the batch.

    A child class of DecoderBase should implement the following methods:
    - _score_spans: Computes the necessary span scores
    - _build_tree: Builds the highest-scoring tree. Returns the tree and the score.
    - _score_tree: Scores a given tree. Returns the score.

    Attributes:
        labels (List[str]): The canonical list of non-terminal labels.
        labels_idx (Dict[str, int]): The index lookup for `labels`.
        verify_score (bool): [for debugging] Verify that the score from
            _build_tree agrees with the score from _score_tree.
        verify_decoder (bool): [for debugging] Verify that the tree from
            _build_tree has a higher score than the gold tree.
        rescore_prediction (bool): Override the score from _build_tree
            with the result from _score_tree. Useful then _build_tree returns
            a detached Tensor (e.g., in CKYWith{Node|Edge}ScoresDecoder).
    """

    def __init__(self, config, meta, span_embedding_dim):
        """
        Configs:
            verify_score (bool): [for debugging] Verify that the score from
                _build_tree agrees with the score from _score_tree.
            verify_decoder (bool): [for debugging] Verify that the tree from
                _build_tree has a higher score than the gold tree.
            rescore_prediction (bool): Override the score from _build_tree
                with the result from _score_tree. Useful then _build_tree returns
                a detached Tensor (e.g., in CKYWith{Node|Edge}ScoresDecoder).
        """
        super().__init__()
        self.span_embedding_dim = span_embedding_dim
        c_dec = config.model.decoder
        self.verify_score = c_dec.verify_score
        self.verify_decoder = c_dec.verify_decoder
        self.rescore_prediction = c_dec.rescore_prediction
        self.labels = meta.nt
        self.labels_idx = meta.nt_x

    def forward(
        self,
        batch,
        x_d: torch.Tensor,
        span_indices: List[List[int]],
        seq_lengths: torch.Tensor,
    ) -> TreeParserOutput:
        """
        Predict a tree and score the gold tree.

        Args:
            batch: List[Example]
            x_d: size = (batch_size, num_spans, span_embedding_dim)
                Span embeddings
            span_indices: Array of size (num_spans, 2)
                The (start, end) indices of each span in x_d
            seq_length: size = (batch_size,)
                The number of tokens in each input utterance

        Returns:
            A TreeParserOutput object.
        """
        batch_span_scores = self._score_spans(x_d, span_indices, seq_lengths)
        batch_parses: List[Any] = []
        batch_predicted_scores: List[Any] = []
        batch_golds: List[Any] = []
        batch_golden_scores: List[Any] = []
        batch_tree_equality: List[bool] = []
        batch_tree_distances: List[float] = []

        for i, span_scores in enumerate(batch_span_scores):
            golden_proto_node = ProtoNode.from_actual_node(batch[i].tree)
            golden_score = self._score_tree(span_scores, golden_proto_node)
            # Note: predicted_score might be DETACHED (no gradient propagation)
            predicted_proto_node, predicted_score = self._build_tree(
                span_scores,
                int(seq_lengths[i]),
                golden_proto_node if self.training else None,
            )
            if self.verify_score:
                score_tree_out = self._score_tree(span_scores, predicted_proto_node)
                assert abs(float(predicted_score - score_tree_out)) < 1e-4, (
                    "build_tree: {} | score_tree: {}\ntree = {}"
                ).format(predicted_score, score_tree_out, predicted_proto_node)
            if self.rescore_prediction:
                predicted_score = self._score_tree(span_scores, predicted_proto_node)
            if self.verify_decoder and float(golden_score - predicted_score) > 1e-4:
                print(
                    "WARNING: gold_score {} > pred_score {}. Bad decoder?".format(
                        golden_score, predicted_score
                    ),
                    "\nGOLD: {}".format(golden_proto_node),
                    "\nPRED: {}".format(predicted_proto_node),
                )
            distance = ProtoNode.hamming_distance(
                predicted_proto_node, golden_proto_node
            )
            batch_parses.append(predicted_proto_node)
            batch_predicted_scores.append(predicted_score)
            batch_golds.append(golden_proto_node)
            batch_golden_scores.append(golden_score)
            batch_tree_equality.append(distance == 0)
            batch_tree_distances.append(distance)
        return TreeParserOutput(
            batch_span_scores=batch_span_scores,
            batch_parses=batch_parses,
            batch_predicted_scores=torch.cat(batch_predicted_scores),
            batch_golds=batch_golds,
            batch_golden_scores=torch.cat(batch_golden_scores),
            batch_tree_equality=batch_tree_equality,
            batch_tree_distances=batch_tree_distances,
        )

    def _score_spans(
        self,
        x_d: torch.Tensor,
        span_indices: List[List[int]],
        seq_lengths: torch.Tensor,
    ) -> List[Any]:
        """
        Computethe necessary span scores for the given input batch.

        Args:
            x_d: size = (batch_size, num_spans, span_embedding_dim)
                Span embeddings
            span_indices: Array of size (num_spans, 2)
                The (start, end) indices of each span in x_d
            seq_length: size = (batch_size,)
                The number of tokens in each input utterance
        Returns:
            List of length batch_size.
        """
        raise NotImplementedError

    def _build_tree(
        self, span_scores: Any, seq_length: int, gold_proto_node: Optional[ProtoNode]
    ) -> Tuple[ProtoNode, torch.Tensor]:
        """
        Builds the highest-scoring tree. Return the root ProtoNode and the score.

        Args:
            span_scores: Results from _score_spans.
            seq_length: Number of tokens.
            gold_proto_node: The gold tree. Useful for cost-augmented decoding.

        Returns:
            The parsed tree (root ProtoNode) and its score (scalar Tensor).
        """
        raise NotImplementedError

    def _score_tree(self, span_scores: Any, root_proto_node: ProtoNode) -> torch.Tensor:
        """
        Scores the given tree (specified by the root ProtoNode).

        Args:
            span_scores: Results from _score_spans.
            root_proto_node: The tree to be scored.

        Returns:
            The tree score (scalar Tensor).
        """
        raise NotImplementedError
