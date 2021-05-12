from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from spanparser.model.span_detection_decoder import SpanDetectionDecoder
from spanparser.model.tree_decoder import TreeDecoder
from spanparser.model.utils import (
    DecodedSpan,
    NumpifiedDecodedSpan,
    ProtoNode,
    Tag,
)
from spanparser.utils import try_gpu


class Candidate(NamedTuple):
    """
    A Candidate for the CKY algorithm.

    Attributes:
        start (int): start index
        end (int): end index
        label:
            - If None, the candidate is a dummy node (collection of subtrees).
            - Otherwise, the candidate represents a subtree whose subtree root
                is a unary chain with the labels in `label`.
        children (List[Any]): list of children Candidate objects.
        score (float): score of the candidate, which is the sum of unary chain
            scores of all constituents.
    """

    start: int
    end: int
    label: Optional[List[str]]
    children: List[Any]  # List[Candidate]
    score: float

    def __repr__(self):
        return "({}, {}, {}, {}, {})".format(
            self.start, self.end, self.label, self.children, self.score
        )

    @staticmethod
    def argmax(*args):
        return max(args, key=lambda x: float("-inf") if x is None else x.score)


# The type for the parse chart
# (start, end, tag of self) -> highest-scoring Candidate
ChartDict = Dict[Tuple[int, int, Tag], Candidate]


class CKYWithNodeScoresDecoder(TreeDecoder):
    """
    Parse with CKY algorithm.

    Dynamic programming cell: (i, j, tag), where tag is INTENT or SLOT

    To construct cells (i, j, t):
    - Children: If span length is 1, skip this step.
        Otherwise, for each r (INTENT or SLOT), consider all splits
        (i, k, r) + (k, j, r). Let S[r] be the best score among all splits.
    - Non-terminal: Consider two options:
        - Do not build a chain at (i, j) (i.e., build a dummy node).
          The best score is S[t].
        - Build a chain c = (l_1, ..., l_k) at (i, j) where tag(l_1) = t.
          The best score is S[~tag(l_k)] + node_score(i, j, c).
          (~ = opposite tag)
    - Among all these combinations, find the ones with the highest score.

    Note that the existence score in DecodedSpan is not used.

    Arbitrary binarization: Dummy nodes have score 0, so any binarization will
        give the same overall tree score.

    Note: The Tensor returned by _build_tree is DETACHED, meaning the gradient will
    not flow through. If gradient is needed (e.g., in margin loss), set the config
    rescore_prediction = True (See TreeDecoder.Config) to get a non-detached Tensor.
    """

    def __init__(self, config, meta, span_embedding_dim):
        """
        Configs:
            punt_on_training (bool): Output a fake tree during training to speed
                things up.
            cost_augment (float): If non-zero, during training, add this amount
                to the scores of all chains (including the NULL chain)
                that disagree with the gold chains.
        """
        super().__init__(config, meta, span_embedding_dim)
        self.span_detector = SpanDetectionDecoder(
            config,
            meta,
            span_embedding_dim=span_embedding_dim,
            num_labels=len(meta.unary_chains),
        )
        c_dec = config.model.decoder
        self.punt_on_training = c_dec.punt_on_training
        self.cost_augment = c_dec.cost_augment
        if self.cost_augment != 0.0:
            assert not self.punt_on_training, (
                "cost_augment is performed during training; "
                "punt_on_training should be set to False"
            )
        # Intent used for punt_on_training
        self.first_intent = self.labels[meta.nt_groups[Tag.INTENT][0]]
        # Record unary chains
        self.unary_chains = meta.unary_chains
        self.chains_idx = meta.unary_chains_x
        self.unary_chain_groups = meta.unary_chain_groups
        # chains_idx_ranges[a, b] = the start and end unary chain indices
        #   of the unary chain group (a, b) (where a and b are INTENT or SLOT)
        self.chains_idx_ranges = {}
        num_chains_so_far = 0
        for key, chains in self.unary_chain_groups.items():
            for i, chain in enumerate(chains):
                assert self.chains_idx[chain] == num_chains_so_far + i
            self.chains_idx_ranges[key] = (
                num_chains_so_far,
                num_chains_so_far + len(chains),
            )
            num_chains_so_far += len(chains)
        print("Labels:", self.labels_idx)
        print("Chains:", self.chains_idx)
        print("Unary chain groups:", self.unary_chain_groups)
        print("Chain ranges:", self.chains_idx_ranges)

    def _score_spans(
        self,
        x_d: torch.Tensor,
        span_indices: List[List[int]],
        seq_lengths: torch.Tensor,
    ) -> List[Any]:
        return self.span_detector(x_d, span_indices, seq_lengths)[0]

    def _build_tree(
        self,
        span_scores: List[DecodedSpan],
        seq_length: int,
        gold_proto_node: Optional[ProtoNode],
    ) -> Tuple[ProtoNode, torch.Tensor]:
        if self.punt_on_training and self.training:
            return ProtoNode(0, seq_length, self.first_intent), torch.tensor([0.0])

        span_cands = {(span.start, span.end): span.numpify() for span in span_scores}
        # (start, end, tag of self) -> highest-scoring Candidate
        chart: ChartDict = {}

        gold_chains = None
        if self.cost_augment != 0.0 and self.training:
            assert gold_proto_node is not None
            gold_chains = {
                key: self.chains_idx[tuple(self.labels_idx.get(l, 0) for l in chain)]
                for (key, chain) in gold_proto_node.to_chains().items()
            }
            # Update the span_cands by adding 1 to each unmatching label
            span_cands = {
                key: span.add_cost(gold_chains.get(key, 999_999), self.cost_augment)
                for (key, span) in span_cands.items()
            }

        # Base case: length = 1
        for i in range(seq_length):
            j = i + 1
            span = span_cands[i, j]
            top_level = i == 0 and j == seq_length
            # Dummy node (no non-terminal) is allowed except at the top level
            if top_level:
                token_cand = None
            else:
                null_cost = 0.0
                if gold_chains and (i, j) in gold_chains:
                    null_cost = self.cost_augment
                token_cand = Candidate(i, j, None, [], null_cost)
            SI_cand = self._build_non_terminal(span, Tag.SLOT, Tag.INTENT, [], 0.0)
            II_cand = self._build_non_terminal(span, Tag.INTENT, Tag.INTENT, [], 0.0)
            SS_cand = self._build_non_terminal(span, Tag.SLOT, Tag.SLOT, [], 0.0)
            IS_cand = self._build_non_terminal(span, Tag.INTENT, Tag.SLOT, [], 0.0)
            chart[i, j, Tag.SLOT] = Candidate.argmax(SI_cand, SS_cand, token_cand)
            chart[i, j, Tag.INTENT] = Candidate.argmax(II_cand, IS_cand, token_cand)
            # print(f"Built {i, j, Tag.SLOT} {chart[i, j, Tag.SLOT]}")
            # print(f"Built {i, j, Tag.INTENT} {chart[i, j, Tag.INTENT]}")

        # Inductive case: length > 1
        for length in range(2, seq_length + 1):
            for i in range(seq_length - length + 1):
                j = i + length
                span = span_cands[i, j]
                top_level = i == 0 and j == seq_length
                null_cost = 0.0
                if gold_chains and (i, j) in gold_chains:
                    null_cost = self.cost_augment

                # Have slots as children
                best_children_combo, children_score = self._get_best_split(
                    chart, span, Tag.SLOT
                )
                # Dummy node (no non-terminal) is allowed except at the top level
                if top_level:
                    dummy_S_cand = None
                else:
                    dummy_S_cand = Candidate(
                        i, j, None, best_children_combo, children_score + null_cost
                    )
                SI_cand = self._build_non_terminal(
                    span, Tag.SLOT, Tag.INTENT, best_children_combo, children_score
                )
                II_cand = self._build_non_terminal(
                    span, Tag.INTENT, Tag.INTENT, best_children_combo, children_score
                )

                # Have intents as children
                best_children_combo, children_score = self._get_best_split(
                    chart, span, Tag.INTENT
                )
                # Dummy node (no non-terminal) is allowed except at the top level
                if top_level:
                    dummy_I_cand = None
                else:
                    dummy_I_cand = Candidate(
                        i, j, None, best_children_combo, children_score + null_cost
                    )
                SS_cand = self._build_non_terminal(
                    span, Tag.SLOT, Tag.SLOT, best_children_combo, children_score
                )
                IS_cand = self._build_non_terminal(
                    span, Tag.INTENT, Tag.SLOT, best_children_combo, children_score
                )

                # Fill in the chart
                chart[i, j, Tag.SLOT] = Candidate.argmax(SI_cand, SS_cand, dummy_S_cand)
                chart[i, j, Tag.INTENT] = Candidate.argmax(
                    II_cand, IS_cand, dummy_I_cand
                )
                # print(f"Built {i, j, Tag.SLOT} {chart[i, j, Tag.SLOT]}")
                # print(f"Built {i, j, Tag.INTENT} {chart[i, j, Tag.INTENT]}")

        root_cand = chart[0, seq_length, Tag.INTENT]
        root_proto_node = self._build_proto_node(root_cand)[0]
        predicted_score = torch.tensor([root_cand.score])
        return root_proto_node, predicted_score

    def _build_non_terminal(
        self,
        span: DecodedSpan,
        top_tag: Tag,
        bottom_tag: Tag,
        children: List[Candidate],
        children_score: float,
    ) -> Optional[Candidate]:
        """
        Find a nonterminal label or chain with the best score.
        Only consider chains that satisfy the given top_tag and bottom_tag.
        """
        chunk = self.chains_idx_ranges[top_tag, bottom_tag]
        if chunk[0] == chunk[1]:
            return None
        label_scores = span.labels[chunk[0] : chunk[1]]
        best_chain_idx = np.argmax(label_scores, 0)
        best_chain = self.unary_chain_groups[top_tag, bottom_tag][int(best_chain_idx)]
        label = [self.labels[i] for i in best_chain]
        score = label_scores[best_chain_idx] + children_score
        return Candidate(span.start, span.end, label, children, score)

    def _get_best_split(
        self, chart: ChartDict, span: NumpifiedDecodedSpan, children_tag: Tag
    ) -> Tuple[List[Candidate], float]:
        """
        Find the split [i,j] -> [i,k] + [k,j] with the highest score.
        The children's tags must be equal to children_tag.
        """
        # Find the best child combination
        i, j = span.start, span.end
        best_children: List[Candidate] = []
        best_score = float("-inf")
        for k in range(i + 1, j):
            left = chart[i, k, children_tag]
            right = chart[k, j, children_tag]
            score = left.score + right.score
            if score > best_score:
                best_score = score
                best_children = [left, right]
        assert len(best_children) != 0
        return best_children, best_score

    def _build_proto_node(self, candidate: Candidate) -> List[ProtoNode]:
        """
        Convert the candidate into ProtoNodes.

        Args:
            candidate (Candidate)

        Returns:
            - If the candidate is a dummy node (label = None), return a list of
                ProtoNodes, one for each child Candidate.
            - Otherwise (candidate is a subtree with non-empty label), return a list
                of length 1 containing the ProtoNode for the candidate subtree.
        """
        if candidate.label is None:
            nodes = []
            for child in candidate.children:
                nodes.extend(self._build_proto_node(child))
            return nodes
        else:
            top_node = ProtoNode(candidate.start, candidate.end, candidate.label[0])
            bottom_node = top_node
            for label in candidate.label[1:]:
                child = ProtoNode(candidate.start, candidate.end, label)
                bottom_node.children.append(child)
                bottom_node = child
            for child in candidate.children:
                bottom_node.children.extend(self._build_proto_node(child))
            return [top_node]

    def _score_tree(
        self, span_scores: List[DecodedSpan], root_proto_node: ProtoNode
    ) -> torch.Tensor:
        """
        Sums the unary chain scores of all spans.

        At test time, unknown unary chains are simply skipped. This should happen
            only when scoring gold trees.
        """
        proto_node_stack: List[Tuple[ProtoNode, List[int]]] = [(root_proto_node, [])]
        span_cands = {(span.start, span.end): span for span in span_scores}
        tree_score = 0
        while proto_node_stack:
            node, chain_so_far = proto_node_stack.pop()
            # If it is still in a chain, push it back to the stack
            if (
                len(node.children) == 1
                and node.children[0].start == node.start
                and node.children[0].end == node.end
            ):
                proto_node_stack.append(
                    (node.children[0], chain_so_far + [self.labels_idx.get(node.label, 0)])
                )
            else:
                chain = tuple(chain_so_far + [self.labels_idx.get(node.label, 0)])
                if chain not in self.chains_idx:
                    print("WARNING: chain {} not in training data".format(chain))
                else:
                    span = span_cands[node.start, node.end]
                    tree_score += span.labels[self.chains_idx[chain]]
                for child in node.children:
                    proto_node_stack.append((child, []))
        if isinstance(tree_score, int):
            # This can happen if all unary chains are unknown.
            print("WARNING: tree_score is 0")
            return try_gpu(torch.zeros(1))
        else:
            return tree_score.view(1)  # noqa
