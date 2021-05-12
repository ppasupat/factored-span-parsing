from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from spanparser.model.span_detection_decoder import SpanDetectionDecoder
from spanparser.model.edge_detection_decoder import EdgeDetectionDecoder
from spanparser.model.tree_decoder import TreeDecoder
from spanparser.model.utils import (
    DecodedEdges,
    DecodedSpan,
    ProtoNode,
    Tag,
)
from spanparser.utils import try_gpu


class CandidateBatch(NamedTuple):
    """
    Contains the scores of cell (start, end, parent label) for all parent labels,
    along with the information for backtracking.

    Attributes:
        start (int): start index
        end (int): end index
        scores (ndarray): size = (num_labels,); type = float
            The score in each cell (start, end, parent label).
        chains (ndarray): size = (num_labels,); type = int
            The chains that give the scores in `scores`.
            Used for backtracking.
        splits (ndarray): size = (num_labels,); type = int
            The binary split points that give the scores in `scores`.
            Used for backtracking.
    """

    start: int
    end: int
    scores: np.ndarray
    chains: np.ndarray
    splits: np.ndarray


# The type for the parse chart
# (start, end) -> CandidateBatch
ChartDict = Dict[Tuple[int, int], CandidateBatch]


class CKYWithEdgeScoresDecoder(TreeDecoder):
    """
    Parse with CKY algorithm.

    Dynamic programming cell: (i, j, parent label)

    To construct cells (i, j, p):
    - Children: If span length is 1, skip this step.
        Otherwise, for each label l, consider all splits
        (i, k, l) + (k, j, l). Let S[l] be the best score among all splits.
    - Non-terminal: Consider two options:
        - Do not build a chain at (i, j) (i.e., build a dummy node).
          The best score is S[p].
        - Build a chain c = (l_1, ..., l_k) at (i, j).
          The best score is S[l_k] + node_score(i, j, c) + edge_score(i, j, l_1, p).
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
        self.edge_detector = EdgeDetectionDecoder(
            config,
            meta,
            span_embedding_dim=span_embedding_dim,
            num_child_labels=len(meta.nt),
            num_parent_labels=len(meta.nt),
        )
        c_dec = config.model.decoder
        self.punt_on_training = c_dec.punt_on_training
        self.cost_augment = c_dec.cost_augment
        # dict(Tag -> List of label indices)
        self.nt_groups = meta.nt_groups
        # Intent used for punt_on_training
        self.first_intent = self.labels[self.nt_groups[Tag.INTENT][0]]
        # label index -> Tag
        self.tag_of = {}
        for tag, labels in self.nt_groups.items():
            for label in labels:
                self.tag_of[label] = tag
        # List of chains
        self.unary_chains = meta.unary_chains
        self.unary_chain_idx = meta.unary_chains_x
        # dict((head Tag, tail Tag) --> List of chains)
        self.unary_chain_groups = meta.unary_chain_groups
        # Lists for indexing arrays
        self.head_of_chain = [chain[0] for chain in self.unary_chains]
        self.tail_of_chain = [chain[-1] for chain in self.unary_chains]
        # Constraints (will be added to the score; -inf = invalid combination)
        self.constraints = np.zeros((len(self.unary_chains), len(self.labels)))
        self.top_intent_constraints = np.zeros(len(self.unary_chains))
        for chain_index, chain in enumerate(self.unary_chains):
            for label_index in range(len(self.labels)):
                if (
                    label_index not in self.tag_of
                    or self.tag_of[label_index] == self.tag_of[chain[0]]
                ):
                    self.constraints[chain_index, label_index] = float("-inf")
            if self.tag_of[chain[0]] != Tag.INTENT:
                self.top_intent_constraints[chain_index] = float("-inf")
        print("Labels:", self.labels_idx)
        print("Chains:", self.unary_chain_idx)

    def _score_spans(
        self,
        x_d: torch.Tensor,
        span_indices: List[List[int]],
        seq_lengths: torch.Tensor,
    ) -> List[Tuple[List[DecodedSpan], List[DecodedEdges]]]:
        """
        Returns both the node and edge scores.
        """
        batch_decoded_spans = self.span_detector(x_d, span_indices, seq_lengths)[0]
        batch_decoded_edges = self.edge_detector(x_d, span_indices, seq_lengths)
        return list(zip(batch_decoded_spans, batch_decoded_edges))

    def _build_tree(
        self,
        span_scores: Tuple[List[DecodedSpan], List[DecodedEdges]],
        seq_length: int,
        gold_proto_node: Optional[ProtoNode],
    ) -> Tuple[ProtoNode, torch.Tensor]:
        if self.punt_on_training and self.training:
            return ProtoNode(0, seq_length, self.first_intent), torch.tensor([0.0])

        decoded_spans, decoded_edges = span_scores
        span_cands = {(span.start, span.end): span.numpify() for span in decoded_spans}
        span_edges = {(span.start, span.end): span.numpify() for span in decoded_edges}
        chart: ChartDict = {}

        gold_chains = None
        if self.cost_augment != 0.0 and self.training:
            assert gold_proto_node is not None
            gold_chains = {
                key: self.unary_chain_idx[tuple(self.labels_idx.get(l, 0) for l in chain)]
                for (key, chain) in gold_proto_node.to_chains().items()
            }
            # Update the span_cands by adding 1 to each unmatching label
            span_cands = {
                key: span.add_cost(gold_chains.get(key, 999999), self.cost_augment)
                for (key, span) in span_cands.items()
            }

        for length in range(1, seq_length + 1):
            for i in range(seq_length - length + 1):
                j = i + length
                span = span_cands[i, j]
                edges = span_edges[i, j]
                null_cost = 0.0
                if gold_chains and (i, j) in gold_chains:
                    null_cost = self.cost_augment
                # Step 1: For each label l, find the best children.
                best_splits, split_scores = self._get_best_split(chart, span)
                # Step 2: For each parent label p, find the best combination of
                # span chain (or lack thereof) and children.
                if i == 0 and j == seq_length:
                    chart[i, j] = self._build_top_intent(
                        chart, best_splits, split_scores, span
                    )
                else:
                    chart[i, j] = self._build_non_terminal(
                        chart, best_splits, split_scores, null_cost, span, edges
                    )
                # print(f"Built {i, j} {chart[i, j]}")

        root_cand = chart[0, seq_length]
        assert root_cand is not None
        root_proto_node = self._build_proto_node(chart, root_cand)[0]
        predicted_score = torch.tensor([root_cand.scores])
        return root_proto_node, predicted_score

    def _get_best_split(
        self, chart: ChartDict, span: DecodedSpan
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns S[l] for all labels l. (See the class docstring for details.)

        Args:
            chart (ChartDict)
            span (DecodedSpan): Used for getting the start and end indices.

        Returns:
            splits (int vector of length num_labels)
            scores (float vector of length num_labels)
        """
        i, j = span.start, span.end
        if j == i + 1:
            # For span of length 1, just return zero scores
            return -np.ones(len(self.labels), dtype=int), np.zeros(len(self.labels))
        # Compute S[i, k] + S[k, j] for all k
        all_split_scores = np.stack(
            [chart[i, k].scores + chart[k, j].scores for k in range(i + 1, j)]
        )
        best_splits = np.argmax(all_split_scores, axis=0)
        split_scores = all_split_scores[best_splits, np.arange(len(self.labels))]
        return best_splits + i + 1, split_scores

    def _build_non_terminal(
        self,
        chart: ChartDict,
        best_splits: np.ndarray,
        split_scores: np.ndarray,
        null_cost: float,
        span: DecodedSpan,
        edges: DecodedEdges,
    ) -> CandidateBatch:
        """
        Computes the scores for the cells (i, j, p) for all parent labels p.
        (See the class docstring for details.)

        Args:
            chart (ChartDict)
            best_splits (ndarray): size = (num_labels,)
                The best_splits from _get_best_split.
                best_splits[l] = The split points that give S[l].
            split_scores (ndarray): size = (num_labels,)
                The split_scores from _get_best_split. This is S[l].
            null_cost (float): For cost-augmented decoding, this is the additional
                cost for building a dummy node. The costs for other labels
                are already added to the label scores.
            span (DecodedSpan): Contains label scores for the span (i, j).
            edges (DecodedEdges): Contains edge scores for the span (i, j).

        Returns:
            CandidateBatch
        """
        # Construct scores
        # Case 1: No chain (1, num_labels)
        no_nt_scores = split_scores.reshape(1, -1) + null_cost
        # Case 2: Build a chain
        # [1] Total children scores (num_chains, 1)
        #     For each chain c = (l_1, ..., l_k), look up S[l_k]
        children_scores = split_scores[self.tail_of_chain, np.newaxis]
        # [2] Chain scores (num_chains, 1)
        #     For each chain c = (l_1, ..., l_k), look up node_score(i, j, c)
        chain_scores = span.labels.reshape(-1, 1)
        # [3] Edge scores (num_chains, num_labels)
        #     For each chain c = (l_1, ..., l_k), look up edge_score(i, j, l_1, :)
        edge_scores = edges.edges[self.head_of_chain, :]
        # Sum the scores and concatenate Case 1 (num_chains + 1, num_labels)
        # Also use self.constraints to mask out invalid combinations.
        total_scores = np.concatenate(
            [
                children_scores + chain_scores + edge_scores + self.constraints,
                no_nt_scores,
            ],
            axis=0,
        )
        # Find the chains (or no-chain) that give the maximum scores (num_labels,)
        best_chains = np.argmax(total_scores, axis=0)
        # Get the corresponding scores (num_labels,)
        best_scores = total_scores[best_chains, np.arange(len(self.labels))]
        # Rearrange best_splits based on the tail label of each chain
        all_splits = np.concatenate(
            [
                np.tile(
                    best_splits[self.tail_of_chain, np.newaxis], (1, len(self.labels))
                ),
                best_splits.reshape(1, -1),
            ],
            axis=0,
        )
        # Get the best split points (num_labels,)
        best_splits = all_splits[best_chains, np.arange(len(self.labels))]
        return CandidateBatch(
            span.start, span.end, best_scores, best_chains, best_splits
        )

    def _build_top_intent(
        self,
        chart: ChartDict,
        best_splits: np.ndarray,
        split_scores: np.ndarray,
        span: DecodedSpan,
    ) -> CandidateBatch:
        """
        Computes the scores for the cells (0, seq_len, parent label = NONE).
        Since this is the top span, there is no edge score term.
        Also the dummy node case is not considered as the top label must be an intent.

        Args:
            chart (ChartDict)
            best_splits (ndarray): size = (num_labels,)
                The best_splits from _get_best_split.
                best_splits[l] = The split points that give S[l].
            split_scores (ndarray): size = (num_labels,)
                The split_scores from _get_best_split. This is S[l].
            span (DecodedSpan): Contains label scores for the span (i, j).

        Returns:
            CandidateBatch
            Note that the ndarray arguments in CandidateBatch are Python scalars
                (scores is float, chains and splits are ints) instead of ndarrays.
        """
        # Construct scores
        # [1] Total children scores (num_chains, 1)
        #     For each chain c = (l_1, ..., l_k), look up S[l_k]
        children_scores = split_scores[self.tail_of_chain]
        # [2] Chain scores (num_chains, 1)
        #     For each chain c = (l_1, ..., l_k), look up node_score(i, j, c)
        chain_scores = span.labels
        # Sum the scores and the constraints (num_chains, num_labels)
        total_scores = children_scores + chain_scores + self.top_intent_constraints
        # Find the chains that give the maximum scores (num_labels,)
        best_chains = np.argmax(total_scores, axis=0)
        # Return the scores, the chains, and the splits
        best_scores = total_scores[best_chains]
        best_splits = best_splits[self.tail_of_chain][best_chains]
        return CandidateBatch(
            span.start, span.end, best_scores, best_chains, best_splits
        )

    def _build_proto_node(
        self, chart: ChartDict, candidate: CandidateBatch, label: int = None
    ) -> List[ProtoNode]:
        """
        Performs backtracking and builds ProtoNodes.

        Args:
            chart (ChartDict)
            candidate (CandidateBatch): The candidate for the span being considered.
            label (int): If `label` is None, the current span is the top span
                (0, seq_len). Otherwise, `label` indicates the label index
                of `candidate.{scores,chains,splits}` to use.

        Returns:
            - If the candidate is a dummy node (label = None), return a list of
                ProtoNodes, one for each child Candidate.
            - Otherwise (candidate is a subtree with non-empty label), return a list
                of length 1 containing the ProtoNode for the candidate subtree.
        """
        i, j = candidate.start, candidate.end
        if label is None:
            # ROOT node
            chain_index = candidate.chains
            k = candidate.splits
        else:
            chain_index = candidate.chains[label]
            k = candidate.splits[label]
        if chain_index == len(self.unary_chains):
            # The best choice is to not build a chain (i.e., dummy node).
            assert label is not None, "Root span cannot be a dummy node"
            nodes = []
            if k >= 0:
                nodes.extend(self._build_proto_node(chart, chart[i, k], label))
                nodes.extend(self._build_proto_node(chart, chart[k, j], label))
            return nodes
        else:
            # The best choice is build a chain.
            chain = self.unary_chains[chain_index]
            # Build a chain of ProtoNode
            top_node = ProtoNode(i, j, self.labels[chain[0]])
            bottom_node = top_node
            for label_index in chain[1:]:
                child = ProtoNode(i, j, self.labels[label_index])
                bottom_node.children.append(child)
                bottom_node = child
            # Build the children and add them to the bottom ProtoNode
            if k >= 0:
                bottom_node.children.extend(
                    self._build_proto_node(chart, chart[i, k], chain[-1])
                )
                bottom_node.children.extend(
                    self._build_proto_node(chart, chart[k, j], chain[-1])
                )
            return [top_node]

    def _score_tree(
        self,
        span_scores: Tuple[List[DecodedSpan], List[DecodedEdges]],
        root_proto_node: ProtoNode,
    ) -> torch.Tensor:
        """
        Sums the unary chain scores and edge scores of all spans.

        At test time, unknown unary chains are simply skipped. This should happen
            only when scoring gold trees.
        """
        decoded_spans, decoded_edges = span_scores
        proto_node_stack: List[Tuple[ProtoNode, int, List[int]]] = [
            (root_proto_node, 0, [])
        ]
        span_cands = {(span.start, span.end): span for span in decoded_spans}
        edge_cands = {(span.start, span.end): span for span in decoded_edges}
        tree_score = 0
        while proto_node_stack:
            node, parent_label, chain_so_far = proto_node_stack.pop()
            # If it is still in a chain, push it back to the stack
            if (
                len(node.children) == 1
                and node.children[0].start == node.start
                and node.children[0].end == node.end
            ):
                proto_node_stack.append(
                    (
                        node.children[0],
                        parent_label,
                        chain_so_far + [self.labels_idx.get(node.label, 0)],
                    )
                )
            else:
                chain = tuple(chain_so_far + [self.labels_idx.get(node.label, 0)])
                # Node score
                if chain not in self.unary_chain_idx:
                    print("WARNING: chain {} not in training data".format(chain))
                else:
                    span = span_cands[node.start, node.end]
                    tree_score += span.labels[self.unary_chain_idx[chain]]
                # Edge score
                if parent_label != 0:
                    edges = edge_cands[node.start, node.end]
                    tree_score += edges.edges[chain[0]][parent_label]
                # Add the children
                for child in node.children:
                    proto_node_stack.append((child, self.labels_idx.get(node.label, 0), []))
        if isinstance(tree_score, int):
            # This can happen if all unary chains are unknown.
            print("WARNING: tree_score is 0")
            return try_gpu(torch.zeros(1))
        else:
            return tree_score.view(1)  # noqa
