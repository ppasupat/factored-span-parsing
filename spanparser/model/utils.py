# Data structure and various tools
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import torch


from spanparser.data.top import Token, Tree
from spanparser.utils import var_to_numpy, INTENT, SLOT


class DecodedSpan(NamedTuple):
    """
    Stores the span scores.

    Attributes:
        start (int): Span start index (inclusive).
        end (int): Span end index (exclusive).
        labels (FloatTensor): The score for each label.
            size = (num_labels,)
        existence (FloatTensor): Span existence score (unused in CKY decoders).
            size = (1,)
    """

    start: int
    end: int
    labels: torch.Tensor
    existence: torch.Tensor

    def convert(self):
        """
        Returns a Span object with the highest-scoring label.
        """
        return Span(self.start, self.end, torch.max(self.labels, 0)[1])

    def convert_compositional(self, threshold: float):
        """
        Returns Span objects with labels that have scores exceeding the threshold.
        """
        spans = []
        for i in range(self.labels.size()[0]):
            if self.labels[i] > threshold:
                spans.append(Span(self.start, self.end, torch.tensor(i)))
        return spans

    def numpify(self):
        """
        Returns a copy but with Tensors converted into numpy arrays.
        The unused existence score is dropped.
        """
        return NumpifiedDecodedSpan(self.start, self.end, var_to_numpy(self.labels))


class NumpifiedDecodedSpan(NamedTuple):
    start: int
    end: int
    labels: np.ndarray

    def add_cost(self, gold_label: int, factor: float) -> "NumpifiedDecodedSpan":
        """
        Adds a cost of `factor` to unmatching labels. Used in margin loss with
        cost-augmented decoding.
        """
        cost = (np.arange(len(self.labels)) != gold_label).astype(float) * factor
        return NumpifiedDecodedSpan(self.start, self.end, self.labels + cost)


class Span(NamedTuple):
    """
    Represents a span and its label(s).

    Attributes:
        start (int): Span start index (inclusive).
        end (int): Span end index (exclusive).
        labels (Tensor): Either a single integer or a k-hot binary array of length
            num_labels, depending on the usage.
    """

    start: int
    end: int
    label: torch.Tensor


class DecodedEdges(NamedTuple):
    """
    Stores the edge scores.

    Attributes:
        start (int): Span start index (inclusive).
        end (int): Span end index (exclusive).
        edges (Tensor): The edge scores.
            size = (num_labels, num_labels)
            The indexing is edges[child_label, parent_label].
    """

    start: int
    end: int
    edges: torch.Tensor

    def numpify(self):
        return NumpifiedDecodedEdges(self.start, self.end, var_to_numpy(self.edges))


class NumpifiedDecodedEdges(NamedTuple):
    start: int
    end: int
    edges: np.ndarray


def convert_compositional_targets(targets, label_to_index: Dict[str, int]):
    """
    Return a dictionary of targets for each tree.
    Dictionary is from span indecies to Span class.
    Span class label will be a k-hot vector over all labels.
    This allows us to have multiple gold standard labels.

    Args:
        targets: either a List[Tree] or a tuple where the first entry is List[Tree].
        label_to_index (Dict[str, int]): Mapping from non-terminal labels to indices.

    Returns:
        A list of Dicts, one for each target Tree.
        The Dict maps (start index, end index) to a Span object whose label
        is a k-hot binary Tensor for the labels.
    """
    all_spans = []
    if not isinstance(targets[0], Tree):
        # Multiple target values, but the first is always the list of Trees
        targets = targets[0]
    for tree in targets:
        # Mapping (start, end) to k-hot vector of the non-terminal labels
        dictionary: Dict[Tuple[int, int], List[int]] = {}
        _traverse_tree(tree.root.children[0], dictionary, label_to_index)
        all_spans.append(
            {
                (start, end): Span(start, end, torch.tensor(value))
                for (start, end), value in dictionary.items()
            }
        )
    return all_spans


def _traverse_tree(root, dictionary, label_to_index):
    # take the start and end of the span
    start, end = root.get_token_span()
    label = root.label

    label_index = label_to_index[label]
    if (start, end) in dictionary:
        dictionary[start, end][label_index] = 1
    else:
        target = [0] * len(label_to_index)
        target[label_index] = 1
        dictionary[start, end] = target
    if root.children:
        for node in root.children:
            if not isinstance(node, Token):
                _traverse_tree(node, dictionary, label_to_index)


class ProtoNode:
    """
    Used to construct a (sub)tree in the decoder.

    Since the actual string tokens are not passed to the decoder,
    it cannot construct the actual annotation Node.
    The method to_actual_node can be used to construct the Node.
    """

    __slots__ = ["start", "end", "label", "children"]

    def __init__(self, start: int, end: int, label: str):
        self.start: int = start
        self.end: int = end
        self.label: str = label
        self.children: List["ProtoNode"] = []

    def __hash__(self):
        return hash((self.start, self.end, self.label))

    def __repr__(self):
        result = ["{}<{}:{}>".format(self.label, self.start, self.end)]
        result.extend(repr(child) for child in self.children)
        return "[" + " ".join(result) + "]"

    def to_actual_node(self, tokens: List[str]) -> Tree:
        """
        Converts the ProtoNode into an actual Tree node.

        Args:
            tokens (List[str]): The tokens

        Returns:
            Tree
        """
        node = Tree(self.label)
        node.start = self.start
        node.end = self.end
        children_rev = sorted(self.children, key=lambda x: x.start, reverse=True)
        i = self.start
        while i < self.end:
            if not children_rev or i < children_rev[-1].start:
                # Token
                token = tokens[i]
                child_node = Token(token, i)
                i += 1
            else:
                # Tree
                proto_child = children_rev.pop()
                child_node = proto_child.to_actual_node(tokens)
                i = proto_child.end
            node.children.append(child_node)
        return node

    @classmethod
    def from_actual_node(cls, node: Tree) -> "ProtoNode":
        """
        Converts the Node object into a ProtoNode.
        """
        start, end = node.range
        proto_node = cls(start, end, node.label)
        for child in node.children:
            if not isinstance(child, Token):
                child_proto_node = cls.from_actual_node(child)
                proto_node.children.append(child_proto_node)
        return proto_node

    @classmethod
    def distance(cls, pred_node: "ProtoNode", gold_node: "ProtoNode") -> float:
        """
        Compute the distance between two trees given their root ProtoNodes.

        Currently returns 0 if the trees are the same and 1 otherwise.
        """
        if (
            pred_node.start != gold_node.start
            or pred_node.end != gold_node.end
            or pred_node.label != gold_node.label
            or len(pred_node.children) != len(gold_node.children)
        ):
            return 1.0
        for pred_child, gold_child in zip(pred_node.children, gold_node.children):
            if cls.distance(pred_child, gold_child) == 1.0:
                return 1.0
        return 0.0

    def to_chains(self) -> Dict[Tuple[int, int], Tuple[str, ...]]:
        """
        Returns a flat mapping from (start, end) to the unary chain covering the span.
        """
        proto_node_stack: List[Tuple[ProtoNode, List[str]]] = [(self, [])]
        chains: Dict[Tuple[int, int], Tuple[str, ...]] = {}
        while proto_node_stack:
            node, chain_so_far = proto_node_stack.pop()
            # If it is still in a chain, push it back to the stack
            if (
                len(node.children) == 1
                and node.children[0].start == node.start
                and node.children[0].end == node.end
            ):
                proto_node_stack.append((node.children[0], chain_so_far + [node.label]))
            else:
                chains[node.start, node.end] = tuple(chain_so_far + [node.label])
                for child in node.children:
                    proto_node_stack.append((child, []))
        return chains

    @classmethod
    def hamming_distance(cls, pred_node: "ProtoNode", gold_node: "ProtoNode") -> float:
        """
        Computes the hamming distance between the two trees.

        The hamming distance is the number of (start, end) spans
        where the two trees do not agree on the non-terminal labels.
        """
        pred_chains = pred_node.to_chains()
        gold_chains = gold_node.to_chains()
        keys = set(pred_chains) | set(gold_chains)
        return len(
            [key for key in keys if pred_chains.get(key) != gold_chains.get(key)]
        )


class TreeParserOutput(NamedTuple):
    """
    The data structure that the TreeDecoder returns.

    Each attribute is a list of length batch_size.

    Attributes:
        batch_span_scores (List[Any]): The output of TreeDecoder._score_spans.
            The default implementation computes the scores for each span
            (type = List[List[DecodedSpan]]).
        batch_parses (List[ProtoNode]): The predicted trees.
        batch_predicted_scores (List[torch.Tensor]): Scores of the predicted trees.
        batch_golds (List[ProtoNode]): The gold trees from the annotation.
        batch_golden_scores (List[torch.Tensor]): Scores of the gold trees.
        batch_tree_equality (List[bool]): Whether the predicted trees are correct.
        batch_tree_distances (List[float]): The distance between each pair of
            predicted and gold trees. When the trees are identical, the distance
            should be 0.
    """

    batch_span_scores: List[Any]
    batch_parses: List[ProtoNode]
    batch_predicted_scores: List[torch.Tensor]
    batch_golds: List[ProtoNode]
    batch_golden_scores: List[torch.Tensor]
    batch_tree_equality: List[bool]
    batch_tree_distances: List[float]


class Tag(object):
    INTENT = "IN"
    SLOT = "SL"

    @staticmethod
    def of(label: str):
        if label.startswith(INTENT):
            return Tag.INTENT
        elif label.startswith(SLOT):
            return Tag.SLOT
        return None


def load_glove(filename_prefix, meta, weight):
    print('Loading word vectors from {}'.format(filename_prefix))
    loaded_words = []
    vectors = np.load(filename_prefix + '-vectors.npy')
    with open(filename_prefix + '-vocab.txt') as fin:
        for i, line in enumerate(fin):
            token = line.rstrip('\n')
            if token in meta.vocab_x:
                weight[meta.vocab_x[token]].copy_(torch.tensor(vectors[i]))
                loaded_words.append(token)
    print('GloVe: {} loaded words: {}'.format(
        len(loaded_words), sorted(loaded_words),
    ))
    missing = set(meta.vocab) - set(loaded_words)
    print('GloVe: {} missing words: {}'.format(
        len(missing), sorted(missing)
    ))
