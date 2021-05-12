from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from spanparser.model.utils import ProtoNode, TreeParserOutput
from spanparser.utils import try_gpu


DUMMY_INTENT = "IN:DUMMY"


class SpanScoreLoss(nn.Module):
    """
    Computes cross-entropy loss on span scores and (optionally) edge scores.
    """

    def __init__(self, config, meta):
        """
        Configs:
            add_edge_loss (bool): Whether to add edge score loss.
            null_weight (float): A multiplier for the loss term when the
                gold class is null (i.e., the span is not in the gold tree).
                Higher null_weight -> fewer spans are predicted.
        """
        super().__init__()
        self.labels = meta.nt
        self.labels_idx = meta.nt_x
        self.unary_chains = meta.unary_chains
        self.chains_idx = meta.unary_chains_x
        # Construct the loss weight; the first class is the NULL class
        self.null_weight = config.model.output_layer.null_weight
        weight = [self.null_weight] + [1.0] * len(self.chains_idx)
        weight = try_gpu(torch.tensor(weight))
        self.node_loss = nn.CrossEntropyLoss(weight=weight)
        if config.model.output_layer.add_edge_loss:
            # Edge scores are already softmax-ed.
            self.edge_loss = nn.NLLLoss()
        else:
            self.edge_loss = None

    def forward(self, logit: TreeParserOutput, batch):
        losses: List[torch.Tensor] = []

        # for each training example
        for i in range(len(logit.batch_span_scores)):
            if self.edge_loss is not None:
                decoded_spans, decoded_edges = logit.batch_span_scores[i]
            else:
                decoded_spans = logit.batch_span_scores[i]
            gold_proto_node = logit.batch_golds[i]
            gold_chains = self.tree_to_chains(gold_proto_node)
            total_loss = try_gpu(torch.zeros([], dtype=torch.float64))

            # Add node losses
            # prediction: (num_spans, num_classes)
            # where num_classes = 1 + len(self.unary_chains)
            # The first column for NULL class is a zero vector.
            prediction = torch.cat(
                [
                    try_gpu(torch.zeros(len(decoded_spans), 1)),
                    torch.stack(
                        [decoded_span.labels for decoded_span in decoded_spans]
                    ),
                ],
                dim=1,
            )
            # targets: (num_spans,)
            # The unary chain indices are shifted by 1 since we padded
            # the prediction with the NULL class at index 0.
            targets = [
                gold_chains.get((decoded_span.start, decoded_span.end), -1) + 1
                for decoded_span in decoded_spans
            ]
            targets = try_gpu(torch.tensor(targets))
            if gold_proto_node.label == DUMMY_INTENT:
                # Only look at the losses of the spans
                dummy_chain_idx = self.chains_idx[(self.labels_idx[DUMMY_INTENT],)]
                good_entries = []
                for i, decoded_span in enumerate(decoded_spans):
                    gold_chain = gold_chains.get((decoded_span.start, decoded_span.end))
                    if (gold_chain is not None and gold_chain != dummy_chain_idx):
                        good_entries.append(i)
                if good_entries:
                    good_entries = try_gpu(torch.tensor(good_entries))
                    prediction = torch.index_select(prediction, 0, good_entries)
                    targets = torch.index_select(targets, 0, good_entries)
                    total_loss = total_loss + self.node_loss(prediction, targets)
            else:
                total_loss = total_loss + self.node_loss(prediction, targets)

            # Add edge losses
            if self.edge_loss is not None and gold_proto_node.label != DUMMY_INTENT:
                span_edges = {(span.start, span.end): span for span in decoded_edges}
                gold_edges = self.tree_to_edges(gold_proto_node)
                if not gold_edges:
                    edge_loss = 0.0
                else:
                    # prediction: (num_gold_edges, num_labels)
                    prediction = []
                    # targets: (num_gold_edges,)
                    targets = []
                    for key, (child_label, parent_label) in gold_edges.items():
                        prediction.append(span_edges[key].edges[child_label])
                        targets.append(parent_label)
                    edge_loss = self.edge_loss(
                        torch.stack(prediction), try_gpu(torch.tensor(targets))
                    )
                total_loss = total_loss + edge_loss

            losses.append(total_loss)

        stacked_losses = torch.stack(losses)
        return stacked_losses.mean()

    def tree_to_chains(self, root_proto_node: ProtoNode) -> Dict[Tuple[int, int], int]:
        """
        Returns a mapping (start, end) -> unary_chain_index.
        """
        proto_node_stack: List[Tuple[ProtoNode, List[int]]] = [(root_proto_node, [])]
        chains = {}
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
                    chains[node.start, node.end] = self.chains_idx[chain]
                for child in node.children:
                    proto_node_stack.append((child, []))
        return chains

    def tree_to_edges(
        self, root_proto_node: ProtoNode
    ) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Returns a mapping (child_start, child_end) -> (child_label, parent_label).
        """
        proto_node_stack = [root_proto_node]
        edges = {}
        while proto_node_stack:
            node = proto_node_stack.pop()
            for child in node.children:
                proto_node_stack.append(child)
                if not (node.start == child.start and node.end == child.end):
                    edges[child.start, child.end] = (
                        self.labels_idx.get(child.label, 0),
                        self.labels_idx.get(node.label, 0),
                    )
        return edges
