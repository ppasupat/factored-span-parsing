"""Factored span-based parsing model."""
from __future__ import (absolute_import, division, print_function)

import torch
import torch.nn as nn

from spanparser.model.span_features_representation import SpanFeaturesRepresentation
from spanparser.model.alternative_representation import AlternativeRepresentation
from spanparser.model.cky_with_node_scores_decoder import CKYWithNodeScoresDecoder
from spanparser.model.cky_with_edge_scores_decoder import CKYWithEdgeScoresDecoder
from spanparser.model.tree_output_layer import TreeOutputLayer
from spanparser.model.utils import load_glove


class SpanModel(nn.Module):

    def __init__(self, config, meta):
        super(Model, self).__init__()

    def initialize(self, config, meta):
        """
        Initialize GloVe.
        """
        pass

    def forward(self, batch):
        """
        Return TreeParserOutput
        """
        raise NotImplementedError

    def get_loss(self, logit, batch):
        """
        Args:
            logit: TreeOutputLayer
            batch: list[Example]
        Returns:
            a scalar tensor
        """
        raise NotImplementedError

    def get_pred(self, logit, batch):
        """
        Args:
            logit: TreeOutputLayer
            batch: list[Example]
        Returns:
            predictions to be read by dataset.evaluate
        """
        raise NotImplementedError
