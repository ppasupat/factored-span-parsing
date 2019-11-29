"""Factored span-based parsing model."""
from __future__ import (absolute_import, division, print_function)

import torch
import torch.nn as nn

from spanparser.model.base import Model
from spanparser.model.span_encoder import SpanEncoder
from spanparser.model.decoder_node_scores_only import DecoderNodeScoresOnly
from spanparser.model.decoder_with_edge_scores import DecoderWithEdgeScores
from spanparser.model.tree_output_layer import TreeOutputLayer
from spanparser.model.utils import load_glove


class SpanModel(Model):

    def __init__(self, config, meta):
        super(SpanModel, self).__init__(config, meta)
        self.span_encoder = SpanEncoder(config, meta)
        if config.model.decoder.name == 'node':
            self.decoder = DecoderNodeScoresOnly(
                config, meta, self.span_encoder.span_embedding_dim)
        elif config.model.decoder.name == 'edge':
            self.decoder = DecoderWithEdgeScores(
                config, meta, self.span_encoder.span_embedding_dim)
        else:
            raise ValueError('Unknown decoder: {}'.format(config.model.decoder.name))
        self.output_layer = TreeOutputLayer(config, meta)

    def initialize(self, config, meta):
        """
        Initialize GloVe.
        """
        c_embedding = config.model.span_encoder.embedding
        if 'glove' in c_embedding:
            load_glove(c_embedding.glove, meta,
                    self.span_encoder.token_embedder.weight.data)

    def forward(self, batch):
        """
        Return TreeParserOutput
        """
        span_features = self.span_encoder(batch)
        tree_parser_output = self.decoder(batch, *span_features)
        return tree_parser_output

    def get_loss(self, logit, batch):
        """
        Args:
            logit: TreeOutputLayer
            batch: list[Example]
        Returns:
            a scalar tensor
        """
        return self.output_layer.get_loss(logit, batch)

    def get_pred(self, logit, batch):
        """
        Args:
            logit: TreeOutputLayer
            batch: list[Example]
        Returns:
            predictions to be read by dataset.evaluate
        """
        return self.output_layer.get_pred(logit, batch)
