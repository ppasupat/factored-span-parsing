"""Factored span-based parsing model."""
from __future__ import (absolute_import, division, print_function)

import torch
import torch.nn as nn

from spanparser.model.base import Model


class SpanModel(Model):

    def __init__(self, config, meta):
        super(SpanModel, self).__init__(config, meta)

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
