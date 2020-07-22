import torch
import torch.nn as nn


from spanparser.model.input_embedder import InputEmbedder
from spanparser.model.cky_with_node_scores_decoder import CKYWithNodeScoresDecoder
from spanparser.model.cky_with_edge_scores_decoder import CKYWithEdgeScoresDecoder
from spanparser.model.tree_output_layer import TreeOutputLayer
from spanparser.model.utils import load_glove


class SpanModel(nn.Module):

    def __init__(self, config, meta):
        super().__init__()
        self.input_embedder = InputEmbedder(config, meta)
        if config.model.decoder.name == 'node':
            self.decoder = CKYWithNodeScoresDecoder(
                config, meta, self.input_embedder.span_embedding_dim
            )
        elif config.model.decoder.name == 'edge':
            self.decoder = CKYWithEdgeScoresDecoder(
                config, meta, self.input_embedder.span_embedding_dim
            )
        else:
            raise ValueError('Unknown decoder: {}'.format(config.model.decoder.name))
        self.output_layer = TreeOutputLayer(config, meta)

    def initialize(self, config, meta):
        # GloVe
        c_embedding = config.model.input_embedder.embedding
        if 'glove' in c_embedding:
            load_glove(
                c_embedding.glove,
                meta,
                self.input_embedder.token_embedder.weight.data
            )

    def forward(self, batch):
        """
        Return TreeParserOutput
        """
        input_representation = self.input_embedder(batch)
        tree_parser_output = self.decoder(batch, *input_representation)
        return tree_parser_output

    def get_loss(self, logit, batch):
        """
        Args:
            logit: TreeOutputLayer
            batch: list[Example]
        """
        return self.output_layer.get_loss(logit, batch)

    def get_pred(self, logit, batch):
        """
        Args:
            logit: TreeOutputLayer
            batch: list[Example]
        """
        return self.output_layer.get_pred(logit, batch)
