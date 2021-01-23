import torch
import torch.nn as nn


class SpanModel(nn.Module):

    def __init__(self, config, meta):
        super().__init__()
        if config.model.input_embedder.name == 'lstm':
            from spanparser.model.input_embedder import InputEmbedder
            self.input_embedder = InputEmbedder(config, meta)
        elif config.model.input_embedder.name == 'bert':
            from spanparser.model.input_embedder_bert import InputEmbedderBert
            self.input_embedder = InputEmbedderBert(config, meta)
        else:
            raise ValueError('Unknown input embedder: {}'.format(config.model.input_embedder.name))
        if config.model.decoder.name == 'node':
            from spanparser.model.cky_with_node_scores_decoder import CKYWithNodeScoresDecoder
            self.decoder = CKYWithNodeScoresDecoder(
                config, meta, self.input_embedder.span_embedding_dim
            )
        elif config.model.decoder.name == 'edge':
            from spanparser.model.cky_with_edge_scores_decoder import CKYWithEdgeScoresDecoder
            self.decoder = CKYWithEdgeScoresDecoder(
                config, meta, self.input_embedder.span_embedding_dim
            )
        else:
            raise ValueError('Unknown decoder: {}'.format(config.model.decoder.name))
        from spanparser.model.tree_output_layer import TreeOutputLayer
        self.output_layer = TreeOutputLayer(config, meta)

    def initialize(self, config, meta):
        # GloVe
        if 'embedding' in config.model.input_embedder:
          c_embedding = config.model.input_embedder.embedding
          if 'glove' in c_embedding:
              from spanparser.model.utils import load_glove
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
