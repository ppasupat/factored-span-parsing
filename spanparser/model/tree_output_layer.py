from torch import nn

from spanparser.model.span_score_loss import SpanScoreLoss
from spanparser.model.utils import TreeParserOutput


class TreeOutputLayer(nn.Module):
    """
    Takes the TreeParserOutput from the TreeDecoder and outputs the actual Tree.
    """

    def __init__(self, config, meta):
        super().__init__()
        self.loss_fn = SpanScoreLoss(config, meta)
        self.verbose = config.model.output_layer.verbose

    def forward(self, logit, batch):
        return self.loss_fn(logit, batch)

    def get_loss(self, logit, batch):
        return self.loss_fn(logit, batch)

    def get_pred(self, logit, batch):
        """
        Builds Tree objects with the actual utterance tokens.
        """
        if self.verbose:
            self._verbose_print(logit, batch)
        parse_trees = []
        for i, parse in enumerate(logit.batch_parses):
            converted = parse.to_actual_node(batch[i].orig_tokens)
            parse_trees.append(converted)
        return parse_trees, logit.batch_predicted_scores

    def _verbose_print(self, logit, batch):
        assert isinstance(logit, TreeParserOutput)
        print("#" * 40)
        for i in range(len(logit.batch_parses)):
            print(f"----- {i} -----")
            print(batch[i].tokens)
            print(
                "GOLD   ({:9.5f}) {}".format(
                    logit.batch_golden_scores[i], logit.batch_golds[i]
                )
            )
            print(
                "PRED {} ({:9.5f}) {}".format(
                    "o" if logit.batch_tree_equality[i] else "x",
                    logit.batch_predicted_scores[i],
                    logit.batch_parses[i],
                )
            )
