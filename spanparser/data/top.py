"""TOP Dataset."""
from __future__ import (absolute_import, division, print_function)
from collections import namedtuple, Counter

import numpy as np

from spanparser.data.base import Dataset
from spanparser.utils import batch_iter, INTENT, SLOT, PAD, UNK, SOS, EOS


class Token(namedtuple('Token', ('text', 'index'))):

    def __str__(self):
        return self.text


class Tree(object):

    def __init__(self, label=None, children=None):
        self.label = label
        if not children:
            self.children = []
        else:
            self.children = children[:]
        self.start = None
        self.end = None

    @property
    def range(self):
        return (self.start, self.end)

    @classmethod
    def from_seqlogical(cls, seqlogical):
        tokens = seqlogical.split()
        root = None
        stack = []
        index = 0
        for x in tokens:
            if x[0] == '[':
                subtree = cls(x[1:])
                subtree.start = index
                if stack:
                    stack[-1].children.append(subtree)
                stack.append(subtree)
            elif x[0] == ']':
                subtree = stack.pop()
                subtree.end = index
                if not stack:
                    assert not root
                    root = subtree
            else:
                stack[-1].children.append(Token(x, index))
                index += 1
        assert root is not None
        return root

    def __str__(self):
        stuff = ['[' + self.label]
        for x in self.children:
            stuff.append(str(x))
        stuff.append(']')
        return ' '.join(stuff)

    __repr__ = __str__


class Example(object):
    """
    Should provide the following:
    - index
    - orig_tokens (before UNKification)
    - tokens (after UNKification)
    - token_indices
    - length
    - tree
    - range_to_labels ((s, e) --> list of label strings)
    - range_to_chain_index ((s, e) --> chain index)
    """

    def __init__(self, index, tree):
        self.index = index
        self.tree = tree
        self._extract_tree_info()

    def _extract_tree_info(self):
        stack = [self.tree]
        self.range_to_labels = {}
        self.orig_tokens = []
        while stack:
            node = stack.pop()
            if isinstance(node, Token):
                self.orig_tokens.append(node.text)
                continue
            self.range_to_labels.setdefault(node.range, []).append(node.label)
            stack.extend(reversed(node.children))
        self.tokens = self.orig_tokens[:]

    def preprocess(self, config):
        """
        Happens before metadata is populated.
        """
        if 'lower' in config.data and config.data.lower:
            self.tokens = [x.lower() for x in self.tokens]

    def postprocess(self, config, meta):
        """
        Happens after metadata is populated.
        """
        self.token_indices = [meta.vocab_x.get(x, 0) for x in self.tokens]
        self.range_to_chain_index = {}
        for (s, e), labels in self.range_to_labels.items():
            chain = tuple(meta.nt_x.get(x, -1) for x in labels)
            self.range_to_chain_index[s, e] = meta.unary_chains_x.get(chain, -1)

    def __str__(self):
        return 'Example({} -> {})'.format(' '.join(self.tokens), self.tree)

    __repr__ = __str__

    @property
    def length(self):
        return len(self.tokens)


class TopDataset(Dataset):

    def __init__(self, config, meta):
        super().__init__(config, meta)
        self._data = {}
        self._iters = {}
        for name in config.data.files:
            self._data[name] = self._read_data(config.data.files[name])
            print('Read {} {} examples'.format(len(self._data[name]), name))
        for name in self._data:
            for example in self._data[name]:
                example.preprocess(config)
        if not hasattr(meta, 'vocab'):
            self._gen_vocab(self._data['train'], config, meta)
        for name in self._data:
            for example in self._data[name]:
                example.postprocess(config, meta)

    def _read_data(self, filename):
        examples = []
        with open(filename) as fin:
            for index, line in enumerate(fin):
                tree = Tree.from_seqlogical(line.strip())
                example = Example(index, tree)
                examples.append(example)
        return examples

    def _gen_vocab(self, examples, config, meta):
        """
        Add the following keys to meta:
        vocab: Sorted list of tokens in training data, including PAD, UNK, SOS, EOS.
        vocab_x: Reverse of vocab (token -> index).
        vocab_cnt: Map from each token to frequency.
        nt: Sorted list of non-terminal labels.
        nt_x: Reverse of nt (non-terminal -> index).
        nt_groups: A mapping from Tag (INTENT or SLOT) to the indices of
            non-terminals with that tag.
        unary_chains: Canonical list of unary chains from training data.
            Each chain is represented as a tuple of non-terminal indices.
            Chains of length 1 are also included.
        unary_chains_x: Reverse of unary_chains (unary_chain -> index)
        unary_chain_groups: Chains from unary_chains categorized based on the tags
            (INTENT or SLOT) of the top and bottom non-terminals of the chains.
            It maps (a, b) -> chains whose top tag = a and bottom tag = b.
        """
        meta.vocab_cnt = Counter()
        meta.nt = set()
        unary_chains = set()
        for example in examples:
            for x in example.tokens:
                meta.vocab_cnt[x] += 1
            for labels in example.range_to_labels.values():
                meta.nt.update(labels)
                unary_chains.add(tuple(labels))
        # Finalize
        meta.vocab_cnt = dict(meta.vocab_cnt)
        if 'min_freq' in config.data:
            meta.vocab_cnt = {
                x: v for (x, v) in meta.vocab_cnt.items()
                if v >= config.data.min_freq
            }
        meta.vocab = [PAD, UNK, SOS, EOS] + sorted(meta.vocab_cnt)
        meta.vocab_x = {x: i for (i, x) in enumerate(meta.vocab)}
        meta.nt = sorted(meta.nt)
        meta.nt_x = {x: i for (i, x) in enumerate(meta.nt)}
        meta.nt_groups = {
            INTENT: [i for (i, x) in enumerate(meta.nt) if x.startswith(INTENT)],
            SLOT: [i for (i, x) in enumerate(meta.nt) if x.startswith(SLOT)],
        }
        # Unary chains
        meta.unary_chain_groups = {
            (INTENT, INTENT): [],
            (INTENT, SLOT): [],
            (SLOT, INTENT): [],
            (SLOT, SLOT): [],
        }
        for chain in unary_chains:
            top_type = chain[0][:2]
            bot_type = chain[-1][:2]
            meta.unary_chain_groups[top_type, bot_type].append(
                tuple(meta.nt_x[x] for x in chain)
            )
        meta.unary_chains = []
        for key in meta.unary_chain_groups:
            meta.unary_chain_groups[key].sort()
            meta.unary_chains += meta.unary_chain_groups[key]
        meta.unary_chains_x = {x: i for (i, x) in enumerate(meta.unary_chains)}
        # Done!
        print('Vocab size: {}'.format(len(meta.vocab)))
        print('Num labels: {}'.format(len(meta.nt)))
        print('Num chains: {}'.format(len(meta.unary_chains)))
        meta.save_keys.update([
            'vocab', 'vocab_x', 'vocab_cnt',
            'nt', 'nt_x', 'nt_groups',
            'unary_chains', 'unary_chains_x', 'unary_chain_groups',
        ])

    def init_iter(self, name):
        """
        Initialize the iterator for the specified data split.
        """
        data = self._data[name][:]
        if name == 'train':
            np.random.shuffle(data)
        print(data[:5])
        self._iters[name] = batch_iter(
            data, self.batch_size, (lambda x: -x.length)
        )

    def get_iter(self, name):
        """
        Get the iterator over the specified data split.
        """
        return self._iters[name]

    def evaluate(self, batch, logits, prediction, stats, fout=None):
        """
        Evaluate the predictions and write the results to stats.

        Args:
            batch: list[Example]
            prediction: Tuple[List[Tree], Tensor]
                The tensor contains prediction scores
            stats: Stats
        """
        pred_trees, pred_scores = prediction
        ordered = sorted(
            zip(batch, pred_trees, pred_scores),
            key=lambda x: x[0].index
        )
        for example, pred_tree, pred_score in ordered:
            if fout:
                print(pred_tree, file=fout)
            if str(example.tree) == str(pred_tree):
                stats.accuracy += 1
