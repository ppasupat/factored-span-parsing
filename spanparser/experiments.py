from __future__ import (absolute_import, division, print_function)
import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from spanparser.data import create_dataset
from spanparser.metadata import Metadata
from spanparser.model import create_model
from spanparser.utils import Stats, try_gpu


class Experiment(object):

    def __init__(self, config, outputter, load_prefix=None, seed=None, force_cpu=False):
        self.config = config
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.outputter = outputter
        self.meta = Metadata(config)
        if load_prefix:
            self.load_metadata(load_prefix)
        self.dataset = create_dataset(self.config, self.meta)
        self.create_model()
        if load_prefix:
            self.load_model(load_prefix, force_cpu=force_cpu)
        else:
            self.model.initialize(self.config, self.meta)

    def close(self):
        pass

    def create_model(self):
        config = self.config
        self.model = create_model(config, self.meta)
        self.model = try_gpu(self.model)
        self.optimizer = optim.Adam(self.model.parameters(),
                lr=config.train.learning_rate,
                weight_decay=config.train.l2_reg)

    def load_metadata(self, prefix):
        print('Loading metadata from {}.meta'.format(prefix))
        self.meta.load(prefix + '.meta')

    def load_model(self, prefix, force_cpu=False):
        print('Loading model from {}.model'.format(prefix))
        if force_cpu:
            state_dict = torch.load(prefix + '.model', map_location='cpu')
        else:
            state_dict = torch.load(prefix + '.model')
        self.model.load_state_dict(state_dict)

    ################################
    # Train loop

    def train(self):
        config = self.config

        # Initial save
        self.outputter.save_model(self.meta.step, self.model, self.meta)

        max_steps = config.timing.max_steps

        train_iter = None
        train_stats = Stats()

        while self.meta.step < max_steps:
            self.meta.step += 1
            
            train_batch = None if train_iter is None else next(train_iter, None)
            if train_batch is None:
                self.dataset.init_iter('train')
                train_iter = self.dataset.get_iter('train')
                train_batch = next(train_iter)
                assert train_batch is not None, 'No training data'

            stats = self.process_batch(train_batch, train=True)
            train_stats.add(stats)

            # Log the aggregate statistics
            if self.meta.step % config.timing.log_freq == 0:
                print('TRAIN @ {}: {}'.format(self.meta.step, train_stats))
                train_stats.log(self.outputter.tb_logger, self.meta.step, 'pn_train_')
                train_stats = Stats()

            # Save the model
            if self.meta.step % config.timing.save_freq == 0:
                self.outputter.save_model(self.meta.step, self.model, self.meta)

            # Evaluate
            if self.meta.step % config.timing.eval_freq == 0:
                dev_stats = Stats()
                self.dataset.init_iter('dev')
                fout_filename = 'pred.dev.{}'.format(self.meta.step)
                with open(self.outputter.get_path(fout_filename), 'w') as fout:
                    for dev_batch in self.dataset.get_iter('dev'):
                        stats = self.process_batch(dev_batch, train=False, fout=fout)
                        dev_stats.add(stats)
                print('DEV @ {}: {}'.format(self.meta.step, dev_stats))
                dev_stats.log(self.outputter.tb_logger, self.meta.step, 'pn_dev_')

    def test(self):
        test_stats = Stats()
        self.dataset.init_iter('test')
        fout_filename = 'pred.test.{}'.format(self.meta.step)
        with open(self.outputter.get_path(fout_filename), 'w') as fout:
            for test_batch in self.dataset.get_iter('test'):
                stats = self.process_batch(test_batch, train=False, fout=fout)
                test_stats.add(stats)
        print('TEST @ {}: {}'.format(self.meta.step, test_stats))
        test_stats.log(self.outputter.tb_logger, self.meta.step, 'pn_test_')

    ################################
    # Processing a batch

    def process_batch(self, batch, train=False, fout=None):
        """
        Process a batch of examples.

        Args:
            batch (list[???])
            train (bool): Whether it is training or testing
            fout (file): Dump predictions to this file
        Returns:
            a Stats containing the model's statistics
        """
        stats = Stats()
        # Initialize the model
        if train:
            self.optimizer.zero_grad()
            self.model.train()
        else:
            self.model.eval()
        # Forward pass
        logit = self.model(batch)
        loss = self.model.get_loss(logit, batch)
        mean_loss = loss / len(batch)
        stats.n = len(batch)
        stats.loss = float(mean_loss)
        # Evaluate
        prediction = self.model.get_pred(logit, batch)
        self.dataset.evaluate(batch, logit, prediction, stats, fout)
        # Gradient
        if train and mean_loss.requires_grad:
            mean_loss.backward()
            stats.grad_norm = clip_grad_norm_(
                self.model.parameters(),
                self.config.train.gradient_clip
            )
            self.optimizer.step()
        return stats

    ################################
    # Server mode

    def serve(self, port):
        from spanparser.server import start_server
        self.model.eval()
        start_server(self, port)
