import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from tqdm import tqdm

from spanparser.data import create_dataset
from spanparser.metadata import Metadata
from spanparser.model import create_model
from spanparser.utils import Stats, try_gpu


class Experiment(object):

    def __init__(self, config, outputter, load_prefix=None, seed=None):
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
            self.load_model(load_prefix)
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

    def load_model(self, prefix):
        print('Loading model from {}.model'.format(prefix))
        state_dict = torch.load(prefix + '.model')
        self.model.load_state_dict(state_dict)

    ################################
    # Train loop

    def train(self):
        config = self.config
        max_epochs = config.timing.max_epochs
        save_every = config.timing.save_every
        eval_every = config.timing.eval_every

        train_stats = Stats()

        # Initial save
        if save_every > 0:
            self.outputter.save_model(self.meta.epoch, self.model, self.meta)

        progress_bar = tqdm(total=max_epochs, desc='TRAIN')

        while self.meta.epoch < max_epochs:
            self.meta.epoch += 1
            progress_bar.update()
            
            self.dataset.init_iter('train')
            for train_batch in tqdm(self.dataset.get_iter('train'), desc='TRAIN'):
                stats = self.process_batch(train_batch, train=True)
                train_stats.add(stats)
            print('TRAIN @ {}: {}'.format(self.meta.epoch, train_stats))
            train_stats.log(self.outputter.tb_logger, self.meta.epoch, 'pn_train_')
            train_stats = Stats()

            if save_every > 0 and self.meta.epoch % save_every == 0:
                # Save the model
                self.outputter.save_model(self.meta.epoch, self.model, self.meta)

            if self.meta.epoch % eval_every == 0:
                # Evaluate
                dev_stats = Stats()
                self.dataset.init_iter('dev')
                fout_filename = 'pred.dev.{}'.format(self.meta.epoch)
                with open(self.outputter.get_path(fout_filename), 'w') as fout:
                    for dev_batch in tqdm(self.dataset.get_iter('dev'), desc='DEV'):
                        stats = self.process_batch(dev_batch, train=False, fout=fout)
                        dev_stats.add(stats)
                print('DEV @ {}: {}'.format(self.meta.epoch, dev_stats))
                dev_stats.log(self.outputter.tb_logger, self.meta.epoch, 'pn_dev_')
                self.meta.update_acc(dev_stats.accuracy / dev_stats.n)

        progress_bar.close()

    def test(self):
        test_stats = Stats()
        self.dataset.init_iter('test')
        fout_filename = 'pred.test.{}'.format(self.meta.epoch)
        with open(self.outputter.get_path(fout_filename), 'w') as fout:
            for test_batch in tqdm(self.dataset.get_iter('test'), desc='TEST'):
                stats = self.process_batch(test_batch, train=False, fout=fout)
                test_stats.add(stats)
        print('TEST @ {}: {}'.format(self.meta.epoch, test_stats))
        test_stats.log(self.outputter.tb_logger, self.meta.epoch, 'pn_test_')


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
        predictions = self.model.get_pred(logit, batch)
        self.dataset.evaluate(batch, predictions, stats, fout)
        # Gradient
        if train and mean_loss.requires_grad:
            mean_loss.backward()
            stats.grad_norm = clip_grad_norm_(
                self.model.parameters(),
                self.config.train.gradient_clip
            )
            self.optimizer.step()
        return stats
