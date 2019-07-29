from pathlib import Path
from typing import List, Union

import random
import logging
from torch.optim.adam import Adam
import torch
import lensnlp.models.nn as nn
from lensnlp.utils.data import Sentence, Corpus, Seq2seqCorpus
from lensnlp.models import RelationExtraction
from lensnlp.utils.training_utils import Metric, clear_embeddings, log_line, init_log, WeightExtractor

from lensnlp.hyper_parameters import Parameter

from torch.optim import Optimizer


log = logging.getLogger('lensnlp')


class Seq2SeqTrainer:

    def __init__(self,
                 model: torch.nn.Module,
                 corpus: Seq2seqCorpus,
                 optimizer: Optimizer = Adam,
                 epoch: int = 0,
                 loss: float = 10000.0,
                 optimizer_state: dict = None,
                 ):

        self.model = model
        self.corpus: Seq2seqCorpus = corpus
        self.optimizer: Optimizer = optimizer
        self.epoch: int = epoch
        self.loss: float = loss
        self.optimizer_state: dict = optimizer_state

    def train(self,
              base_path: Union[Path, str],
              learning_rate: float = 0.001,
              mini_batch_size: int = 32,
              eval_mini_batch_size: int = 20,
              max_epochs: int = 150,
              test_mode: bool = False,
              **kwargs
              ):

        if eval_mini_batch_size is None:
            eval_mini_batch_size = mini_batch_size

        if type(base_path) is str:
            base_path = Path(base_path)

        init_log(log, base_path)

        log_line(log)
        # log.info(f'Evaluation method: Mircro F1-Score')

        weight_extractor = WeightExtractor(base_path)

        optimizer = self.optimizer(self.model.parameters(), **kwargs)

        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)

        train_data = self.corpus.train

        try:

            for epoch in range(0 + self.epoch, max_epochs + self.epoch):
                log_line(log)

                if learning_rate < 0.0001:
                    log_line(log)
                    log.info('learning rate too small - quitting training!')
                    log_line(log)
                    break

                if not test_mode:
                    random.shuffle(train_data)

                batches = [train_data[x:x + mini_batch_size] for x in range(0, len(train_data), mini_batch_size)]

                self.model.train()

                train_loss: float = 0
                seen_sentences = 0
                modulo = max(1, int(len(batches) / 10))

                for batch_no, batch in enumerate(batches):
                    loss = self.model.forward_loss(batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    seen_sentences += len(batch)
                    train_loss += loss.item()

                    if batch_no % modulo == 0:
                        log.info(f'epoch {epoch + 1} - iter {batch_no}/{len(batches)} - loss '
                                 f'{train_loss / seen_sentences:.8f}')
                        iteration = epoch * len(batches) + batch_no
                        weight_extractor.extract_weights(self.model.state_dict(), iteration)

                train_loss /= len(train_data)

                self.model.eval()

                log_line(log)
                log.info(f'EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate:.4f}')

            self.model.save(base_path / 'final-model.pt')

        except KeyboardInterrupt:
            log_line(log)
            log.info('Exiting from training early.')
            log.info('Saving model ...')
            self.model.save(base_path / 'final-model.pt')
            log.info('Done.')


