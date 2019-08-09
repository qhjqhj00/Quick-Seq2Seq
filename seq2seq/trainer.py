from pathlib import Path
from typing import Union

import random
import logging
from torch.optim.adam import Adam
import torch
from seq2seq.data import Seq2seqCorpus
from seq2seq.utils import log_line, init_log

from torch.optim import Optimizer


log = logging.getLogger('seq2seq')


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
              max_epochs: int = 100,
              test_mode: bool = False,
              **kwargs
              ):

        if type(base_path) is str:
            base_path = Path(base_path)

        init_log(log, base_path)

        optimizer = self.optimizer(self.model.parameters(), **kwargs)

        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)

        train_data = self.corpus.train
        test_data = self.corpus.test

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

                train_loss /= len(train_data)

                self.model.eval()
                eval_loss = self.evaluate(test_data)

                log_line(log)
                log.info(f'EPOCH {epoch + 1} done: loss {train_loss:.4f}'
                         f' - test_loss {eval_loss:.4f} - lr {learning_rate:.4f}')

            self.model.save(base_path / 'final-model.pt')

        except KeyboardInterrupt:
            log_line(log)
            log.info('Exiting from training early.')
            log.info('Saving model ...')
            self.model.save(base_path / 'final-model.pt')
            log.info('Done.')

    def evaluate(self, test_data, eval_batch_size: int = 32):
        eval_loss = 0
        batches = [test_data[x:x + eval_batch_size] for x in range(0, len(test_data), eval_batch_size)]
        with torch.no_grad():
            for i, batch in enumerate(batches):

                loss = self.model.forward_loss(batch, teacher_forcing_ratio=0)  # turn off teacher forcing

                eval_loss += loss.item()
        return eval_loss


