import warnings
import logging
from pathlib import Path

import torch
from lensnlp.utils.data import Dictionary, Sentence, Label, Token, SentenceSrc, Seq2seqCorpus
from lensnlp.hyper_parameters import Parameter, device
import os

import random

from typing import List, Union

log = logging.getLogger('lensnlp')

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(input_dim, emb_dim)

        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(torch.nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(output_dim, emb_dim)

        self.rnn = torch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.out = torch.nn.Linear(hid_dim, output_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(torch.nn.Module):

    def __init__(self,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 src_dict: Dictionary,
                 trg_dict: Dictionary,
                 ):

        super(Seq2Seq, self).__init__()

        self.src_dict = src_dict.item2idx
        self.trg_dict = trg_dict.item2idx

        self.trg_vocab = trg_dict.idx2item

        self.PAD_IDX = self.src_dict['_PAD_'.encode('utf-8')]

        self.encoder = encoder
        self.decoder = decoder
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)

        self.start_embed = '<sos>'

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

        self.apply(self.init_weights)

        self.to(device)

    @staticmethod
    def init_weights(m):
        for name, param in m.named_parameters():
            torch.nn.init.uniform_(param.data, -0.08, 0.08)

    def sentences_to_idx(self, sentences: List[Sentence], map_dict) -> torch.LongTensor:
        sentences_length = [len(sent) for sent in sentences]
        max_length = max(sentences_length)
        sentences_idx_tensor = torch.zeros(
            [len(sentences), max_length], dtype=torch.int64, device=device).fill_(self.PAD_IDX)
        for i, sent_zip in enumerate(zip(sentences, sentences_length)):
            sent, length = sent_zip
            sent_idx = [map_dict[t.text.encode('utf-8')] for t in sent]
            sentences_idx_tensor[i][:length] = torch.LongTensor(sent_idx)
        sentences_idx_tensor = torch.einsum('ij->ji', sentences_idx_tensor)
        return sentences_idx_tensor

    def forward(self, src_idx_tensor, trg_idx_tensor, teacher_forcing_ratio=0.5):

        batch_size = trg_idx_tensor.shape[1]

        if trg_idx_tensor.shape[1] != 1:
            max_len = trg_idx_tensor.shape[0]
        else:
            max_len = src_idx_tensor.shape[0] + 5

        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src_idx_tensor)

        # first input to the decoder is the <sos> tokens
        trg_input = trg_idx_tensor[0, :]

        for t in range(1, max_len):

            output, hidden, cell = self.decoder(trg_input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            trg_input = (trg_idx_tensor[t] if teacher_force else top1)

        outputs = torch.einsum('ijk->jki', outputs[1:])

        return outputs

    def forward_loss(self, sentences: Union[SentenceSrc, List[SentenceSrc]]):
        if isinstance(sentences, SentenceSrc):
            sentences = [SentenceSrc]

        trg = [sent.trg for sent in sentences]
        trg_idx_tensor = self.sentences_to_idx(trg, self.trg_dict)

        src = [sent.src for sent in sentences]
        src_idx_tensor = self.sentences_to_idx(src, self.src_dict)

        outputs = self.forward(src_idx_tensor, trg_idx_tensor)
        loss = self.loss_function(outputs, torch.einsum('ij->ji', trg_idx_tensor[1:, :]))
        return loss

    def save(self, model_file: Union[str, Path]):
        """
        存模型
        """
        model_state = {
            'state_dict': self.state_dict(),
            'src_dict': self.src_dict,
            'trg_dict': self.trg_dict,
            'encoder': self.encoder,
            'decoder': self.decoder
        }

        torch.save(model_state, str(model_file), pickle_protocol=4)

    def predict(self, sentences: Union[List[SentenceSrc], SentenceSrc], mini_batch_size: int = 32):
        """
        预测
        输入为 Sentence 数量不限
        返回 Sentence，标签存入对应的位置
        mini_batch_size为每个batch预测的数量
        """
        if isinstance(sentences, SentenceSrc):
            sentences = [sentences]

        with torch.no_grad():
            batches = [sentences[x:x + mini_batch_size] for x in range(0, len(sentences), mini_batch_size)]
            for batch in batches:
                outputs = self.forward(batch, teacher_forcing_ratio=0)
                predicted_seq = torch.argmax(outputs, dim=1)
                for i, sent in enumerate(batch):
                    for idx in predicted_seq[i]:
                        if idx != self.PAD_IDX:
                            sent.trg.add_token(Token(self.trg_vocab[idx]))
        return sentences

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning('Ignore {} sentence(s) with no tokens.'.format(len(sentences) - len(filtered_sentences)))
        return filtered_sentences

    @classmethod
    def load_from_file(cls, model_file: Union[str, Path]):
        """
        :param model_file: 模型地址
        :return: 加载好的模型
        """
        state = Seq2Seq._load_state(model_file)

        model = Seq2Seq(
            encoder=state['encoder'],
            decoder=state['decoder'],
            src_dict=state['src_dict'],
            trg_dict=state['trg_dict']
        )

        model.load_state_dict(state['state_dict'])
        model.eval()
        model.to(device)

        return model

    @classmethod
    def _load_state(cls, model_file: Union[str, Path]):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            state = torch.load(str(model_file), map_location=device)
            return state

    @staticmethod
    def load(model_file: str):
        if model_file == 'seq2seq':
            classifier: Seq2Seq = Seq2Seq.load_from_file(Path(CACHE_ROOT) / 'seq2seq/best-mdoel.pt')
        else:
            try:
                classifier: Seq2Seq = Seq2Seq.load_from_file(Path(CACHE_ROOT) / model_file)
            except NameError('specify a model!'):
                raise
        return classifier