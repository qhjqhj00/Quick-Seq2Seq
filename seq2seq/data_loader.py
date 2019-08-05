from seq2seq.data import SentenceSrc, Sentence, Seq2seqCorpus
from typing import List
import random
from tqdm import tqdm


def read_seq2seq_data(path):
    data = []
    lines = [line.strip().split('\t') for line in open(path, encoding="utf-8")]
    src_tokenizer = None
    trg_tokenizer = None
    for line in tqdm(lines, desc="Loading data"):
        if src_tokenizer is None and trg_tokenizer is None:
            src_sentence = Sentence(line[0])
            src_tokenizer = src_sentence.Tokenizer
            trg_sentence = Sentence(line[-1])
            trg_tokenizer = trg_sentence.Tokenizer
        else:
            src_sentence = Sentence(line[0], tokenizer=src_tokenizer)
            trg_sentence = Sentence(line[-1], tokenizer=trg_tokenizer)

        sentence = SentenceSrc(src_sentence, trg_sentence)
        data.append(sentence)
    return data


def load_seq2seq_data(train_file, test_rate: float = 0.2, test_file=None):
    train_ = read_seq2seq_data(train_file)
    random.shuffle(train_)
    if test_file is not None:
        test_ = read_seq2seq_data(test_file)
    else:
        sample_amount = len(train_)
        test_: List[SentenceSrc] = train_[:int(test_rate*sample_amount)]
        train_ = train_[int(test_rate*sample_amount):]
    seq_corpus = Seq2seqCorpus(train_, test_)
    return seq_corpus
