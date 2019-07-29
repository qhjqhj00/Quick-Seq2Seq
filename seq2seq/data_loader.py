from seq2seq.data import SentenceSrc, Sentence, Seq2seqCorpus
from typing import List
import random


def read_seq2seq_data(path):
    data = []
    lines = [line.strip().split('\t') for line in open(path, encoding="utf-8")]
    for line in lines:
        src_sentence = Sentence(line[0])
        trg_sentence = Sentence(line[-1])
        sentence = SentenceSrc(src_sentence, trg_sentence)
        data.append(sentence)
    return data


def load_seq2seq_data(train_file, test_file = None):
    train_ = read_seq2seq_data(train_file)
    if test_file is not None:
        test_ = read_seq2seq_data(test_file)
    else:
        test_: List[SentenceSrc] = [train_[i] for i in __sample(len(train_), 0.2)]
        train_ = [sentence for sentence in train_ if sentence not in test_]
    seq_corpus = Seq2seqCorpus(train_, test_)
    return seq_corpus


def __sample(total_number_of_sentences: int, percentage: float = 0.1) -> List[int]:
    """
    :param total_number_of_sentences: 数据集的大小
    :param percentage: 提取比例
    :return: sample的indices
    """
    sample_size: int = round(total_number_of_sentences * percentage)
    sample = random.sample(range(1, total_number_of_sentences), sample_size)
    return sample
