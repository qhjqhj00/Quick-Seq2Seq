# Partly modified from zalandoresearch's flair

from typing import Dict

from pypinyin import pinyin, Style
from collections import Counter

from typing import List
import langid

from .utils import get_sample

import jieba
from segtok.segmenter import split_single
from segtok.tokenizer import split_contractions
from segtok.tokenizer import word_tokenizer
import re


class Dictionary:
    def __init__(self, add_unk=True):
        self.item2idx: Dict[str, int] = {}
        self.idx2item: List[str] = []
        if add_unk:
            self.add_item('<unk>')

    def add_item(self, item: str) -> int:
        item = item.encode('utf-8')
        if item not in self.item2idx:
            self.idx2item.append(item)
            self.item2idx[item] = len(self.idx2item) - 1
        return self.item2idx[item]

    def get_idx_for_item(self, item: str) -> int:
        item = item.encode('utf-8')
        if item in self.item2idx.keys():
            return self.item2idx[item]
        else:
            return 0

    def get_items(self) -> List[str]:
        items = []
        for item in self.idx2item:
            items.append(item.decode('UTF-8'))
        return items

    def __len__(self) -> int:
        return len(self.idx2item)

    def get_item_for_index(self, idx):
        return self.idx2item[idx].decode('UTF-8')

    def save(self, savefile):
        import pickle
        with open(savefile, 'wb') as f:
            mappings = {
                'idx2item': self.idx2item,
                'item2idx': self.item2idx
            }
            pickle.dump(mappings, f)

    @classmethod
    def load(cls, filename: str):
        import pickle
        dictionary: Dictionary = Dictionary()
        with open(filename, 'rb') as f:
            mappings = pickle.load(f, encoding='latin1')
            idx2item = mappings['idx2item']
            item2idx = mappings['item2idx']
            dictionary.item2idx = item2idx
            dictionary.idx2item = idx2item
        return dictionary


class Token:
    def __init__(self,
                 text: str,
                 idx: int = None,
                 whitespace_after: bool = True,
                 start_position: int = None,
                 sp: str = None
                 ):
        self.text: str = text
        if sp == 'lt':
            self.latin = self.uyghur_to_latin()
        if sp == 'py':
            self.pinyin = self.converter()
        self.idx: int = idx
        self.whitespace_after: bool = whitespace_after

        self.start_pos = start_position
        self.end_pos = start_position + len(text) if start_position is not None else None

    def converter(self):
        p = pinyin(self.text, style=Style.TONE2)
        p = [t[0] for t in p]
        return ' '.join(p)

    def uyghur_to_latin(self):
        latin_map = {"ا": "a", "ە": "e", "ى": "i", "ې": "é", "و": "o",
                     "ۇ": "u", "ۆ": "ö", "ۈ": "ü", "ب": "b", "پ": "p", "ت": "t", "ژ": "j",
                     "چ": "ç", "خ": "x", "د": "d", "ر": "r", "ز": "z", "ج": "j", "س": "s",
                     "ش": "ş", "ف": "f", "غ": "ğ", "ق": "q", "ك": "k", "گ": "g", "ڭ": "ñ",
                     "ل": "l", "م": "m", "ن": "n", "ھ": "h", "ي": "y", "ۋ": "w", "ئ": "", ".": ".",
                     "؟": "?", "!": "!", "،": ",", "؛": ";", ":": ":", "«": "«", "»": "»",
                     "-": "-", "—": "—", "(": "(", ")": ")"}
        new = ''
        for c in self.text:
            if c in latin_map:
                new += latin_map[c]
            else:
                new += c
        return new

    @property
    def start_position(self) -> int:
        return self.start_pos

    @property
    def end_position(self) -> int:
        return self.end_pos

    def __str__(self) -> str:
        return 'Token: {} {}'.format(self.idx, self.text) if self.idx is not None else 'Token: {}'.format(self.text)

    def __repr__(self) -> str:
        return 'Token: {} {}'.format(self.idx, self.text) if self.idx is not None else 'Token: {}'.format(self.text)


class Sentence:

    def __init__(self, text: str = None, language_type: str = None, sp_op: str = None,
                 max_length: int = None, tokenizer=None):

        super(Sentence, self).__init__()

        self.tokens: List[Token] = []
        self.language_type = language_type
        self.sp_op = sp_op

        if tokenizer is not None:
            self.Tokenizer = tokenizer
        elif self.language_type is not None:
            self.Tokenizer = Tokenizer(self.language_type, self.sp_op)
        elif text is not None:
            self.Tokenizer = Tokenizer(example=text)
        else:
            self.Tokenizer = None

        if self.Tokenizer is not None:
            tokenized = self.Tokenizer.word_tokenizer(text)

            for token in tokenized:
                self.add_token(token)

        if max_length is not None:
            self.tokens = self.tokens[:max_length]

    def get_token(self, token_id: int) -> Token:
        for token in self.tokens:
            if token.idx == token_id:
                return token

    def add_token(self, token: Token):
        self.tokens.append(token)

        token.sentence = self
        if token.idx is None:
            token.idx = len(self.tokens)

    def to_tokenized_string(self, lang: str = None) -> str:
        if lang == 'ug':
            return ' '.join([t.latin for t in self.tokens])
        elif lang == 'py':
            return ' '.join([t.pinyin for t in self.tokens])
        else:
            return ' '.join([t.text for t in self.tokens])

    def to_plain_string(self, lang: str = None):
        plain = ''
        for token in self.tokens:
            if lang == 'ug':
                plain += token.latin
            elif lang == 'py':
                plain += token.pinyin
            else:
                plain += token.text
            if token.whitespace_after:
                plain += ' '
        return plain.rstrip()

    def infer_space_after(self):
        last_token = None
        quote_count: int = 0

        for token in self.tokens:
            if token.text == '"':
                quote_count += 1
                if quote_count % 2 != 0:
                    token.whitespace_after = False
                elif last_token is not None:
                    last_token.whitespace_after = False

            if last_token is not None:

                if token.text in ['.', ':', ',', ';', ')', 'n\'t', '!', '?', '،', '؛', '؟']:
                    last_token.whitespace_after = False

                if token.text.startswith('\''):
                    last_token.whitespace_after = False

            if token.text in ['(']:
                token.whitespace_after = False

            last_token = token
        return self

    def to_original_text(self) -> str:
        str = ''
        pos = 0
        for t in self.tokens:
            while t.start_pos != pos:
                str += ' '
                pos += 1

            str += t.text
            pos += len(t.text)

        return str

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self):
        return 'Sentence: "{}" - {} Tokens'.format(' '.join([t.text for t in self.tokens]), len(self))

    def __str__(self) -> str:
        return 'Sentence: "{}" - {} Tokens'.format(' '.join([t.text for t in self.tokens]), len(self))

    def __len__(self) -> int:
        return len(self.tokens)


class SentenceSrc:
    def __init__(self, src: Sentence, trg: Sentence = None):
        self.src = src
        self.src.tokens.insert(0, Token('<sos>'))
        self.src.tokens.append(Token('<eos>'))
        if trg is not None:
            self.trg = trg
            self.trg.tokens.insert(0, Token('<sos>'))
            self.trg.tokens.append(Token('<eos>'))
        else:
            self.trg = Sentence()
            token = Token('<sos>')
            self.trg.add_token(token)


class Seq2seqCorpus:
    def __init__(self,
                 train: List[SentenceSrc],
                 test: List[SentenceSrc],
                 name: str = 'seq2seq'):
        self._train: List[SentenceSrc] = train
        self._test: List[SentenceSrc] = test
        self.name: str = name

    @property
    def train(self) -> List[SentenceSrc]:
        return self._train

    @property
    def test(self) -> List[SentenceSrc]:
        return self._test

    def _get_most_common_tokens(self, sentence_type, max_tokens, min_freq) -> List[str]:
        tokens_and_frequencies = Counter(self._get_all_tokens(sentence_type))
        tokens_and_frequencies = tokens_and_frequencies.most_common()

        tokens = []
        for token, freq in tokens_and_frequencies:
            if (min_freq != -1 and freq < min_freq) or (max_tokens != -1 and len(tokens) == max_tokens):
                break
            tokens.append(token)
        return tokens

    def make_vocab_dictionary(self, sentence_type, max_tokens=-1, min_freq=1) -> Dictionary:
        tokens = self._get_most_common_tokens(sentence_type, max_tokens, min_freq)

        vocab_dictionary: Dictionary = Dictionary(add_unk=False)
        vocab_dictionary.add_item('_PAD_')
        for token in tokens:
            vocab_dictionary.add_item(token)
        return vocab_dictionary

    def get_all_sentence(self) -> List[SentenceSrc]:
        all_sentences: List[SentenceSrc] = []
        all_sentences.extend(self.train)
        all_sentences.extend(self.test)
        return all_sentences

    def _get_all_tokens(self, sentence_type) -> List[str]:
        if sentence_type == 'src':
            tokens = list(map((lambda s: s.src.tokens), self.get_all_sentence()))
        elif sentence_type == 'trg':
            tokens = list(map((lambda s: s.trg.tokens), self.get_all_sentence()))
        else:
            raise ValueError
        tokens = [token for sublist in tokens for token in sublist]
        return list(map((lambda t: t.text), tokens))

    @staticmethod
    def sample(rate: float, sentences: List[SentenceSrc]):
        len_size = len(sentences)
        sample_indices = get_sample(len_size, rate)
        sample_data = [sentences[i] for i in sample_indices]
        return sample_data

    def sample_corpus(self, rate):
        self._train = self.sample(len(self.train), rate)
        self._test = self.sample(len(self.test), rate)


class Tokenizer:
    def __init__(self, language_type: str = None, example: str = None, sp_op: str = None):

        if language_type is None and example is None:
            raise ValueError("Must specify language type or provides an example")

        if sp_op is not None and sp_op not in ['char', 'lt', 'py']:
            raise ValueError("Not support the operation yet")

        if language_type is not None:
            self.language_type = language_type
        else:
            self.language_type = langid.classify(example)[0]

        self.sp_op = sp_op

    def word_tokenizer(self, text) -> List[Token]:
        tokenized = []
        if self.language_type == 'zh':
            if self.sp_op == 'char':
                for index, char in enumerate(text):
                    token = Token(char, start_position=index)
                    tokenized.append(token)
            elif self.sp_op == 'py':
                for index, char in enumerate(text):
                    token = Token(char, start_position=index, sp='py')
                    tokenized.append(token)
            else:
                seg_list = list(jieba.tokenize(text))
                for t in seg_list:
                    token = Token(t[0], start_position=t[1])
                    tokenized.append(token)

        elif self.language_type == 'ug':
            text = self.uy_preprocess(text)
            word = ''
            for index, char in enumerate(text):
                if char == ' ':
                    if len(word) > 0:
                        token = Token(word, start_position=index - len(word), sp=self.sp_op)
                        tokenized.append(token)

                    word = ''
                else:
                    word += char
            index += 1
            if len(word) > 0:
                token = Token(word, start_position=index - len(word), sp=self.sp_op)
                tokenized.append(token)

        else:
            tokenized = []
            tokens = []
            sentences = split_single(text)
            for sentence in sentences:
                contractions = split_contractions(word_tokenizer(sentence))
                tokens.extend(contractions)

            index = text.index
            running_offset = 0
            last_word_offset = -1
            last_token = None
            for word in tokens:
                try:
                    word_offset = index(word, running_offset)
                    start_position = word_offset
                except:
                    word_offset = last_word_offset + 1
                    start_position = running_offset + 1 if running_offset > 0 else running_offset

                token = Token(word, start_position=start_position)
                tokenized.append(token)

                if word_offset - 1 == last_word_offset and last_token is not None:
                    last_token.whitespace_after = False

                word_len = len(word)
                running_offset = word_offset + word_len
                last_word_offset = running_offset - 1
                last_token = token

        return tokenized

    def sentence_split(self, text) -> List[str]:
        pass

    @staticmethod
    def uy_preprocess(text):
        text = re.sub('،', ' ، ', text)
        text = re.sub(r'\.', ' . ', text)
        text = re.sub('!', ' ! ', text)
        text = re.sub('؟', ' ؟ ', text)
        text = re.sub(r'\?', ' ? ', text)
        text = re.sub(r'\(', '( ', text)
        text = re.sub(r'\)', ' )', text)
        text = re.sub('»', ' »', text)
        text = re.sub('«', '« ', text)
        text = re.sub(':', ' :', text)
        text = re.sub('"', ' " ', text)
        text = re.sub('><', '> <', text)
        text = re.sub(r'( )*-( )*', '-', text)

        return text


