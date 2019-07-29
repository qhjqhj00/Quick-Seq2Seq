# Quick-Seq2Seq
Still in progress, hopefully will release first available version in this month...

Quick-Seq2Seq is a simple framework to train Seq2Seq models.

The library is designed to provides RNN-based Seq2Seq models with multiple customizable modules (e.g. Attention). 

Expected training codes is like:

```python
from seq2seq.data_loader import load_seq2seq_data
from seq2seq.models import Seq2Seq, Encoder, Decoder
from seq2seq.trainer import Seq2SeqTrainer

corpus = load_seq2seq_data('./data/sample.txt')

src_dict = corpus.make_vocab_dictionary('src')
trg_dict = corpus.make_vocab_dictionary('trg')

print('Data Loaded...')

encoder = Encoder(len(src_dict.idx2item), 256, 512, 2, 0.5)
decoder = Decoder(len(trg_dict.idx2item), 256, 512, 2, 0.5)

seq2seq = Seq2Seq(encoder,decoder,src_dict,trg_dict)

print('Model initialized...')
trainer = Seq2SeqTrainer(seq2seq, corpus)

print('Begin to train...')

trainer.train('test')
```

Before training, sequence pairs shall be processed as:

```
source sentence 1\t target sentence 1\n
source sentence 2\t target sentence 2\n
...
source sentence k-1\t target sentence k-1\n
source sentence k\t target sentence k\n
```

sample.txt is a sample for data format.

Currently, tokenizer of the library supports Chinese, English, Uyghur, which can be automatically switched.  

