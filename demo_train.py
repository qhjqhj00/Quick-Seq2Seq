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