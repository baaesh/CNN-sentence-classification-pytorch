import numpy as np

from torchtext import data
from torchtext import datasets

from gensim.models import KeyedVectors

import mydatasets

def getVectors(args, data):
	vectors = []

	if args.mode != 'rand':
		word2vec = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

		for i in range(len(data.TEXT.vocab)):
			word = data.TEXT.vocab.itos[i]
			if word in word2vec.vocab:
				vectors.append(word2vec[word])
			else:
				vectors.append(np.random.uniform(-0.01, 0.01, args.word_dim))
	else:
		for i in range(len(data.TEXT.vocab)):
			vectors.append(np.random.uniform(-0.01, 0.01, args.word_dim))

	return np.array(vectors)


class DATA():

	def __init__(self, args):
		self.TEXT = data.Field(batch_first=True, lower=True, fix_length=70)
		self.LABEL = data.Field(sequential=False, unk_token=None)

		if args.dataset == 'TREC':
			self.train, self.test = datasets.TREC.splits(self.TEXT, self.LABEL)
		elif args.dataset == 'MR':
			self.train, self.test = mydatasets.MR.splits(self.TEXT, self.LABEL)
		elif args.dataset == 'SUBJ':
			self.train, self.test = mydatasets.SUBJ.splits(self.TEXT, self.LABEL)
		elif args.dataset == 'SST-1':
			self.train, self.dev, self.test = datasets.SST.splits(self.TEXT, self.LABEL, fine_grained=True,
																  train_subtrees=False)  # , filter_pred=lambda ex: ex.label != 'neutral')
		elif args.dataset == 'SST-2':
			self.train, self.dev, self.test = datasets.SST.splits(self.TEXT, self.LABEL, fine_grained=False,
																  train_subtrees=False,
																  filter_pred=lambda ex: ex.label != 'neutral')
		else:
			print("invalid dataset name")
			return


		self.TEXT.build_vocab(self.train, self.test)
		self.train_iter, self.test_iter = \
				data.BucketIterator.splits((self.train, self.test),
										   batch_size=args.batch_size,
										   device=args.gpu)
		self.LABEL.build_vocab(self.train)


