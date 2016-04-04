""" 
args:
- corpus 
- model file
- cut_off
- iter
- left model

python -i train_continued.py ../exp1/1cleanse/imdb.txt imdb_right.w2v 124 5 imdb_left.w2v 

"""

from gensim.models import Word2Vec
import sys
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus_location = sys.argv[1]
output_file = sys.argv[2]
cut_off = int(sys.argv[3])
dim = 512
myiter = int(sys.argv[4])
left_model = Word2Vec.load(sys.argv[5])

f = open(corpus_location)
lines = f.read().split('\n')
corpus = [line.split() for line in lines]

left_vocab = left_model.vocab.keys()
print "left model has " + str(len(left_vocab)) + " vocabs\n\n"


# this is meh... I just want to make a model with just the vocab and random instantiated weights, no training
right_model = Word2Vec(corpus, size=dim, alpha=0.015, window=10, min_count=cut_off, workers=8, sg=1, hs=1, negative=0, sample=1e-3, iter=0)

print "transferring the vocab weights now!"
for word in left_vocab:
	this_index = right_model.vocab[word].index
	right_model.syn0[this_index] = left_model[word]
	right_model.syn0_lockf[this_index] = 0
assert np.sum(right_model.syn0_lockf) == len(right_model.vocab) - len(left_model.vocab)

print "starting training on the rest of the words"
for i in range(myiter):
	print "on iter " + str(i) + "\n\n\n"
	right_model.train(corpus)

right_model.save(output_file)
