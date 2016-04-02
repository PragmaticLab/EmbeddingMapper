"""
Takes in 2 models, a mapper, and a list of words that were not used during training
Outputs a score (% correct)

# exp2
python -i eval_mapping.py ../exp2/2model/imdb_left.w2v ../exp2/2model/imdb_right.w2v ../exp2/3mapping/lr.mapping ../exp2/1venn/outer2.txt 5

# exp3
python -i eval_mapping.py ../exp2/2model/imdb_left.w2v 1model/imdb_right.w2v 2mapping/lr.512to512.mapping ../exp2/1venn/outer2.txt 3

"""

from sklearn import linear_model
import numpy as np 
from gensim.models import Word2Vec
import sys
import cPickle as pickle
from scipy.spatial.distance import cosine
import random

def getOuterVocab(filedir):
	lines = open(filedir).read().split('\n')
	lines = lines[:len(lines) - 1]
	words = [line.split('\t')[0] for line in lines]
	return words

left_model = Word2Vec.load(sys.argv[1])
right_model = Word2Vec.load(sys.argv[2])
mapping = pickle.load(open(sys.argv[3]))
outer_vocab = getOuterVocab(sys.argv[4])
topnfilter = int(sys.argv[5])

def mapped_most_similar(word, topn=3, verbose=0):
	if verbose == 1:
		print "left: ------"
		print left_model.most_similar(word)
		print "right: ------"
		print right_model.most_similar(word)
		print "\nmapped: ------"
	score_dict = {}
	vector = left_model[word] # the 512 dim vector
	mapped_vector = mapping.predict(vector)[0] # the 64 dim vector
	for word, left_vector in right_model.vocab.items():
		score_dict[word] = -cosine(right_model[word], mapped_vector)
	sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
	return sorted_list[:topn]


# simple scoring loop to see if a word is top 5
myscore = 0
mysize = 250
count = 0
for word in random.sample(outer_vocab, mysize):
	if count % 10 == 0:
		print str(count) + " / " + str(mysize)
	results = mapped_most_similar(word, topnfilter)
	for res_word, score in results:
		if res_word == word:
			myscore += 1
			break
	count += 1

print "my score is: " + str(float(myscore) / mysize)
