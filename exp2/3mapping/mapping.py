"""
python -i mapping.py ../2model/imdb_left.w2v ../2model/imdb_right.w2v

"""
from sklearn import linear_model
import numpy as np 
from gensim.models import Word2Vec
import sys
import cPickle as pickle
from scipy.spatial.distance import cosine

mapping = pickle.load(open('lr.mapping'))
left_model = Word2Vec.load(sys.argv[1])
right_model = Word2Vec.load(sys.argv[2])

def mapped_most_similar(word, topn=20):
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
