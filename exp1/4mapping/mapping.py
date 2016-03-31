"""
python mapping.py ../3model/shakespeare.w2v ../3model/imdb.w2v

mapped_most_similar('venom')
"""
from sklearn import linear_model
import numpy as np 
from gensim.models import Word2Vec
import sys
import cPickle as pickle

mapping = pickle.load(open('lr.mapping'))
shakespeare_model = Word2Vec.load(sys.argv[1])
imdb_model = Word2Vec.load(sys.argv[2])

def mapped_most_similar(word, topn=20):
	score_dict = {}
	imdb_vector = imdb_model[word]
	mapped_vector = mapping.predict(imdb_vector)[0]
	for word, shakespeare_vector in shakespeare_model.vocab.items():
		score_dict[word] = shakespeare_model[word].dot(mapped_vector)
	sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
	return sorted_list[:topn]
