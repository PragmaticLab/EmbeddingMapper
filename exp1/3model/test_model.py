"""
python test_model.py shakespeare.w2v shakespeare
python test_model.py imdb.w2v imdb
"""
from gensim.models import Word2Vec
import sys

model = Word2Vec.load(sys.argv[1])

shakespeare_words = ['night', 'knave', 'highness', 'hither', 'senators', 'wanton', 'son', 'kinsman']
imdb_words = ['night', 'movie', 'actually', 'tv', 'drama', 'america']

def print_similar(word):
	print "-----------------"
	print word + " : "
	print model.most_similar(word)
	print "-----------------\n\n"

if sys.argv[2] == "shakespeare":
	words = shakespeare_words
elif sys.argv[2] == "imdb":
	words = imdb_words

for word in words:
	print_similar(word)
