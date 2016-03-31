"""
python test.py imdb_left.w2v imdb_right.w2v
"""
from gensim.models import Word2Vec
import sys

left_model = Word2Vec.load(sys.argv[1])
right_model = Word2Vec.load(sys.argv[2])

imdb_words = ['night', 'movie', 'actually', 'tv', 'drama', 'america', 'time', 'dance', 'ignore', 'brave', 'racism']

def print_similar(model, word):
	print "-----------------"
	print word + " : "
	print model.most_similar(word)

for word in imdb_words:
	print_similar(left_model, word)
	print_similar(right_model, word)
	print "\n\n"
