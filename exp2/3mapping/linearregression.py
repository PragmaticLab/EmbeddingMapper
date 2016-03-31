"""
python linearregression.py ../2model/imdb_left.w2v ../2model/imdb_right.w2v
"""
from sklearn import linear_model
import numpy as np 
from gensim.models import Word2Vec
import sys
import cPickle as pickle

left_model = Word2Vec.load(sys.argv[1])
right_model = Word2Vec.load(sys.argv[2])

intersection_set = []
for line in open('../1venn/intersection.txt').read().split('\n'):
	words = line.split()
	if len(words) == 2:
		count = int(words[1])
		intersection_set += [words[0]]
print "Using " + str(len(intersection_set)) + " words as intersection"

X = []
y = []
for word in intersection_set:
	X += [left_model[word]]
	y += [right_model[word]]
X = np.array(X)
y = np.array(y)

regr = linear_model.LinearRegression()
regr.fit(X, y)

print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X) - y) ** 2))
print('Variance score: %.2f' % regr.score(X, y))
pickle.dump(regr, open('lr.mapping', 'wb'))
