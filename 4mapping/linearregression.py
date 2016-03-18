"""
python linearregression.py ../3model/shakespeare.w2v ../3model/imdb.w2v
"""
from sklearn import linear_model
import numpy as np 
from gensim.models import Word2Vec
import sys
import cPickle as pickle

shakespeare_model = Word2Vec.load(sys.argv[1])
imdb_model = Word2Vec.load(sys.argv[2])

intersection_set = []
for line in open('../2venn/intersection.txt').read().split('\n'):
	words = line.split()
	if len(words) == 2:
		intersection_set += [words[0]]

X = []
y = []
for word in intersection_set:
	X += [imdb_model[word]]
	y += [shakespeare_model[word]]
X = np.array(X)
y = np.array(y)

regr = linear_model.LinearRegression()
regr.fit(X, y)

print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(X) - y) ** 2))
print('Variance score: %.2f' % regr.score(X, y))
pickle.dump(regr, open('lr.mapping', 'wb'))
