from sklearn.decomposition import PCA
import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

model_left = Word2Vec.load("imdb_left.w2v")
model_right = Word2Vec.load("imdb_right.w2v")

# left_indices = []
# right_indices = []
# for word in ["one", "two", "three", "four", "five"]:
# 	left_indices += [model_left.vocab[word].index]
# 	right_indices += [model_right.vocab[word].index]

# pca = PCA(n_components=2)
# left_syn0 = pca.fit_transform(model_left.syn0[left_indices,])
# pca = PCA(n_components=2)
# right_syn0 = pca.fit_transform(model_right.syn0[right_indices,])

# for i in range(5):
# 	model = left_syn0[i]
# 	plt.scatter(model[0],model[1])
# 	plt.annotate(str(i), (model[0],model[1]))
# plt.show()

# for i in range(5):
# 	model = right_syn0[i]
# 	plt.scatter(model[0],model[1])
# 	plt.annotate(str(i), (model[0],model[1]))
# plt.show()


