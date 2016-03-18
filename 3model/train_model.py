""" 
args:
- corpus 
- model file
- cut_off
- dim
- iter

python train_model.py ../1cleanse/shakespeare.txt shakespeare.w2v 4 64 20
python train_model.py ../1cleanse/imdb.txt imdb.w2v 19 512 5

"""

from gensim.models import Word2Vec
import sys

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus_location = sys.argv[1]
output_file = sys.argv[2]
cut_off = int(sys.argv[3])
dim = int(sys.argv[4])
myiter = int(sys.argv[5])

f = open(corpus_location)
lines = f.read().split('\n')
corpus = [line.split() for line in lines]

model = Word2Vec(corpus, size=dim, alpha=0.015, window=10, min_count=cut_off, workers=8, sg=1, hs=1, negative=0, sample=1e-3, iter=myiter)

model.save(output_file)
