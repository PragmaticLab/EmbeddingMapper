import re
from collections import Counter

inputFiles = ['train-pos.txt', 'train-neg.txt', 'test-pos.txt', 'test-neg.txt']

txt = ""
for inputFile in inputFiles:
	print 'reading ../0corpus/' + inputFile
	f = open('../corpus/' + inputFile)
	content = f.read()
	txt += content
formatted_txt = re.sub(r'([^\s\w]|_)+', '', txt.lower())

# get cleaned corpus
g = open('imdb.txt', 'wb')
g.write(formatted_txt)

# get vocab
wordcount = Counter(formatted_txt.split())
# for item in wordcount.items(): print("{}\t{}".format(*item))
wordlist = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)
k = open('imdb_vocab.txt', 'wb')
for word, count in wordlist:
	k.write(word)
	k.write('\t')
	k.write(str(count))
	k.write('\n')
