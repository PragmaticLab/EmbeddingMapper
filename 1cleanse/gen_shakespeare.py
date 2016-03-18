import re
from collections import Counter

f = open('../0corpus/t8.shakespeare.txt')
txt = f.read().lower()
formatted_txt = re.sub(r'([^\s\w]|_)+', '', txt)

# get cleaned corpus
g = open('shakespeare.txt', 'wb')
g.write(formatted_txt)

# get vocab
wordcount = Counter(formatted_txt.split())
# for item in wordcount.items(): print("{}\t{}".format(*item))
wordlist = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)
k = open('shakespeare_vocab.txt', 'wb')
for word, count in wordlist:
	k.write(word)
	k.write('\t')
	k.write(str(count))
	k.write('\n')
