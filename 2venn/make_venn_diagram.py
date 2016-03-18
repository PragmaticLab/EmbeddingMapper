threshold_shakespeare = 4
threshold_imdb = 19

shakespeare = open('../1cleanse/shakespeare_vocab.txt').read()
imdb = open('../1cleanse/imdb_vocab.txt').read()

def get_thresholded_list(rawfile, threshold):
	my_list = {}
	lines = rawfile.split('\n')
	for line in lines:
		words = line.split('\t')
		if len(words) < 2:
			continue
		word = words[0]
		count = int(words[1])
		if count >= threshold:
			my_list[word] = count
	return my_list

vocab_shakespeare = get_thresholded_list(shakespeare, threshold_shakespeare)
vocab_imdb = get_thresholded_list(imdb, threshold_imdb)

intersection = set(vocab_shakespeare.keys()) & set(vocab_imdb.keys())
only_shakespeare = set(vocab_shakespeare.keys()) - intersection
only_imdb = set(vocab_imdb.keys()) - intersection

def write_set_to_file(myset, countDict, filename):
	f = open(filename, 'wb')
	wordlist = sorted(countDict.items(), key=lambda kv: kv[1], reverse=True)
	for word, count in wordlist:
		if word in myset:
			f.write(word)
			f.write('\t')
			f.write(str(count))
			f.write('\n')

write_set_to_file(intersection, vocab_shakespeare, "intersection.txt")
write_set_to_file(only_shakespeare, vocab_shakespeare, "only_shakespeare.txt")
write_set_to_file(only_imdb, vocab_imdb, "only_imdb.txt")
