# EmbeddingMapper
Mapping one embedding space to another

--------------------------------------------

finished experiments:
1. imdb -> shakespeare (512 -> 64)
2. imdb -> imdb (512 -> 64)
3. imdb -> imdb (512 -> 512)
4. transfer learning (small vocab -> big vocab)
	- 5000 -> 6000 (by freq)
	- 5000 -> 6000 (by random sample)
	- 5000 -> 10000 (by freq)
	- 5000 -> 10000 (by random sample)


to experiment:
- different vocab during training
	- vocab right is subset of vocab left
	- vocab left is subset of vocab right
	- vocab left and right both have uniques

--------------------------------------------

existing papers: 
- http://arxiv.org/pdf/1309.4168.pdf
- https://www.reddit.com/r/MachineLearning/comments/3mkmdd/word2vec_in_gensim_how_to_generate_translation/

to read:
- http://www.aclweb.org/anthology/W14-4015
- http://www.cs.tau.ac.il/~nachum/papers/jw2v.pdf
- http://www.cs.ust.hk/~dekai/ssst8/slides/AlkouliGutaNey_Ssst2014_slides.pdf
