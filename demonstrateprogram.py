# to install nltk/matplotlib
# run sudo pip install -U nltk
# python3 -mpip install matplotlib

#to run in Terminal: python3 program.py

import re
import nltk
import matplotlib.pyplot
import os
import itertools
import pickle
import operator
import numpy as np
from nltk.collocations import *
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
bigram_measures = nltk.collocations.BigramAssocMeasures()

#PRE-PROCESSING OF THE TEXT DATA

def raw_preprocess(textfile):
	#order of functions:
		#read in x
		#tokenise x
		#normalize x
		#perform tests on most common words x
		#contractions (needs to come before removal of punctuation and negation) x
		#remove stopwords x
		#negation (needs to come before removal of punctuation) x
		#lemmatize x
		#remove punctuation x
		#remove other words x
		#tag x 4: default, RE, unigram, bigram x

	domain = open(textfile) #read in data from text file
	raw_domain = domain.read()

	tokens = nltk.word_tokenize(raw_domain) #tokenise 
	words = [w.lower() for w in tokens] #normalize

	wnl = nltk.WordNetLemmatizer()
    
    #find the lexical diversity of the debate
	def lex_diversity(text):
		return len(set(text)) / len(text) * 100

	#find the most common collocations within the debate
	def collocations(text): 
		finder = BigramCollocationFinder.from_words(tokens)
		return sorted(finder.above_score(bigram_measures.raw_freq,3.0 / len(tuple(nltk.bigrams(tokens)))))

	#process and replace contracted words with full phrases
	def contractions(text):
		#list of contractions
		contractions=[
		['don','t','do','not'],
		['can','t','can','not'],
		['isn','t','is','not'],
		['aren','t','are','not'],
		['wasn','t','was','not'],
		['weren','t','were','not'],
		['hasn','t','has','not'],
		['haven','t','have','not'],
		['hadn','t','had','not'],
		['won','t','will','not'],
		['wouldn','t','would','not'],
		['doesn','t','does','not'],
		['didn','t','did','not'],
		['couldn','t','could','not'],
		['shouldn','t','should','not'],
		['mightn','t','might','not'],
		['mustn','t','must','not'],

		['would','ve','would','have'],
		['could','ve','could','have'],
		['should','ve','should','have'],

		['I','m','I','am'],
		['I','ll','I','will'],
		['I','d','I','would'],
		['I','ve','I','have'],
		['I','d','I','had'],

		['you','re','you','are'],
		['you','ll','you','will'],
		['you','d','you','would'],
		['you','ve','you','have'],
		['you','d','you','had'],

		['he','s','he','am'],
		['he','ll','he','will'],
		['he','d','he','would'],
		['he','s','he','has'],
		['he','d','he','had'],

		['she','s','she','am'],
		['she','ll','she','will'],
		['she','d','she','would'],
		['she','s','she','has'],
		['she','d','she','had'],

		['it','s','it','is'],
		['it','ll','it','will'],
		['it','d','it','would'],
		['it','s','it','has'],
		['it','d','it','had'],

		['we','re','we','are'],
		['we','ll','we','will'],
		['we','d','we','would'],
		['we','ve','we','have'],
		['we','d','we','had'],

		['they','re','they','are'],
		['they','ll','they','will'],
		['they','d','they','would'],
		['they','ve','they','have'],
		['they','d','they','had'],

		['that','s','that','is'],
		['that','ll','that','will'],
		['that','d','that','would'],
		['that','s','that','has'],
		['that','d','that','had'],

		['who','s','who','is'],
		['who','ll','who','will'],
		['who','d','who','would'],
		['who','s','who','has'],
		['who','d','who','had'],

		['what','s','what','is'],
		['what','re','what','are'],
		['what','ll','what','will'],
		['what','d','what','would'],
		['what','s','what','has'],
		['what','d','what','had'],

		['where','s','where','is'],
		['where','re','where','are'],
		['where','ll','where','will'],
		['where','d','where','would'],
		['where','s','where','has'],
		['where','d','where','had'],

		['when','s','when','is'],
		['when','re','when','are'],
		['when','ll','when','will'],
		['when','d','when','would'],
		['when','s','when','has'],
		['when','d','when','had'],

		['why','s','why','is'],
		['why','re','why','are'],
		['why','ll','why','will'],
		['why','d','why','would'],
		['why','s','why','has'],
		['why','d','why','had'],

		['how','s','how','is'],
		['how','re','how','are'],
		['how','ll','how','will'],
		['how','d','how','would'],
		['how','s','how','has'],
		['how','d','how','had']
		]

		location = 0
		for word in text:
			if word == '’':
				for (before,after,newbefore,newafter) in contractions:
					if text[location-1] == before:
						if text[location+1] == after:
							text[location-1] = newbefore
							text[location+1] = newafter
			location+=1
		return text

	#append NOT_ to each word that follows 'not' within a sentence or clause
	def negation(text): # apply negation
		for w in range(0,len(text)): #for word in the text
			if (text[w] == 'not'): #if that word is not - add more??
				n=1
				not_ = w
				while True : #while still in sentence/word clause and not at end of text
					text[not_+n] = 'NOT_'+text[not_+n] #add 'NOT' to each word
					n+=1
					if ((not_ + n) >= len(text)): break
					if text[not_+n] in ('NOT_.','.','NOT_?','?','NOT_!','!','NOT_,',',','NOT_:',':','NOT_;',';'):#until end of sentence
						break
		return text

	#common, potentially skewing, words in text
	words_to_remove=[
	'baroness',
	'lord',
	'lords',
	'noble'
	]

	words = contractions(words) #apply above function: deal with contractions and apostrophes
	words = [word for word in words if word not in stopwords.words('english') or word in ('not')] #remove stopwords
	words = negation(words) #deal with negations in the text
	words = [wnl.lemmatize(w) for w in words] #lemmatize
	words = [word for word in words if word.isalpha() or word.startswith('NOT_')] #remove punctuation
	#print("distinct words: ", distinct_words(words))
	#print("lexical diversity: ",lex_diversity(words),'%')
	#print("collocations: ",collocations(words))
	words = [word for word in words if word not in words_to_remove]
	words = [word.replace('NOT_'+r'^-?[0-9]+$','NOT_') for word in words] #remove punctuation/special characters aside from the underscore in NOT_
	words = [word.replace('NOT_'+r'(\.|\,|!\?)','NOT_') for word in words]
	words = [word for word in words if word not in ('NOT_')]
	
	#print(words)
	#TAG REMAINING WORDS

	#find more frequent (therefore default) tagger
	bts = brown.tagged_sents(categories='news', tagset = 'universal')
	tags = [t for(w,t) in brown.tagged_words(categories='news', tagset = 'universal')]                                       
	fd = nltk.FreqDist(tags)
	fd.most_common(1) #most common is NOUN

	#regular expressions tagger
	patterns=[
	(r'.+ing$','VERB'),	#	gerunds
	(r'.+ed$','VERB'),		#    past tense of verbs
	(r'.+es$','VERB'),		#    present tense
	(r'.+ould$','VERB'),	#    modal verb
	(r'.+\'s$','NOUN'),		#    possessive
	(r'.+s$','NOUN'),		#	 plural nouns
	(r'^-?[0-9]+$','NUM'),	#     cardinal numbers
	(r'.+ly$','ADV'),         #     adverbs
	(r'(^the$|^a$|^wh)','DET'), # determiner
	(r'^[A-Z]','NOUN'),		#     proper names
	(r'(^he$|^she$|^they$|^him$|^her$|^his$|^hers$|^theirs$)','PRO'), #personal pronouns
	(r'\b(at|in|of)(?:\w+\s){0,3}([A-Z]\w+)', 'PPO'), # prepositions
	(r'(^can$|^may$|^must$|^should$|^would$|^could$)','MOD'), #modals
	(r'(\.|\,|!\?)','.')
	] 

	#unigram
	size = int(len(bts)*0.9)
	train = bts[:size]
	test = bts[size:]
	unigram_tagger=nltk.UnigramTagger(train)
	unigram_tagger.evaluate(test)

	#bringing in context: n-gram except issue with sparse data problem
	#bigram
	bg_tag = nltk.BigramTagger(train)
	bg_tag.evaluate(test)
	bg_tagged_sents = [bg_tag.tag(s) for s in words]
	bad_tags = [s for s in bg_tagged_sents if None in [tag for (w,tag) in s]]
	bad_tags[0]

	#combine different taggers to get the most accurate
	t0=nltk.DefaultTagger('NOUN')
	t1=nltk.RegexpTagger(patterns, backoff =t0)
	t2=nltk.UnigramTagger(train, backoff=t1)
	t3=nltk.BigramTagger(train, backoff= t2)
	t3.evaluate(test)

	tagged_words = [t3.tag(words)]
 
	#count up the NOUN/VERB/ADV/ADJ in the text
	#c = tuple(i for i in tagged_words[0])
	#d = Counter(elem[1] for elem in c)
	#print(d)
	#print(tagged_words)
	return tagged_words


#raw_preprocess is called once for lexicon based: the file we want to know the sentiment polarity of
def lexiconb_prep(textfile,root):
	filepath = root + '/' + textfile #UNCOMMENT FOR MULTIPLE FILES
	#filepath = root + textfile #UNCOMMENT FOR SINGLE FILE
	print(" ")
	print(textfile)
	processed_words = raw_preprocess(filepath)
	return processed_words

#called multiple times for ML: training data
def mlb_prep(root):
	root_list = []
	for textfile in os.listdir(root):
		print(textfile)
		filepath = root + '/' + textfile
		processed_words = raw_preprocess(filepath)
		root_list.extend(processed_words)
	return root_list


#LEXICON BASED SENTIMENT ANALYSIS 

def lexicon_based(textfile,root):
	processed_words = lexiconb_prep(textfile,root);

	#import and clean up lexicon
	domain = open('SentiWords.txt')
	raw_domain = domain.read()

	lexicon = raw_domain.splitlines()

	#split array into parts
	def splitl(word):
		b = ''
		c = ''
		temp = word.split("#")
		a = temp[0]
		t1 = temp[1]
		b = t1[0]
		c = t1[2:]
		full = (a,b,c)
		return full

	#match parts of speech formats
	def pos_match(a,b):
		res = ''
		if (a == 'NOUN' and b == 'n'):
			res = 'TRUE'
		elif (a == 'ADJ' and b == 'a'):
			res = 'TRUE'
		elif (a == 'ADV' and b == 'r'):
			res = 'TRUE'
		elif (a == 'VERB' and b == 'v'):
			res = 'TRUE'
		else:
			res = 'FALSE'
		return res

	#divide up lexicon and append to dictionary
	lexiconsplit = [splitl(w) for w in lexicon] #split into tuples in a dictionary: word/phrase, type of word, polarity

	#calculate the polarity of the debate
	def calc_polarity(processed_words,lexiconsplit):
		polarity_score = 0
		for w in processed_words: #each list (just 1 document)
			word_score = 0
			word = ''
			lexiconword = ''
			no_counted_words = 0
			for (a,b) in w: #each tuple: word/part of speech
				word = a
				pos = b
				negated_word = ''
				#print(word)
				if word.startswith('NOT_'): #if the word is negated
					negated_word = word.replace("NOT_", "") #remove the NOT: but will reverse the polarity later
					word = ''
				for l in lexiconsplit: #each tuple: word/part of speech/polarity
					lexiconword = l[0]
					lexiconPOS = l[1]
					if negated_word == lexiconword:
						#if pos_match(pos,lexiconPOS) == 'TRUE': #match on parts of speech
						word_score = float(l[2]) * -1 #reverse the polarity of negated word
						polarity_score += word_score
						no_counted_words += 1
						#print("word score: ",word_score)
						#print("polarity score: ",polarity_score)
						break
					elif word == lexiconword:
						#if pos_match(pos,lexiconPOS) == 'TRUE': 
						word_score = l[2]
						polarity_score += float(word_score) #match up processed word with lexicon
						no_counted_words += 1
						#print("word score: ",word_score)
						#print("polarity score: ",polarity_score)
						break
			polarity_score = polarity_score/no_counted_words #AVERAGE
			return polarity_score

	polarity = calc_polarity(processed_words,lexiconsplit)
	return polarity
	
#MACHINE LEARNING SENTIMENT ANALYSIS 

def machine_learning(root_read,root_test):

	def train_classifier(root):

		#training data
		mlb_pos = mlb_prep("HoL/pos")
		#returns cleaned list of positive words
		mlb_neu = mlb_prep("HoL/neu")
		#returns cleaned list of neutral words
		mlb_neg = mlb_prep("HoL/neg")
		#returns cleaned list of negative words

		#remove duplicates
		mlb_pos_flat = [item for sublist in mlb_pos for item in sublist]
		mlb_neu_flat = [item for sublist in mlb_neu for item in sublist]
		mlb_neg_flat = [item for sublist in mlb_neg for item in sublist]

		#dedupe each list
		mlb_pos_flat.sort()
		mlb_neu_flat.sort()
		mlb_neg_flat.sort()

		pos_train = list(mlb_pos_flat for mlb_pos_flat,_ in itertools.groupby(mlb_pos_flat))
		neu_train = list(mlb_neu_flat for mlb_neu_flat,_ in itertools.groupby(mlb_neu_flat))
		neg_train = list(mlb_neg_flat for mlb_neg_flat,_ in itertools.groupby(mlb_neg_flat))

		#join all lists to create a set of all words
		all_words = sum([pos_train,neu_train,neg_train],[])

		#dedupe words list
		all_words.sort()
		words = list(all_words for all_words,_ in itertools.groupby(all_words))

		classifier_list = []

		for w in words:
			pos_count = 0
			neu_count = 0
			neg_count = 0

			pos_prob = 0
			neu_prob = 0
			neg_prob = 0

			#how many documents in class does the word appear in out of how many overall? calculate probabilities for each class
			for a in mlb_pos:
				try:
					pos_loc = a.index(w)
				except ValueError:
					pos_loc = -1
				if pos_loc >= 0:
					pos_count += 1
				pos_prob = (pos_count + 1)/len(mlb_pos) #laplace smoothing: always a possibility a word could appear there

			for b in mlb_neu:
				try: 
					neu_loc = b.index(w) 
				except ValueError:
					neu_loc = -1
				if neu_loc >= 0:
					neu_count += 1
				neu_prob = (neu_count + 1)/len(mlb_neu) #laplace smoothing: always a possibility a word could appear there

			for c in mlb_neg:
				try: 
					neg_loc = c.index(w) 
				except ValueError:
					neg_loc = -1
				if neg_loc >= 0:
					neg_count += 1
				neg_prob = (neg_count + 1)/len(mlb_neg) #laplace smoothing: always a possibility a word could appear there
			
			probs = []
			probs.append(pos_prob)
			probs.append(neu_prob)
			probs.append(neg_prob)

			word_classifier = []
			word_classifier.append(w[0]) #word
			word_classifier.append(w[1]) #POS
			word_classifier.append(probs) #3 probabilities

			classifier_list.append(word_classifier)

		return classifier_list

	#probability/classifier is represented as an array of 3 numbers: the probablility the word is in each class

	#use classifier to classify test data
	def use_classifier(processed_test,classifier_list):
		polarity = 0
		pol_score = 0
		found_words = 0
		overall_probs = []

		#flatten and dedupe processed_test
		processed_test = [item for sublist in processed_test for item in sublist]
		processed_test.sort()
		processed_test = list(processed_test for processed_test,_ in itertools.groupby(processed_test))

		for (w,x) in processed_test:
			for (a,b,c) in classifier_list:
				word_lex = a
				word_POS = b
				word_polarity = c

				if word_lex == w and word_POS == x:
					found_words += 1
					found = [a,b,c]
					probs = found[2]

					if (found_words == 1):
						prob_pos = probs[0]
						prob_neu = probs[1]
						prob_neg = probs[2]

					else:
						prob_pos = prob_pos * probs[0]
						prob_neu = prob_neu * probs[1]
						prob_neg = prob_neg * probs[2]

		overall_probs.append(prob_pos)
		overall_probs.append(prob_neu)
		overall_probs.append(prob_neg)

		maxx = max(overall_probs) #get highest class probability
		loc = overall_probs.index(maxx) #0 is positive 1 is neutral 2 is negative

		return loc

	classifier_list = train_classifier(root_read)

	with open("test.txt", "wb") as fp:   #pickling
		pickle.dump(classifier_list, fp)

	with open("test.txt", "rb") as fp:   #unpickling
		classifier_list = pickle.load(fp)

	print(classifier_list)

	#call test classifier for each document
	polarity_scores = []
	for textfile in os.listdir(root_test):
		store = []
		polarity = ''
		filepath = root_test + '/' + textfile
		processed_test = raw_preprocess(filepath)
		polarity_score = use_classifier(processed_test, classifier_list)

		if (polarity_score == 0):
			polarity = 'positive'
		elif(polarity_score == 1):
			polarity = 'neutral'
		elif(polarity_score == 2):
			polarity = 'negative'

		#print(polarity_score)
		#print(polarity)

		store.append(textfile)
		store.append(polarity)
		store.append(polarity_score)
		polarity_scores.append(store)

	return polarity_scores


#function to input user opinion of text based on position in folders
def manual_labels(root):
	polarity_scores = []

	for textfile in os.listdir(root + '/pos'):
		store = []
		polarity = 'positive'
		polarity_score = 1

		store.append(textfile)
		store.append(polarity)
		store.append(polarity_score)

		polarity_scores.append(store)

	for textfile in os.listdir(root + '/neu'):
		store = []
		polarity = 'neutral'
		polarity_score = 0

		store.append(textfile)
		store.append(polarity)
		store.append(polarity_score)

		polarity_scores.append(store)

	for textfile in os.listdir(root + '/neg'):
		store = []
		polarity = 'negative'
		polarity_score = -1

		store.append(textfile)
		store.append(polarity)
		store.append(polarity_score)

		polarity_scores.append(store)

	return polarity_scores

def lexicon_test(root):
	polarities = []
	for textfile in os.listdir(root):
		store = []
		polarity = ''
		filepath = root + '/' + textfile
		polarity_score = lexicon_based(textfile,root)
		print("polarity score: ",polarity_score)
		store.append(textfile)
		if polarity_score >= 0.12:
			polarity = 'positive'
		elif(0.09 <= polarity_score < 0.12):
			polarity = 'neutral'
		elif(polarity_score < 0.09):
			polarity = 'negative'
		store.append(polarity)
		store.append(polarity_score)
		polarities.append(store)
	return polarities


#evaluation functions
def eval(alglex_test,manual,classno):

	results = []

	#sort both to align debates
	alglex_test.sort(key=operator.itemgetter(0))
	manual.sort(key=operator.itemgetter(0))

	#confusion matrix function
	def conf_mat(alglex_test,manual,classno):
		# initialize a 3 x 3 confusion matrix to zeros
		cm= np.zeros((3,3))
		#loop through all results and update the confusion matrix
		#set positions for polarity classes in matrix (co-ordinates)
		def convert(clss):
			if (clss == 'negative'):
				return 0
			if (clss == 'neutral'):
				return 1
			if (clss == 'positive'):
				return 2
			else:
				print('Error')
				return 3

		#iterate through results in each list: sentiment analysis vs manual labels
		for i in range(0,len(alglex_test)):
			for j in range(0,len(manual)):

				#access polarity values
				alglex = alglex_test[i]
				man = manual[j]

				title_alglex = alglex[0]
				title_man = man[0]

				if(title_alglex == title_man): #check titles to match 
					polarity_alglex = alglex[1]
					polarity_man = man[1]
					#use polarities as co-ordinates to produce counts in confusion matrix
					count = cm[convert(polarity_alglex),convert(polarity_man)]
					cm[convert(polarity_alglex),convert(polarity_man)] = count + 1
		return cm

	#calculate precision given the true/predicted labels
	def accuracy_precision(alglex_test,manual):
		store = []
		overall_accuracy = 0
		f1 = 0
		macroaverage = 0
		microaverage = 0

		TruePos = 0
		TrueNeu = 0
		TrueNeg = 0

		TPosFNeu = 0
		TPosFNeg = 0
		TNeuFPos = 0
		TNeuFNeg = 0
		TNegFPos = 0
		TNegFNeu = 0

		for i in range(0,(len(alglex_test))):
			for j in range(0,len(manual)):
				alglex = alglex_test[i]
				man = manual[j]

				title_alglex = alglex[0]
				title_man = man[0]

				if(title_alglex == title_man):

					polarity_alglex = alglex[1]
					polarity_man = man[1]

					#print(title_alglex)
					#print(title_man)

					#print("sentiment analysis: ",polarity_alglex)
					#print("manual label: ",polarity_man)
					#print(" ")
					
					if (polarity_alglex == polarity_man):
						if (polarity_man == 'positive'):
							TruePos += 1
						if (polarity_man == 'neutral'):
							TrueNeu += 1
						elif (polarity_man == 'negative'):
							TrueNeg += 1

					elif (polarity_alglex != polarity_man):
						if (polarity_man == 'positive' and polarity_alglex == 'neutral'): #true polarity was +ve, manual was -ve, false negative
							TPosFNeu += 1	
						elif (polarity_man == 'positive' and polarity_alglex == 'negative'): 
							TPosFNeg += 1
						elif (polarity_man == 'neutral' and polarity_alglex == 'positive'): 
							TNeuFPos += 1
						elif (polarity_man == 'neutral' and polarity_alglex == 'negative'): 
							TNeuFNeg += 1
						elif (polarity_man == 'negative' and polarity_alglex == 'positive'): 
							TNegFPos += 1
						elif (polarity_man == 'negative' and polarity_alglex == 'neutral'): 
							TNegFNeu += 1

		#three classes
		overall_accuracy = (TruePos + TrueNeu + TrueNeg)/(TruePos + TrueNeu + TrueNeg + TPosFNeu + TPosFNeg + TNeuFPos + TNeuFNeg + TNegFPos + TNegFNeu)

		#with regard to the positive class
		POS_TP = TruePos 
		POS_FP = TNeuFPos + TNegFPos
		POS_TN = TrueNeu + TrueNeg + TNegFNeu + TNeuFNeg
		POS_FN = TPosFNeu + TPosFNeg

		POS_accuracy = (POS_TP + POS_TN) / (POS_TP + POS_FP + POS_TN + POS_FN)
		POS_precision = POS_TP/(POS_TP + POS_FP)
		POS_recall = POS_TP/(POS_TP + POS_FN)
		POS_f1 = 2 * ((POS_recall * POS_precision) / (POS_recall + POS_precision))

		#with regard to the neutral class
		NEU_TP = TrueNeu
		NEU_FP = TPosFNeu + TNegFNeu
		NEU_TN = TruePos + TrueNeg + TPosFNeg + TNegFPos
		NEU_FN = TNeuFPos + TNeuFNeg

		NEU_accuracy = (NEU_TP + NEU_TN) / (NEU_TP + NEU_FP + NEU_TN + NEU_FN)
		NEU_precision = NEU_TP/(NEU_TP + NEU_FP)
		NEU_recall = NEU_TP/(NEU_TP + NEU_FN)
		NEU_f1 = 2 * ((NEU_recall * NEU_precision) / (NEU_recall + NEU_precision))

		#with regard to the negative class
		NEG_TP = TrueNeg
		NEG_FP = TPosFNeg + TNeuFNeg
		NEG_TN = TruePos + TrueNeu + TPosFNeu + TNeuFPos
		NEG_FN = TNegFPos + TNegFNeu

		NEG_accuracy = (NEG_TP + NEG_TN) / (NEG_TP + NEG_FP + NEG_TN + NEG_FN)
		NEG_precision = NEG_TP/(NEG_TP + NEG_FP)
		NEG_recall = NEG_TP/(NEG_TP + NEG_FN)
		NEG_f1 = 2 * ((NEG_recall * NEG_precision) / (NEG_recall + NEG_precision))

		microaverage_precision = (POS_TP + NEU_TP + NEG_TP) / (POS_TP + POS_FP + NEU_TP + NEU_FP + NEG_TP + NEG_FP)
		microaverage_recall = (POS_TP + NEU_TP + NEG_TP) / (POS_TP + POS_FN + NEU_TP + NEU_FN + NEG_TP + NEG_FN)
		microaverage_f1 = 2 * ((microaverage_recall * microaverage_precision) / (microaverage_recall + microaverage_precision))
		macroaverage_precision = (POS_precision + NEU_precision + NEG_precision) / 3
		macroaverage_recall = (POS_recall + NEU_recall + NEG_recall) / 3
		macroaverage_f1 = (POS_f1 + NEU_f1 + NEG_f1) / 3

		store.append(overall_accuracy)
		store.append(microaverage_precision)
		store.append(macroaverage_precision)
		store.append(microaverage_recall)
		store.append(macroaverage_recall)
		store.append(microaverage_f1)
		store.append(macroaverage_f1)

		store.append(TruePos)
		store.append(TrueNeu)
		store.append(TrueNeg)
		store.append(TPosFNeu)
		store.append(TPosFNeg)
		store.append(TNeuFPos)
		store.append(TNeuFNeg)
		store.append(TNegFPos)
		store.append(TNegFNeu)

		store.append(POS_accuracy)
		store.append(POS_precision)
		store.append(POS_recall)
		store.append(POS_f1)

		store.append(NEU_accuracy)
		store.append(NEU_precision)
		store.append(NEU_recall)
		store.append(NEU_f1)

		store.append(NEG_accuracy)
		store.append(NEG_precision)
		store.append(NEG_recall)
		store.append(NEG_f1)

		return store

	all_eval = accuracy_precision(alglex_test,manual)
	confmat = conf_mat(alglex_test,manual,classno)
	
	results.append("overall accuracy: " + str(all_eval[0] * 100) + "%")
	results.append("microaverage of precision: " + str(all_eval[1]))
	results.append("macroaverage of precision: " + str(all_eval[2]))
	results.append("microaverage of recall: " + str(all_eval[3]))
	results.append("macroaverage of recall:" + str(all_eval[4]))
	results.append("microaverage of f1: " + str(all_eval[5]))
	results.append("macroaverage of f1: " + str(all_eval[6]))
	results.append("confusion matrix: ")
	results.append(str(confmat))
	results.append("")
	results.append("true positive: " + str(all_eval[7]))
	results.append("true neutral: " + str(all_eval[8]))
	results.append("true negative: " + str(all_eval[9]))
	results.append("")
	results.append("true positive false neutral: " + str(all_eval[10]))
	results.append("true positive false negative: " + str(all_eval[11]))
	results.append("true neutral false positive: " + str(all_eval[12]))
	results.append("true neutral false negative: " + str(all_eval[13]))
	results.append("true negative false positive: " + str(all_eval[14]))
	results.append("true negative false neutral: " + str(all_eval[15]))
	results.append("")
	results.append("accuracy with regard to the positive class: " + str(all_eval[16] * 100) + "%")
	results.append("precision with regard to the positive class: " + str(all_eval[17]))
	results.append("recall with regard to the positive class: " + str(all_eval[18]))
	results.append("f1 with regard to the positive class: " + str(all_eval[19]))
	results.append("")
	results.append("accuracy with regard to the neutral class: " + str(all_eval[20] * 100) + "%")
	results.append("precision with regard to the neutral class: " + str(all_eval[21]))
	results.append("recall with regard to the neutral class: " + str(all_eval[22]))
	results.append("f1 with regard to the neutral class: " + str(all_eval[23]))
	results.append("")
	results.append("accuracy with regard to the negative class: " + str(all_eval[24] * 100) + "%")
	results.append("precision with regard to the negative class: " + str(all_eval[25]))
	results.append("recall with regard to the negative class: " + str(all_eval[26]))
	results.append("f1 with regard to the negative class: " + str(all_eval[27]))
	results.append("")
	return results

#TO CALL FUNCTIONS

#RUN LEXICON BASED
#lex_one = lexicon_based('positive.txt','')
#print(lex_one)
#lex_all = lexicon_test('HoL/all')

#with open("pickle/lex_all.txt", "wb") as fp:   #pickling
	#pickle.dump(lex_all, fp)

with open("pickle/lex_all.txt", "rb") as fp:   #unpickling
	lex_all = pickle.load(fp)
print(" ")
print('lexicon analysis results:')
print(lex_all)

#RUN ML BASED
#alg_test = machine_learning('HoL/all','HoL/all_test')

#with open("pickle/alg.txt", "wb") as fp:   #pickling
	#pickle.dump(alg_test, fp)

with open("pickle/alg.txt", "rb") as fp:   #unpickling
	alg_test = pickle.load(fp)
print(" ")
print('naive Bayes analysis results:')
print(alg_test)

#RUN MANUAL LABELLING INPUTS
#manual = manual_labels('HoL')	

#with open("pickle/manual.txt", "wb") as fp:   #pickling
	#pickle.dump(manual, fp)

with open("pickle/manual.txt", "rb") as fp:   #unpickling
	manual = pickle.load(fp)

print(" ")
print('manually labelled results:')
print(manual)

#COMPARE MANUALLY LABELLED AND LEXICON BASED
lex_result = eval(lex_all,manual,3)
print(" ")
print("LEXICON RESULTS: ")
for a in lex_result:
	print(a)

#COMPARE MANUALLY LABELLED AND ML BASED
ml_result = eval(alg_test,manual,3)
print(" ")
print("MACHINE LEARNING RESULTS: ")
for b in ml_result:
	print(b)
