from util import *





class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""
		# we are using porter stemmer for our application
		reducedText = []
		from nltk.stem import WordNetLemmatizer
		wordnet_lemmatizer = WordNetLemmatizer()
		for line in text:
			reducedText.append([PorterStemmer().stem(word) for word in line])
			# reducedText.append([wordnet_lemmatizer.lemmatize(word) for word in line])


		return reducedText


