from util import *





class InflectionReduction:

	def __init__(self):
		self.lemmatizer = WordNetLemmatizer()
		self.stemmer = PorterStemmer()

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
		reducedText = []
		for line in text:
			# reducedText.append([self.stemmer.stem(word) for word in line])
			reducedText.append([self.lemmatizer.lemmatize(word) for word in line])

		return reducedText


