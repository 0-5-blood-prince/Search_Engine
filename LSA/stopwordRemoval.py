from util import *





class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""
		set_stopwords = set(stopwords.words('english'))
		stopwordRemovedText = []
		for sentence in text:
			flitered_sentence = [w for w in sentence if not w in set_stopwords]
			stopwordRemovedText.append(flitered_sentence)
		#Fill in code here

		return stopwordRemovedText




	